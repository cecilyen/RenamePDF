"""Resolve and validate replacements selected by a current AuditPDF report."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

import audit_report
import collection_mutation
import document_classification
import filename_policy
import metadata_providers
import pdf_evidence
import pdf_identity_cache as identity_cache_storage
import pdf_process
import pdf_rewrite
import publication_status
import record_matching
import report_io


CROSSREF_WORKS = "https://api.crossref.org/works"
BIORXIV_PUBS = "https://api.biorxiv.org/pubs"
UNPAYWALL_V2 = "https://api.unpaywall.org/v2"
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
MAX_PDF_DOWNLOAD_BYTES = 512 * 1024 * 1024
PDF_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_INSPECTION_TIMEOUT = 120.0
IDENTITY_CACHE_FILENAME = identity_cache_storage.CACHE_FILENAME
IDENTITY_CACHE_NAMESPACE = (
    identity_cache_storage.DEFAULT_LOCAL_CACHE_NAMESPACE
)
NON_CACHEABLE_PDF_STATUSES = pdf_rewrite.REWRITE_BLOCKING_STATUSES | {
    "pdf_processing_error",
    "pdf_processing_timeout",
}

REPORT_FIELDS = (
    "path",
    "kind",
    "replacement_action",
    "audit_report",
    "source_title",
    "source_doi",
    "pdf_status",
    "pdf_note",
    "published_doi",
    "published_title",
    "journal",
    "year",
    "confidence",
    "match_method",
    "oa_version",
    "oa_url",
    "target_name",
    "candidate_retraction_status",
    "candidate_retraction_evidence",
    "candidate_retraction_note",
    "status",
    "note",
    "archive_path",
    "target_path",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    path: Path
    kind: str
    title: str
    source_doi: str | None
    replacement_action: str = document_classification.FIND_PUBLISHED_VERSION
    audit_report: str = ""
    expected_size: int | None = None
    expected_mtime_ns: int | None = None


class AuditReportError(ValueError):
    """The audit report cannot safely authorize replacement work."""


@dataclass(frozen=True, slots=True)
class PublishedMatch:
    doi: str
    message: dict[str, Any]
    confidence: float
    method: str


@dataclass(frozen=True, slots=True)
class PublishedResolution:
    decision: record_matching.ReplacementDecision
    message: dict[str, Any] | None
    source_message: dict[str, Any] | None = None

    @property
    def match(self) -> PublishedMatch | None:
        record = self.decision.record
        if not self.decision.accepted or record is None or self.message is None:
            return None
        return PublishedMatch(
            doi=record.doi,
            message=self.message,
            confidence=self.decision.confidence,
            method=self.decision.method,
        )


_AUDIT_SELECTION_FIELDS = frozenset(
    {
        "audit_schema_version",
        "path",
        "relative_path",
        "file_size",
        "mtime_ns",
        "identity_status",
        "doi",
        "title",
        "pdf_title_hint",
        "publication_stage",
        "replacement_action",
        "replacement_eligible",
    }
)
_SUPPORTED_REPLACEMENT_ACTIONS = document_classification.AUTOMATED_REPLACEMENT_ACTIONS


def _report_integer(row: dict[str, object], field: str) -> int:
    value = str(row.get(field) or "")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise AuditReportError(
            f"Audit row has invalid {field}: {value!r}"
        ) from exc
    if parsed < 0:
        raise AuditReportError(
            f"Audit row has invalid {field}: {value!r}"
        )
    return parsed


def load_audit_candidates(
    root: Path,
    report: Path,
) -> list[Candidate]:
    """Load eligible rows only after exact path and file-state checks."""
    try:
        fieldnames, rows = report_io.read_csv(report)
    except OSError as exc:
        raise AuditReportError(f"Could not read audit report: {exc}") from exc
    missing = sorted(_AUDIT_SELECTION_FIELDS - set(fieldnames))
    if missing:
        raise AuditReportError(
            "Audit report lacks required columns: " + ", ".join(missing)
        )

    resolved_root = root.resolve()
    selected: list[Candidate] = []
    seen: set[Path] = set()
    for row_number, row in enumerate(rows, 2):
        schema = str(row.get("audit_schema_version") or "")
        if schema != audit_report.AUDIT_SCHEMA_VERSION:
            raise AuditReportError(
                f"Audit row {row_number} uses unsupported schema {schema!r}"
            )
        eligible = str(row.get("replacement_eligible") or "").casefold()
        if eligible not in {"yes", "true", "1"}:
            continue
        action = str(row.get("replacement_action") or "")
        if action not in _SUPPORTED_REPLACEMENT_ACTIONS:
            raise AuditReportError(
                f"Audit row {row_number} has unsupported eligible action: "
                f"{action or '<missing>'}"
            )

        relative = Path(str(row.get("relative_path") or ""))
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise AuditReportError(
                f"Audit row {row_number} has an invalid relative path"
            )
        unresolved_path = resolved_root / relative
        if unresolved_path.is_symlink():
            raise AuditReportError(
                f"Audit row {row_number} names a symbolic link"
            )
        path = unresolved_path.resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise AuditReportError(
                f"Audit row {row_number} escapes the collection root"
            ) from exc
        recorded_path = Path(str(row.get("path") or "")).expanduser()
        if not recorded_path.is_absolute() or recorded_path.resolve() != path:
            raise AuditReportError(
                f"Audit row {row_number} path does not match relative_path"
            )
        if path.suffix.casefold() != ".pdf" or not path.is_file():
            raise AuditReportError(
                f"Audit row {row_number} no longer names a regular PDF: "
                f"{path}"
            )
        metadata = path.stat()
        expected_size = _report_integer(row, "file_size")
        expected_mtime_ns = _report_integer(row, "mtime_ns")
        if (
            metadata.st_size != expected_size
            or metadata.st_mtime_ns != expected_mtime_ns
        ):
            raise AuditReportError(
                f"Audit row {row_number} is stale for {path}; rerun AuditPDF"
            )
        if path in seen:
            raise AuditReportError(
                f"Audit report selects the same PDF more than once: {path}"
            )
        seen.add(path)

        stage = str(row.get("publication_stage") or "")
        if action == document_classification.FIND_PUBLISHED_VERSION and stage not in {
            "preprint",
            "proceeding",
        }:
            raise AuditReportError(
                f"Audit row {row_number} has action {action} for stage "
                f"{stage or '<missing>'}"
            )
        title = str(row.get("title") or row.get("pdf_title_hint") or "")
        doi = pdf_evidence.clean_doi(str(row.get("doi") or "")) or None
        if action == document_classification.FIND_EXACT_CLEAN_COPY:
            if str(row.get("identity_status") or "") != "accepted":
                raise AuditReportError(
                    f"Audit row {row_number} requests an exact copy without "
                    "an accepted identity"
                )
            if not doi:
                raise AuditReportError(
                    f"Audit row {row_number} requests an exact copy without "
                    "a DOI"
                )
        selected.append(
            Candidate(
                path=path,
                kind=stage or "unknown",
                title=title,
                source_doi=doi,
                replacement_action=action,
                audit_report=str(report),
                expected_size=expected_size,
                expected_mtime_ns=expected_mtime_ns,
            )
        )
    return sorted(selected, key=lambda item: str(item.path).casefold())


def expected_first_author(path: Path) -> str | None:
    match = re.match(r"([A-Z][A-Za-z'-]*?[a-z])([A-Z]{1,3})_", path.name)
    return filename_policy.name_token(match.group(1)).lower() if match else None


def source_year(path: Path) -> int | None:
    years = [int(value) for value in re.findall(r"(?<!\d)(20\d{2}|19\d{2})(?!\d)", path.stem)]
    if years:
        return min(years)
    short = re.search(
        r"(?:arXiv|bioRxiv|medRxiv|Preprint|ConferencePaper|Meeting)(\d{2})(?:_|$)",
        path.stem,
        re.IGNORECASE,
    )
    if not short:
        return None
    return filename_policy.resolve_two_digit_year(int(short.group(1)))


def best_source_doi(hints: pdf_evidence.PdfEvidence) -> str | None:
    for item in hints.dois:
        doi = pdf_evidence.clean_doi(item.doi)
        if doi.lower().startswith(("10.1101/", "10.48550/arxiv.")):
            return doi
    if hints.arxiv_doi:
        return pdf_evidence.clean_doi(hints.arxiv_doi)
    return pdf_evidence.clean_doi(hints.dois[0].doi) if hints.dois else None


class ApiClient:
    def __init__(self, email: str):
        self.email = email
        self.client = httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(30, connect=15),
            headers={
                "User-Agent": f"RenamePDF-published-version/1.0 (mailto:{email})",
                "Accept": "application/json",
            },
            verify=metadata_providers.build_http_verify_context(),
        )

    def close(self) -> None:
        self.client.close()

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        for attempt in range(4):
            try:
                response = self.client.get(url, params=params)
                if response.status_code == 404:
                    return None
                if response.status_code == 429 or response.status_code >= 500:
                    time.sleep(min(8, 2 ** attempt))
                    continue
                response.raise_for_status()
                payload = response.json()
                return payload if isinstance(payload, dict) else None
            except (httpx.HTTPError, json.JSONDecodeError):
                if attempt == 3:
                    return None
                time.sleep(min(8, 2 ** attempt))
        return None


def crossref_message(client: ApiClient, doi: str) -> dict[str, Any] | None:
    payload = client.get_json(
        f"{CROSSREF_WORKS}/{urllib.parse.quote(pdf_evidence.clean_doi(doi), safe='')}"
    )
    message = payload.get("message") if payload else None
    return message if isinstance(message, dict) else None


def relation_dois(message: dict[str, Any]) -> tuple[str, ...]:
    relation = message.get("relation")
    if not isinstance(relation, dict):
        return ()
    found: list[str] = []
    for key in ("is-preprint-of", "is-version-of", "is-identical-to"):
        entries = relation.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            identifier = entry.get("id")
            if entry.get("id-type") == "doi" and isinstance(identifier, str):
                cleaned = pdf_evidence.clean_doi(identifier)
                if cleaned and not any(
                    pdf_evidence.dois_equal(cleaned, existing)
                    for existing in found
                ):
                    found.append(cleaned)
    return tuple(found)


def _author_text_matches(path: Path, authors: str) -> bool:
    expected = expected_first_author(path)
    return bool(expected and re.search(rf"\b{re.escape(expected)}\b", authors, re.IGNORECASE))


def _source_title_matches(candidate: Candidate, source_title: str, authors: str = "") -> bool:
    similarity = record_matching.title_similarity(candidate.title, source_title)
    return similarity >= 0.93 or (
        similarity >= 0.83 and _author_text_matches(candidate.path, authors)
    )


def biorxiv_published_doi(
    client: ApiClient,
    source_doi: str,
    candidate: Candidate,
) -> str | None:
    if not source_doi.lower().startswith("10.1101/"):
        return None
    preferred = "medrxiv" if "medrxiv" in candidate.path.name.lower() else "biorxiv"
    for server in (preferred, "biorxiv" if preferred == "medrxiv" else "medrxiv"):
        payload = client.get_json(
            f"{BIORXIV_PUBS}/{server}/{urllib.parse.quote(source_doi, safe='/')}/na/json"
        )
        collection = payload.get("collection") if payload else None
        if not isinstance(collection, list):
            continue
        for record in collection:
            if not isinstance(record, dict):
                continue
            preprint_title = str(record.get("preprint_title") or "")
            preprint_authors = str(record.get("preprint_authors") or "")
            if not _source_title_matches(candidate, preprint_title, preprint_authors):
                continue
            published = record.get("published_doi")
            if isinstance(published, str) and published.strip():
                return pdf_evidence.clean_doi(published)
    return None


def _crossref_title(message: dict[str, Any]) -> str:
    titles = message.get("title")
    if not isinstance(titles, list) or not titles:
        return ""
    return pdf_evidence.clean_title_candidate(str(titles[0]))


def _crossref_year(message: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "published", "issued"):
        parts = (message.get(key) or {}).get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            try:
                return int(parts[0][0])
            except (TypeError, ValueError):
                continue
    return None


def _source_work(
    candidate: Candidate,
) -> record_matching.PreliminaryWork:
    return record_matching.PreliminaryWork(
        kind=candidate.kind,
        title=candidate.title,
        doi=candidate.source_doi,
        first_author=expected_first_author(candidate.path),
        year=source_year(candidate.path),
    )


def _crossref_record(
    message: dict[str, Any],
) -> record_matching.PublishedRecord:
    def first_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list) and value:
            return str(value[0] or "").strip()
        return ""

    authors: list[str] = []
    for author in message.get("author") or []:
        if not isinstance(author, dict):
            continue
        family_value = str(author.get("family") or "").strip()
        if not family_value:
            name_parts = str(author.get("name") or "").split()
            family_value = name_parts[-1] if name_parts else ""
        family = filename_policy.name_token(
            family_value
        ).lower()
        if family and family != "unknown":
            authors.append(family)
    journal = first_text(
        message.get("short-container-title")
    ) or first_text(message.get("container-title"))
    return record_matching.PublishedRecord(
        doi=pdf_evidence.clean_doi(str(message.get("DOI") or "")),
        title=_crossref_title(message),
        year=_crossref_year(message),
        record_type=str(message.get("type") or ""),
        journal=journal,
        authors=tuple(authors),
    )


def _resolution(
    decision: record_matching.ReplacementDecision,
    messages: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    *,
    source_message: dict[str, Any] | None = None,
) -> PublishedResolution:
    message = None
    if decision.record is not None:
        message = next(
            (
                item
                for item in messages
                if pdf_evidence.dois_equal(
                    str(item.get("DOI") or ""),
                    decision.record.doi,
                )
            ),
            None,
        )
    return PublishedResolution(
        decision=decision,
        message=message,
        source_message=source_message,
    )


def journal_title_match(
    client: ApiClient,
    candidate: Candidate,
) -> PublishedResolution:
    payload = client.get_json(
        CROSSREF_WORKS,
        {
            "query.title": candidate.title,
            "filter": "type:journal-article",
            "rows": 8,
        },
    )
    items = (payload.get("message") or {}).get("items") if payload else None
    if not isinstance(items, list):
        return _resolution(
            record_matching.reject_replacement(
                "Crossref title search returned no record collection",
                method="crossref_journal_title",
            )
        )

    messages = [item for item in items if isinstance(item, dict)]
    decision = record_matching.decide_published_title(
        _source_work(candidate),
        tuple(_crossref_record(message) for message in messages),
    )
    return _resolution(decision, messages)


def canonical_published_record(
    client: ApiClient,
    doi: str,
) -> tuple[str, dict[str, Any] | None]:
    cleaned = pdf_evidence.clean_doi(doi)
    message = crossref_message(client, cleaned)
    match = re.fullmatch(r"(10\.7554/elife\.\d+)(?:\.\d+)?", cleaned, re.IGNORECASE)
    if not match:
        return cleaned, message
    canonical_doi = match.group(1)
    canonical_message = message if cleaned.lower() == canonical_doi.lower() else crossref_message(
        client, canonical_doi
    )
    if canonical_message and canonical_message.get("type") == "journal-article":
        relation = canonical_message.get("relation")
        same_as = relation.get("is-same-as") if isinstance(relation, dict) else None
        versioned_dois: list[tuple[int, str]] = []
        for entry in same_as if isinstance(same_as, list) else []:
            if not isinstance(entry, dict) or entry.get("id-type") != "doi":
                continue
            related = pdf_evidence.clean_doi(str(entry.get("id") or ""))
            version = re.fullmatch(
                rf"{re.escape(canonical_doi)}\.(\d+)",
                related,
                re.IGNORECASE,
            )
            if version:
                versioned_dois.append((int(version.group(1)), related))
        for _, versioned_doi in sorted(versioned_dois, reverse=True):
            versioned_message = crossref_message(client, versioned_doi)
            if (
                versioned_message
                and versioned_message.get("type") == "journal-article"
                and record_matching.title_similarity(
                    _crossref_title(canonical_message),
                    _crossref_title(versioned_message),
                ) >= 0.95
            ):
                return versioned_doi, versioned_message
        return canonical_doi, canonical_message
    return cleaned, message


def resolve_published(
    client: ApiClient,
    candidate: Candidate,
) -> PublishedResolution:
    source_message = None
    if candidate.source_doi:
        direct_dois: list[str] = []

        def add_direct(doi: str | None) -> None:
            if (
                doi
                and not pdf_evidence.dois_equal(doi, candidate.source_doi or "")
                and not any(
                    pdf_evidence.dois_equal(doi, existing)
                    for existing in direct_dois
                )
            ):
                direct_dois.append(pdf_evidence.clean_doi(doi))

        add_direct(
            biorxiv_published_doi(
                client,
                candidate.source_doi,
                candidate,
            )
        )
        source_message = crossref_message(client, candidate.source_doi)
        if source_message:
            source_title = _crossref_title(source_message)
            source_authors = " ".join(
                str(author.get("family") or "")
                for author in source_message.get("author", [])
                if isinstance(author, dict)
            )
            if _source_title_matches(candidate, source_title, source_authors):
                for related in relation_dois(source_message):
                    add_direct(related)

        canonical: dict[str, tuple[str, dict[str, Any] | None]] = {}
        for direct in direct_dois:
            doi, message = canonical_published_record(client, direct)
            canonical.setdefault(doi.casefold(), (doi, message))
        if len(canonical) > 1:
            return _resolution(
                record_matching.conflicting_publication_relations(
                    tuple(item[0] for item in canonical.values())
                ),
                source_message=source_message,
            )
        if canonical:
            direct, published_message = next(iter(canonical.values()))
            if not published_message:
                return _resolution(
                    record_matching.reject_replacement(
                        (
                            "direct publication relation target has no "
                            f"Crossref record: {direct}"
                        ),
                        method="direct_publication_relation",
                    ),
                    source_message=source_message,
                )
            decision = record_matching.decide_published_relation(
                _source_work(candidate),
                _crossref_record(published_message),
                source_identity_verified=True,
            )
            return _resolution(
                decision,
                [published_message],
                source_message=source_message,
            )
    resolution = journal_title_match(client, candidate)
    if source_message is None:
        return resolution
    return PublishedResolution(
        resolution.decision,
        resolution.message,
        source_message,
    )


def resolve_exact_copy(
    client: ApiClient,
    candidate: Candidate,
) -> PublishedResolution:
    """Resolve the same audited DOI, never a successor publication."""
    method = "exact_doi_copy"
    doi = pdf_evidence.clean_doi(candidate.source_doi or "")
    if not doi:
        return _resolution(
            record_matching.reject_replacement(
                "exact-copy replacement requires an audited DOI",
                method=method,
            )
        )
    message = crossref_message(client, doi)
    if not message:
        return _resolution(
            record_matching.reject_replacement(
                f"Crossref returned no record for audited DOI {doi}",
                method=method,
            )
        )
    record = _crossref_record(message)
    if not pdf_evidence.dois_equal(record.doi, doi):
        return _resolution(
            record_matching.reject_replacement(
                "Crossref record DOI differs from the audited DOI",
                method=method,
            ),
            source_message=message,
        )
    similarity = (
        record_matching.title_similarity(candidate.title, record.title)
        if candidate.title and record.title
        else None
    )
    if similarity is not None and similarity < 0.80:
        return _resolution(
            record_matching.ReplacementDecision(
                outcome="rejected",
                record=None,
                confidence=0.0,
                method=method,
                reason=(
                    "Crossref title no longer corroborates the audited "
                    f"identity ({similarity:.3f})"
                ),
                title_similarity=similarity,
            ),
            source_message=message,
        )
    decision = record_matching.ReplacementDecision(
        outcome="accepted",
        record=record,
        confidence=1.0,
        method=method,
        reason="accepted exact-copy target from the audited DOI",
        title_similarity=similarity,
    )
    return _resolution(decision, [message], source_message=message)


def crossref_licensed_pdf(
    message: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    if not message:
        return None, None
    licenses = message.get("license")
    has_open_license = any(
        isinstance(item, dict)
        and (
            "creativecommons.org/licenses/" in str(item.get("URL") or "").lower()
            or "creativecommons.org/publicdomain/" in str(item.get("URL") or "").lower()
        )
        for item in licenses or []
    )
    if not has_open_license:
        return None, None
    links = message.get("link")
    for desired, label in (("vor", "publishedVersion"), ("am", "acceptedVersion")):
        for link in links or []:
            if not isinstance(link, dict):
                continue
            url = link.get("URL")
            if (
                link.get("content-type") == "application/pdf"
                and link.get("content-version") == desired
                and isinstance(url, str)
                and url.startswith(("http://", "https://"))
            ):
                return label, url
    return None, None


def europe_pmc_pdf(client: ApiClient, doi: str) -> str | None:
    payload = client.get_json(
        EUROPE_PMC_SEARCH,
        {
            "query": f'DOI:"{doi}"',
            "format": "json",
            "pageSize": 5,
        },
    )
    results = ((payload or {}).get("resultList") or {}).get("result")
    if not isinstance(results, list):
        return None
    for result in results:
        if not isinstance(result, dict):
            continue
        pmcid = result.get("pmcid")
        if isinstance(pmcid, str) and pmcid.upper().startswith("PMC"):
            return f"https://europepmc.org/articles/{pmcid}?pdf=render"
    return None


def oa_locations(
    client: ApiClient,
    doi: str,
    message: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []

    def add(version: str | None, url: str | None) -> None:
        if version and url and all(existing_url != url for _, existing_url in found):
            found.append((version, url))

    version, url = crossref_licensed_pdf(message)
    add(version, url)
    payload = client.get_json(
        f"{UNPAYWALL_V2}/{urllib.parse.quote(doi, safe='')}",
        {"email": client.email},
    )
    locations = payload.get("oa_locations") if payload else None
    if isinstance(locations, list):
        preferred: list[tuple[int, int, str, str]] = []
        version_rank = {"publishedVersion": 0, "acceptedVersion": 1}
        for location in locations:
            if not isinstance(location, dict):
                continue
            location_version = location.get("version")
            location_url = location.get("url_for_pdf")
            if (
                location_version not in version_rank
                or not isinstance(location_url, str)
                or not location_url.startswith(("http://", "https://"))
            ):
                continue
            host_rank = 0 if location.get("host_type") == "publisher" else 1
            preferred.append(
                (version_rank[location_version], host_rank, location_version, location_url)
            )
        preferred.sort()
        for _, _, location_version, location_url in preferred:
            add(location_version, location_url)
    add("publishedVersion", europe_pmc_pdf(client, doi))
    return found


def oa_location(
    client: ApiClient,
    doi: str,
    message: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    locations = oa_locations(client, doi, message)
    return locations[0] if locations else (None, None)


def archive_original(
    root: Path, candidate: Candidate, *, verify_preserved: Callable[[], None] | None = None,
) -> Path:
    return collection_mutation.archive_source(
        root, candidate.path, verify_preserved=verify_preserved,
    )


def _download_identity(path: Path) -> tuple[int, int] | None:
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return int(info.st_dev), int(info.st_ino)


def _remove_owned_download(
    path: Path,
    identity: tuple[int, int] | None,
) -> None:
    if identity is None:
        return
    current = _download_identity(path)
    if current is None:
        if not path.exists() and not path.is_symlink():
            return
        raise RuntimeError(f"download path changed before cleanup: {path}")
    if identity is None or current != identity:
        raise RuntimeError(f"download path changed before cleanup: {path}")
    path.unlink()


def download_pdf_to_path(
    client: ApiClient,
    url: str,
    destination: Path,
    *,
    referer: str,
    max_bytes: int = MAX_PDF_DOWNLOAD_BYTES,
) -> Path:
    """Stream one PDF into an exclusively created, size-bounded path."""
    if max_bytes < 5:
        raise ValueError("PDF download limit must be at least 5 bytes")
    headers = {
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.5",
        "Referer": referer,
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/136.0 Safari/537.36"
        ),
    }
    for attempt in range(4):
        destination_identity: tuple[int, int] | None = None
        descriptor: int | None = None
        try:
            with client.client.stream("GET", url, headers=headers) as response:
                status_code = int(getattr(response, "status_code", 200))
                if status_code == 404:
                    raise RuntimeError(f"PDF URL returned HTTP 404: {url}")
                if status_code == 429 or status_code >= 500:
                    if attempt == 3:
                        response.raise_for_status()
                    time.sleep(min(8, 2 ** attempt))
                    continue
                response.raise_for_status()
                length_text = response.headers.get("Content-Length", "")
                try:
                    declared_length = int(length_text)
                except (TypeError, ValueError):
                    declared_length = 0
                if declared_length > max_bytes:
                    raise RuntimeError(
                        "PDF download exceeded the configured limit before transfer: "
                        f"{declared_length} > {max_bytes} bytes"
                    )

                descriptor = os.open(
                    destination,
                    os.O_CREAT
                    | os.O_EXCL
                    | os.O_WRONLY
                    | getattr(os, "O_BINARY", 0),
                    0o600,
                )
                opened = os.fstat(descriptor)
                destination_identity = (int(opened.st_dev), int(opened.st_ino))
                prefix = bytearray()
                downloaded = 0
                with os.fdopen(descriptor, "wb") as output:
                    descriptor = None
                    for chunk in response.iter_bytes(
                        chunk_size=PDF_DOWNLOAD_CHUNK_BYTES
                    ):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise RuntimeError(
                                "PDF download exceeded the configured limit during "
                                f"transfer: {downloaded} > {max_bytes} bytes"
                            )
                        if len(prefix) < 5:
                            prefix.extend(chunk[: 5 - len(prefix)])
                            if len(prefix) == 5 and prefix != b"%PDF-":
                                raise RuntimeError("PDF URL did not return a PDF")
                        output.write(chunk)
                    if prefix != b"%PDF-":
                        raise RuntimeError("PDF URL did not return a PDF")
                    output.flush()
                    os.fsync(output.fileno())
                if _download_identity(destination) != destination_identity:
                    raise RuntimeError(
                        f"download path changed during transfer: {destination}"
                    )
                return destination
        except httpx.HTTPError:
            if descriptor is not None:
                os.close(descriptor)
            _remove_owned_download(destination, destination_identity)
            if attempt == 3:
                raise
            time.sleep(min(8, 2 ** attempt))
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            _remove_owned_download(destination, destination_identity)
            raise
    raise RuntimeError(f"PDF download failed after retries: {url}")


def _download_validated_pdf_evidence(
    client: ApiClient,
    candidate: Candidate,
    match: PublishedMatch,
    url: str,
    work_dir: Path,
) -> tuple[Path, pdf_evidence.PdfEvidence]:
    raw = work_dir / "download.pdf"
    normalized = work_dir / "normalized.pdf"
    download_pdf_to_path(
        client,
        url,
        raw,
        referer=f"https://doi.org/{match.doi}",
    )
    try:
        raw_evidence = pdf_evidence.inspect_pdf(raw, max_text_pages=3)
        if raw_evidence.pdf_status in pdf_rewrite.REWRITE_BLOCKING_STATUSES:
            raise ValueError(
                f"{raw_evidence.pdf_status}: {raw_evidence.pdf_note}"
            )
        pdf_rewrite.write_validated_copy(
            raw,
            normalized,
            remove_page_indices=tuple(
                cover.page_index for cover in raw_evidence.inserted_cover_pages
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            "PyMuPDF could not normalize and validate the downloaded PDF: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    hints = pdf_evidence.inspect_pdf(normalized, max_text_pages=3)
    rejection = record_matching.replacement_pdf_rejection(
        hints, match.doi, _crossref_title(match.message),
        require_text=candidate.replacement_action == document_classification.FIND_EXACT_CLEAN_COPY,
    )
    if rejection:
        raise RuntimeError(rejection)
    return normalized, hints


def download_validated_pdf(
    client: ApiClient,
    candidate: Candidate,
    match: PublishedMatch,
    url: str,
    work_dir: Path,
) -> Path:
    """Download and validate a published PDF, returning its local path."""
    normalized, _evidence = _download_validated_pdf_evidence(
        client,
        candidate,
        match,
        url,
        work_dir,
    )
    return normalized


def _processing_error_evidence(
    status: str,
    note: str,
) -> pdf_evidence.PdfEvidence:
    return pdf_evidence.PdfEvidence(
        dois=[],
        title=None,
        pii=None,
        arxiv_doi=None,
        pdf_status=status,
        pdf_note=note,
    )


def _disable_identity_cache(
    cache: identity_cache_storage.IdentityCache,
    action: str,
    error: Exception,
) -> None:
    cache.disable(
        f"Disabled after {action} failed: {type(error).__name__}: {error}"
    )


def _store_identity_evidence(
    cache: identity_cache_storage.IdentityCache | None,
    path: Path,
    evidence: pdf_evidence.PdfEvidence,
) -> None:
    if cache is None or not cache.usable:
        return
    try:
        if evidence.pdf_status in NON_CACHEABLE_PDF_STATUSES:
            cache.discard(path)
        else:
            cache.put(
                path,
                pdf_evidence.evidence_to_cache_payload(evidence),
            )
    except Exception as exc:
        _disable_identity_cache(cache, "cache write", exc)


def _relocate_source_evidence(
    cache: identity_cache_storage.IdentityCache | None,
    source: Path,
    archive_path: str,
    evidence: pdf_evidence.PdfEvidence,
) -> None:
    if cache is None or not cache.usable or not archive_path:
        return
    _store_identity_evidence(cache, Path(archive_path), evidence)
    if not cache.usable:
        return
    try:
        cache.discard(source)
    except Exception as exc:
        _disable_identity_cache(cache, "source cache relocation", exc)


def _fresh_replacement_labels(candidate: Candidate, evidence: pdf_evidence.PdfEvidence):
    classification = document_classification.classify_failed_pdf(
        candidate.path, title=evidence.title or "", text_samples=evidence.text_samples,
        text_markers=evidence.classification_markers,
        doi_candidates=[(item.doi, item.source, item.score) for item in evidence.dois],
        arxiv_doi=evidence.arxiv_doi or "",
    )
    return document_classification.classify_audit(
        path=candidate.path,
        status="crossref_resolved" if candidate.source_doi else "no_confident_metadata_found",
        doi=candidate.source_doi or "", arxiv_doi=evidence.arxiv_doi or "",
        title=candidate.title, pdf_title_hint=evidence.title or "",
        pdf_status=evidence.pdf_status, failure_category=classification.category,
    )


def inspect_candidate_sources(
    paths: Iterable[Path],
    *,
    pdf_password: str = "",
    workers: int = pdf_process.DEFAULT_MAX_WORKERS,
    inspection_timeout: float = DEFAULT_INSPECTION_TIMEOUT,
    progress: Callable[
        [int, int, pdf_evidence.PdfEvidence],
        None,
    ]
    | None = None,
) -> dict[Path, pdf_evidence.PdfEvidence]:
    """Freshly inspect audit-selected paths in one reusable spawned pool.

    Workers receive pathnames, open their own PDFs, and return serializable
    evidence through ``pdf_process``. Cache hits never authorize replacement.
    """
    ordered_paths = list(paths)
    evidence_by_path: dict[Path, pdf_evidence.PdfEvidence] = {}
    worker_completed = 0

    def accept_worker_outcome(
        completed: int,
        _total: int,
        outcome: pdf_process.PdfBatchInspectionOutcome,
    ) -> None:
        nonlocal worker_completed
        worker_completed = completed
        evidence = outcome.evidence or _processing_error_evidence(
            outcome.error_status or "pdf_processing_error",
            outcome.error_note or "Inspection worker returned no evidence",
        )
        evidence_by_path[outcome.path] = evidence
        if progress is not None:
            progress(completed, len(ordered_paths), evidence)

    if ordered_paths:
        try:
            pdf_process.inspect_pdf_batch(
                ordered_paths,
                pdf_password=pdf_password,
                max_workers=workers,
                timeout=inspection_timeout,
                progress=accept_worker_outcome,
            )
        except Exception as exc:
            remaining = [
                path for path in ordered_paths if path not in evidence_by_path
            ]
            for offset, path in enumerate(remaining, start=1):
                evidence = _processing_error_evidence(
                    "pdf_processing_error",
                    "Inspection pool failed: "
                    f"{type(exc).__name__}: {exc}",
                )
                evidence_by_path[path] = evidence
                if progress is not None:
                    progress(
                        worker_completed + offset,
                        len(ordered_paths),
                        evidence,
                    )

    return {path: evidence_by_path[path] for path in ordered_paths}


def make_report_row(candidate: Candidate) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(REPORT_FIELDS, "")
    row.update(
        {
            "path": str(candidate.path),
            "kind": candidate.kind,
            "replacement_action": candidate.replacement_action,
            "audit_report": candidate.audit_report,
            "source_title": candidate.title,
            "source_doi": candidate.source_doi or "",
            "status": "no_confident_published_match",
        }
    )
    return row


def _ensure_candidate_state(candidate: Candidate) -> None:
    """Reject a source that changed after AuditPDF selected it."""
    if (
        candidate.expected_size is None
        or candidate.expected_mtime_ns is None
    ):
        return
    try:
        metadata = os.lstat(candidate.path)
    except OSError as exc:
        raise RuntimeError(
            f"Audited source is no longer readable: {candidate.path}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(
            f"Audited source is no longer a regular file: {candidate.path}"
        )
    if (
        metadata.st_size != candidate.expected_size
        or metadata.st_mtime_ns != candidate.expected_mtime_ns
    ):
        raise RuntimeError(
            f"Audited source changed; rerun AuditPDF: {candidate.path}"
        )


def _existing_target_state(path: Path) -> tuple[int, int, int, int, int]:
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError("Existing target is not a regular PDF")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def process_candidate(
    root: Path,
    client: ApiClient,
    candidate: Candidate,
    commit: bool,
    identity_cache: identity_cache_storage.IdentityCache | None = None,
) -> dict[str, Any]:
    _ensure_candidate_state(candidate)
    row = make_report_row(candidate)
    resolution = (
        resolve_exact_copy(client, candidate)
        if candidate.replacement_action == document_classification.FIND_EXACT_CLEAN_COPY
        else resolve_published(client, candidate)
    )
    match = resolution.match
    if not match:
        row["status"] = (
            "ambiguous_published_match"
            if resolution.decision.outcome == "ambiguous"
            else "no_confident_published_match"
        )
        row["match_method"] = resolution.decision.method
        row["note"] = resolution.decision.reason
        return row

    row["published_doi"] = match.doi
    formal_update = publication_status.assess_retractions(
        crossref_message=match.message,
        crossref_checked=True,
        crossref_expected=True,
    )
    row.update(
        {
            "candidate_retraction_status": formal_update.status,
            "candidate_retraction_evidence": formal_update.evidence_text,
            "candidate_retraction_note": formal_update.note,
        }
    )
    if formal_update.status != "no_retraction_signal":
        row["status"] = "candidate_formal_update_requires_review"
        row["match_method"] = match.method
        row["note"] = formal_update.note
        return row

    meta = metadata_providers.parse_crossref_message(match.message)
    if not meta:
        row["status"] = "published_metadata_incomplete"
        row["note"] = "Published record lacks filename metadata"
        return row

    target_name = filename_policy.make_filename(meta)
    target = root / target_name
    locations = oa_locations(client, match.doi, match.message)
    version, url = locations[0] if locations else (None, None)
    row.update(
        {
            "published_doi": match.doi,
            "published_title": meta.title,
            "journal": meta.journal_abbrev or "",
            "year": meta.year,
            "confidence": f"{match.confidence:.2f}",
            "match_method": match.method,
            "oa_version": version or "",
            "oa_url": url or "",
            "target_name": target_name,
            "note": resolution.decision.reason,
        }
    )
    if os.path.lexists(target) and target != candidate.path:
        row["target_path"] = str(target)
        try:
            target_state = _existing_target_state(target)
            evidence = pdf_evidence.inspect_pdf(target, max_text_pages=3)
            rejection = record_matching.replacement_pdf_rejection(
                evidence, match.doi, meta.title,
                require_text=candidate.replacement_action == document_classification.FIND_EXACT_CLEAN_COPY,
            )
            if rejection:
                raise RuntimeError(rejection)
            if _existing_target_state(target) != target_state:
                raise RuntimeError("Existing target changed during inspection")
            if commit:
                _ensure_candidate_state(candidate)
                def verify_target() -> None:
                    if _existing_target_state(target) != target_state:
                        raise RuntimeError("Existing target changed before or during archiving")

                row["archive_path"] = str(archive_original(
                    root, candidate, verify_preserved=verify_target,
                ))
                _store_identity_evidence(identity_cache, target, evidence)
        except (OSError, RuntimeError, ValueError) as exc:
            row["status"] = "existing_target_requires_review"
            row["note"] = str(exc)
            return row
        row["status"] = "published_copy_already_present"
        return row
    if not url:
        row["status"] = "published_record_no_eligible_oa_pdf"
        row["note"] = (
            "No eligible publishedVersion or acceptedVersion PDF was found"
        )
        return row

    row["status"] = "published_oa_available"
    if not commit:
        return row

    errors: list[str] = []
    normalized: Path | None = None
    normalized_evidence: pdf_evidence.PdfEvidence | None = None
    used_version = version
    used_url = url
    temporary = tempfile.TemporaryDirectory(prefix=".published-", dir=root)
    try:
        work_dir = Path(temporary.name)
        for location_version, location_url in locations:
            try:
                normalized, normalized_evidence = (
                    _download_validated_pdf_evidence(
                        client,
                        candidate,
                        match,
                        location_url,
                        work_dir,
                    )
                )
                used_version = location_version
                used_url = location_url
                break
            except Exception as exc:
                errors.append(f"{location_url}: {exc}")
                for temporary_pdf in work_dir.glob("*.pdf"):
                    temporary_pdf.unlink(missing_ok=True)
        if normalized is None:
            raise RuntimeError("All open-access PDF locations failed: " + " | ".join(errors))
        _ensure_candidate_state(candidate)
        outcome = collection_mutation.install_replacement(
            root,
            candidate.path,
            normalized,
            target,
        )
        archive = outcome.archive
        target = outcome.target
        if normalized_evidence is not None:
            _store_identity_evidence(
                identity_cache,
                target,
                normalized_evidence,
            )
    finally:
        temporary.cleanup()
    row["status"] = "published_oa_installed"
    row["oa_version"] = used_version
    row["oa_url"] = used_url
    row["archive_path"] = str(archive)
    row["target_path"] = str(target)
    return row


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    report_io.write_csv_atomic(
        path,
        rows,
        fieldnames=REPORT_FIELDS,
        extrasaction="ignore",
    )


def _open_identity_cache(
    root: Path,
    pdf_password: str,
) -> identity_cache_storage.IdentityCache | None:
    if pdf_password:
        print(
            "PDF identity cache disabled because "
            "RENAMEPDF_PDF_PASSWORD is set."
        )
        return None
    candidate = identity_cache_storage.IdentityCache(
        root,
        extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION,
        snapshot_filename=IDENTITY_CACHE_FILENAME,
        local_cache_namespace=IDENTITY_CACHE_NAMESPACE,
    )
    try:
        candidate.open()
    except Exception as exc:
        try:
            candidate.close()
        except Exception:
            pass
        print(
            "PDF identity cache unavailable; continuing without it: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    print(f"PDF identity cache (local): {candidate.local_path}")
    for note in candidate.notes:
        print(f"PDF identity cache: {note}")
    return candidate


def _publish_identity_cache(
    cache: identity_cache_storage.IdentityCache,
) -> None:
    if not cache.usable:
        print(
            "PDF identity cache snapshot was not published: "
            f"{cache.disabled_reason}"
        )
        return
    try:
        cache.prune_missing()
        snapshot = cache.publish()
    except Exception as exc:
        print(
            "PDF identity cache snapshot was not published: "
            f"{type(exc).__name__}: {exc}"
        )
        return
    print(
        "PDF identity cache: "
        f"{cache.hit_count} hit(s), "
        f"{cache.miss_count} miss(es), "
        f"{cache.write_count} write(s); "
        f"snapshot: {snapshot}"
    )


@dataclass(frozen=True, slots=True)
class ReplacementOptions:
    folder: Path
    email: str
    commit: bool = False
    audit_report: Path | None = None
    report: Path | None = None


def run(options: ReplacementOptions) -> int:
    """Resolve reviewed audit selections and optionally install replacements."""
    root = options.folder.expanduser().resolve()
    if not root.is_dir():
        print(f"Folder not found or not a directory: {root}")
        return 1
    for recover in (
        collection_mutation.recover_interrupted_replacements,
        pdf_rewrite.recover_interrupted_rewrites,
    ):
        recovery = recover(root, commit=options.commit)
        for outcome in recovery:
            print(f"Recovery {outcome.status}: {outcome.note}")
        if any(outcome.blocks_run for outcome in recovery):
            print("Recovery requires review or --commit; run stopped.")
            return 2

    report = (
        options.report or root / "0_replace_results.csv"
    ).expanduser().resolve()
    audit_report_path = (
        options.audit_report or root / audit_report.DEFAULT_REPORT_NAME
    ).expanduser().resolve()
    if audit_report_path == report:
        print("Audit and replacement reports must use different paths.")
        return 1
    try:
        selected = load_audit_candidates(root, audit_report_path)
    except AuditReportError as exc:
        print(f"Audit report rejected: {exc}")
        return 1
    print(f"Candidates: {len(selected)}")

    pdf_password = os.environ.get("RENAMEPDF_PDF_PASSWORD", "")
    identity_cache = _open_identity_cache(root, pdf_password)
    try:
        def print_inspection_progress(
            completed: int,
            total: int,
            evidence: pdf_evidence.PdfEvidence,
        ) -> None:
            is_error = evidence.pdf_status in NON_CACHEABLE_PDF_STATUSES
            if completed == total or completed % 10 == 0 or is_error:
                detail = f"; {evidence.pdf_status}" if is_error else ""
                print(
                    f"Source inspection: {completed}/{total}{detail}",
                    flush=True,
                )

        try:
            source_evidence = inspect_candidate_sources(
                (candidate.path for candidate in selected),
                pdf_password=pdf_password,
                progress=print_inspection_progress,
            )

            rows: list[dict[str, Any]] = []
            if selected:
                client = ApiClient(options.email)
                try:
                    for index, selection in enumerate(selected, 1):
                        candidate = selection
                        path = selection.path
                        hints = source_evidence[path]
                        try:
                            _store_identity_evidence(
                                identity_cache,
                                path,
                                hints,
                            )
                            source_doi = best_source_doi(hints)
                            if (
                                source_doi
                                and selection.source_doi
                                and not pdf_evidence.dois_equal(
                                    source_doi,
                                    selection.source_doi,
                                )
                            ):
                                raise RuntimeError(
                                    "Fresh PDF inspection conflicts with the "
                                    "audited source DOI"
                                )
                            if (
                                hints.title
                                and selection.title
                                and record_matching.title_similarity(
                                    hints.title,
                                    selection.title,
                                ) < 0.70
                            ):
                                raise RuntimeError(
                                    "Fresh PDF inspection conflicts with the "
                                    "audited title"
                                )
                            candidate = Candidate(
                                path=path,
                                kind=selection.kind,
                                title=hints.title or selection.title,
                                source_doi=(
                                    source_doi
                                    or selection.source_doi
                                ),
                                replacement_action=(
                                    selection.replacement_action
                                ),
                                audit_report=selection.audit_report,
                                expected_size=selection.expected_size,
                                expected_mtime_ns=(
                                    selection.expected_mtime_ns
                                ),
                            )
                            row = make_report_row(candidate)
                            row["pdf_status"] = hints.pdf_status
                            row["pdf_note"] = hints.pdf_note
                            if hints.pdf_status in NON_CACHEABLE_PDF_STATUSES:
                                row["status"] = hints.pdf_status
                                row["note"] = hints.pdf_note
                            elif hints.needs_rewrite:
                                row["status"] = "source_normalization_required"
                                row["note"] = (
                                    "Normalize the source, then rerun AuditPDF "
                                    "before replacement"
                                )
                            elif (
                                not (fresh_labels := _fresh_replacement_labels(candidate, hints)).replacement_eligible
                                or fresh_labels.replacement_action != candidate.replacement_action
                            ):
                                row["status"] = "replacement_not_eligible_fresh_evidence"
                                row["note"] = (
                                    "Fresh source evidence does not authorize the audited action: "
                                    + fresh_labels.replacement_reason
                                )
                            elif (
                                not candidate.title
                                and candidate.replacement_action
                                != document_classification.FIND_EXACT_CLEAN_COPY
                            ):
                                row["status"] = "no_usable_source_title"
                                row["note"] = (
                                    "Could not extract a reliable title"
                                )
                            else:
                                row = process_candidate(
                                    root,
                                    client,
                                    candidate,
                                    options.commit,
                                    identity_cache,
                                )
                                row["pdf_status"] = hints.pdf_status
                                row["pdf_note"] = hints.pdf_note
                                if options.commit and row["archive_path"]:
                                    _relocate_source_evidence(
                                        identity_cache,
                                        path,
                                        str(row["archive_path"]),
                                        hints,
                                    )
                        except Exception as exc:
                            row = make_report_row(candidate)
                            row["status"] = f"error:{type(exc).__name__}"
                            row["note"] = str(exc)
                        rows.append(row)
                        print(
                            f"[{index}/{len(selected)}] "
                            f"{row['status']} - {path.name}"
                        )
                finally:
                    client.close()
        except KeyboardInterrupt:
            print(
                "\nExecution interrupted; the existing replacement report "
                "was not replaced."
            )
            return 130

        try:
            write_report(report, rows)
        except Exception as exc:
            print(
                "Replacement report was not written: "
                f"{type(exc).__name__}: {exc}"
            )
            return 1
        print(f"Report: {report}")
        if identity_cache is not None:
            _publish_identity_cache(identity_cache)
        return 0
    finally:
        if identity_cache is not None:
            try:
                identity_cache.close()
            except Exception as exc:
                print(
                    "PDF identity cache close warning: "
                    f"{type(exc).__name__}: {exc}"
                )
