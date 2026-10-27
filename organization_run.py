"""Two-phase organization of a scientific PDF collection."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Protocol, TypeVar

import httpx

import audit_report
import collection_mutation
import document_classification
import filename_policy
import metadata_providers
import pdf_evidence
import file_metadata
from concurrent.futures.process import BrokenProcessPool
import pdf_identity_cache as identity_cache_storage
import pdf_process
import pdf_rewrite
import publication_audit
import record_matching
from record_matching import ResolvedRecord

__all__ = [
    "OrganizationOutcome",
    "OrganizationRunError",
    "RunOptions",
    "retry_incomplete_audits",
    "run",
]


PDF_WORKER_CONCURRENCY = 10
METADATA_CONCURRENCY = 5
BLOCKING_PDF_STATUSES = pdf_rewrite.REWRITE_BLOCKING_STATUSES
RESOLVED_STATUSES = {
    "pubmed_resolved",
    "crossref_resolved",
    "datacite_resolved",
    "pubmed_title_resolved",
    "crossref_title_resolved",
    "datacite_title_resolved",
}
CSV_FIELDS = (
    "file",
    "new_name",
    "status",
    "doi",
    "pii",
    "arxiv_doi",
    "pmid",
    "title",
    "pdf_title_hint",
    "year",
    "journal_abbrev",
    "first_author",
    "corresponding_author",
    "match_source",
    "confidence",
    *publication_audit.compact_report_field_names(),
    "note",
    "pdf_status",
    "pdf_note",
    "failure_category",
    "classification_note",
    "target_path",
)


class OrganizationRunError(RuntimeError):
    """The organization run could not be initialized or completed."""


class CrossrefAlternativeLookup(Protocol):
    async def records_by_alternative_id(
        self,
        alternative_id: str,
    ) -> metadata_providers.ProviderLookup: ...


class PubMedMemory(Protocol):
    def remember_doi_pmid(self, doi: str, pmid: str) -> None: ...


class ProviderSet(Protocol):
    pubmed: PubMedMemory
    crossref: CrossrefAlternativeLookup

    async def records_by_doi(
        self,
        doi: str,
    ) -> metadata_providers.ProviderLookup: ...

    async def records_by_title(
        self,
        title: str,
    ) -> metadata_providers.ProviderLookup: ...


@dataclass(frozen=True, slots=True)
class RunOptions:
    root: Path
    commit: bool = False
    api_key: str = ""
    email: str = ""
    newer_than: float | None = None
    retry_failed: bool = False
    pdf_password: str = ""
    scite_api_key: str = ""
    include_failed: bool = False
    inspection_timeout: float = pdf_process.DEFAULT_INSPECTION_TIMEOUT

    def __post_init__(self) -> None:
        object.__setattr__(self, "inspection_timeout", pdf_process.inspection_timeout_seconds(self.inspection_timeout))


@dataclass(frozen=True, slots=True)
class OrganizationOutcome:
    source: Path
    status: str = "no_doi_or_title_found"
    new_name: str = ""
    doi: str | None = None
    pii: str = ""
    arxiv_doi: str = ""
    pmid: str | None = None
    title: str = ""
    pdf_title_hint: str = ""
    year: int | str = ""
    journal_abbrev: str = ""
    first_author: str = ""
    corresponding_author: str = ""
    match_source: str = ""
    confidence: float | None = None
    retraction_status: str = "identity_unresolved"
    retraction_evidence: str = ""
    retraction_note: str = ""
    commentary_status: str = "identity_unresolved"
    pubmed_comment_links: str = ""
    scite_total: int | str = ""
    scite_supporting: int | str = ""
    scite_contradicting: int | str = ""
    scite_mentioning: int | str = ""
    scite_unclassified: int | str = ""
    scite_citing_publications: int | str = ""
    commentary_note: str = ""
    audit_checked_at: str = ""
    note: str = ""
    pdf_status: str = "ok"
    pdf_note: str = ""
    failure_category: str = ""
    classification_note: str = ""
    target_path: Path | None = None
    source_token: file_metadata.FileToken | None = None
    placement_blocked: bool = False

    def to_legacy_dict(self) -> dict[str, object]:
        return {
            "file": str(self.source),
            "new_name": self.new_name or self.source.name,
            "status": self.status,
            "doi": self.doi,
            "pii": self.pii,
            "arxiv_doi": self.arxiv_doi,
            "pmid": self.pmid,
            "title": self.title,
            "pdf_title_hint": self.pdf_title_hint,
            "year": self.year,
            "journal_abbrev": self.journal_abbrev,
            "first_author": self.first_author,
            "corresponding_author": self.corresponding_author,
            "match_source": self.match_source,
            "confidence": (
                f"{self.confidence:.2f}"
                if self.confidence is not None
                else ""
            ),
            **publication_audit.compact_report_values(
                {
                    "retraction_status": self.retraction_status,
                    "retraction_evidence": self.retraction_evidence,
                    "retraction_note": self.retraction_note,
                    "commentary_status": self.commentary_status,
                    "pubmed_comment_links": self.pubmed_comment_links,
                    "scite_total": self.scite_total,
                    "scite_supporting": self.scite_supporting,
                    "scite_contradicting": self.scite_contradicting,
                    "scite_mentioning": self.scite_mentioning,
                    "scite_unclassified": self.scite_unclassified,
                    "scite_citing_publications": (
                        self.scite_citing_publications
                    ),
                    "commentary_note": self.commentary_note,
                    "audit_checked_at": self.audit_checked_at,
                }
            ),
            "note": self.note,
            "pdf_status": self.pdf_status,
            "pdf_note": self.pdf_note,
            "failure_category": self.failure_category,
            "classification_note": self.classification_note,
            "target_path": str(self.target_path) if self.target_path else "",
        }


@dataclass(frozen=True, slots=True)
class _Analysis:
    outcome: OrganizationOutcome
    evidence: pdf_evidence.PdfEvidence | None = None
    requires_normalization: bool = False
    remove_page_indices: tuple[int, ...] = ()
    compress_if_large: bool = False

    @property
    def needs_rewrite(self) -> bool:
        return self.requires_normalization or self.compress_if_large


ProgressAdapter = Callable[[int, int, OrganizationOutcome], None]
T = TypeVar("T")


def _append_note(outcome: OrganizationOutcome, message: str) -> OrganizationOutcome:
    note = outcome.note.strip()
    if message in note:
        return outcome
    return replace(
        outcome,
        note=f"{note}; {message}" if note else message,
    )


def _mark_duplicate(
    outcome: OrganizationOutcome,
    message: str,
) -> OrganizationOutcome:
    status = outcome.status
    if not status.endswith("_duplicate"):
        status = f"{status}_duplicate"
    return _append_note(replace(outcome, status=status), message)


def _is_resolved_status(status: str) -> bool:
    return status in RESOLVED_STATUSES


def _base_outcome(
    pdf_path: Path,
    evidence: pdf_evidence.PdfEvidence,
) -> OrganizationOutcome:
    classification = document_classification.classify_failed_pdf(
        pdf_path,
        title=evidence.title or "",
        text_samples=evidence.text_samples,
        text_markers=evidence.classification_markers,
        doi_candidates=[
            (candidate.doi, candidate.source, candidate.score)
            for candidate in evidence.dois
        ],
        arxiv_doi=evidence.arxiv_doi or "",
    )
    return OrganizationOutcome(
        source=pdf_path,
        new_name=pdf_path.name,
        pii=evidence.pii or "",
        arxiv_doi=evidence.arxiv_doi or "",
        pdf_title_hint=evidence.title or "",
        pdf_status=evidence.pdf_status,
        pdf_note=evidence.pdf_note,
        failure_category=classification.category,
        classification_note=classification.reason,
        source_token=evidence.source_token,
    )


def _resolved_outcome(
    outcome: OrganizationOutcome,
    resolved: ResolvedRecord,
) -> OrganizationOutcome:
    meta = resolved.meta
    return replace(
        outcome,
        status=resolved.status,
        doi=meta.doi,
        pmid=resolved.pmid,
        title=meta.title,
        year=meta.year,
        journal_abbrev=meta.journal_abbrev or "",
        first_author=f"{meta.first_author.family},{meta.first_author.given}",
        corresponding_author=(
            meta.corresponding_author.family
            if meta.corresponding_author
            else ""
        ),
        new_name=filename_policy.make_filename(
            meta,
            is_arxiv=bool(outcome.arxiv_doi),
        ),
        match_source=resolved.source,
        confidence=resolved.confidence,
        retraction_status="not_checked",
        commentary_status="not_checked",
        note=resolved.reason,
        failure_category="",
        classification_note="",
    )


async def _resolve_doi(
    providers: ProviderSet,
    candidate: pdf_evidence.DoiCandidate,
    title_hint: str | None,
) -> record_matching.MatchDecision:
    lookup = await providers.records_by_doi(candidate.doi)
    decision = record_matching.decide_doi_match(
        candidate,
        title_hint,
        list(lookup.records),
    )
    if (
        decision.resolved is not None
        and decision.resolved.provider == "pubmed"
        and decision.resolved.pmid
        and decision.resolved.meta.doi
        and pdf_evidence.dois_equal(
            candidate.doi,
            decision.resolved.meta.doi,
        )
    ):
        providers.pubmed.remember_doi_pmid(
            candidate.doi,
            decision.resolved.pmid,
        )
    if decision.outcome == "accepted" or not lookup.notes:
        return decision
    return record_matching.MatchDecision(
        decision.outcome,
        decision.resolved,
        "; ".join([decision.reason, *lookup.notes[:3]]),
    )


async def _resolve_pii(
    providers: ProviderSet,
    pii: str | None,
    title_hint: str | None,
) -> record_matching.MatchDecision:
    if not pii:
        return record_matching.MatchDecision("rejected", None, "")
    lookup = await providers.crossref.records_by_alternative_id(pii)
    decision = record_matching.decide_pii_match(
        pii,
        title_hint,
        list(lookup.records),
    )
    if decision.outcome == "accepted" or not lookup.notes:
        return decision
    return record_matching.MatchDecision(
        decision.outcome,
        decision.resolved,
        "; ".join([decision.reason, *lookup.notes[:3]]),
    )


async def _resolve_title(
    providers: ProviderSet,
    title_hint: str | None,
    filename_hint: str,
) -> record_matching.MatchDecision:
    if not title_hint or not pdf_evidence.looks_like_title(title_hint):
        return record_matching.MatchDecision(
            "rejected",
            None,
            "No usable PDF title was available for title search",
        )
    lookup = await providers.records_by_title(title_hint)
    return record_matching.decide_title_match(
        title_hint,
        filename_hint,
        list(lookup.records),
    )


def _disable_identity_cache(
    cache: identity_cache_storage.IdentityCache,
    action: str,
    error: Exception,
) -> None:
    cache.disable(
        f"Disabled after {action} failed: {type(error).__name__}: {error}"
    )


async def _inspect_pdf_evidence(
    pdf_path: Path,
    pdf_password: str,
    pdf_runner: pdf_process.PdfProcessRunner,
    identity_cache: identity_cache_storage.IdentityCache | None,
) -> pdf_evidence.PdfEvidence:
    before = file_metadata.file_token(pdf_path)
    cache = (
        identity_cache
        if identity_cache is not None
        and identity_cache.usable
        and not pdf_password
        else None
    )
    payload: dict[str, object] | None = None
    if cache is not None:
        try:
            payload = cache.get(pdf_path)
        except Exception as exc:
            _disable_identity_cache(cache, "cache read", exc)
            cache = None
    if payload is not None and cache is not None:
        try:
            evidence = pdf_evidence.evidence_from_cache_payload(payload)
            if evidence.pdf_status in BLOCKING_PDF_STATUSES:
                raise ValueError(
                    "blocking PDF evidence must not be reused from cache"
                )
        except (KeyError, TypeError, ValueError):
            try:
                cache.discard(pdf_path)
            except Exception as exc:
                _disable_identity_cache(
                    cache,
                    "invalid-entry removal",
                    exc,
                )
            payload = None
        else:
            if file_metadata.file_token(pdf_path) == before:
                return replace(evidence, source_token=before)

    evidence = await pdf_runner.inspect(
        pdf_path,
        pdf_password=pdf_password,
    )
    if file_metadata.file_token(pdf_path) != before:
        return pdf_evidence.PdfEvidence(
            dois=[], title=None, pii=None, arxiv_doi=None,
            pdf_status="pdf_source_changed", pdf_note="Source changed during inspection",
        )
    evidence = replace(evidence, source_token=before)
    if cache is not None and cache.usable:
        try:
            if evidence.pdf_status in BLOCKING_PDF_STATUSES:
                cache.discard(pdf_path)
            else:
                cache.put(
                    pdf_path,
                    pdf_evidence.evidence_to_cache_payload(evidence),
                )
        except Exception as exc:
            _disable_identity_cache(cache, "cache write", exc)
    return evidence


async def _analyze_pdf(
    pdf_path: Path,
    pdf_password: str,
    providers: ProviderSet,
    pdf_runner: pdf_process.PdfProcessRunner,
    identity_cache: identity_cache_storage.IdentityCache | None = None,
) -> _Analysis:
    evidence = await _inspect_pdf_evidence(
        pdf_path,
        pdf_password,
        pdf_runner,
        identity_cache,
    )
    outcome = _base_outcome(pdf_path, evidence)
    compress_if_large = pdf_rewrite.exceeds_lossless_compression_threshold(
        pdf_path
    )
    if compress_if_large and outcome.pdf_status == "ok":
        outcome = replace(
            outcome,
            pdf_status="pdf_compression_needed",
            pdf_note=(
                "PDF is larger than 3 MB and is a lossless-compression "
                "candidate; compression is attempted only with --commit"
            ),
        )
    analysis = _Analysis(
        outcome=outcome,
        evidence=evidence,
        requires_normalization=evidence.needs_rewrite,
        remove_page_indices=tuple(
            item.page_index for item in evidence.inserted_cover_pages
        ),
        compress_if_large=compress_if_large,
    )

    if evidence.pdf_status in BLOCKING_PDF_STATUSES:
        return replace(
            analysis,
            outcome=replace(
                outcome,
                status=evidence.pdf_status,
                note=evidence.pdf_note,
            ),
        )

    if (
        not evidence.dois
        and not evidence.title
        and not evidence.pii
        and not evidence.arxiv_doi
    ):
        return replace(
            analysis,
            outcome=replace(
                outcome,
                note=(
                    "No DOI, publisher PII, arXiv identifier, or usable title "
                    "was extracted from the PDF"
                ),
            ),
        )

    rejection_notes: list[str] = []
    provider_conflicts: list[str] = []
    if evidence.dois:
        outcome = replace(outcome, doi=evidence.dois[0].doi)

    for candidate in evidence.dois[:8]:
        decision = await _resolve_doi(
            providers,
            candidate,
            evidence.title,
        )
        if decision.resolved:
            return replace(
                analysis,
                outcome=_resolved_outcome(outcome, decision.resolved),
            )
        if decision.outcome == "conflicting_provider_records":
            conflict_note = (
                f"{candidate.doi} ({candidate.source}): {decision.reason}"
            )
            provider_conflicts.append(conflict_note)
            if candidate.score >= 80:
                return replace(
                    analysis,
                    outcome=replace(
                        outcome,
                        status="conflicting_provider_records",
                        note=conflict_note,
                    ),
                )
        elif decision.reason:
            rejection_notes.append(
                f"{candidate.doi} ({candidate.source}): {decision.reason}"
            )

    pii_decision = await _resolve_pii(
        providers,
        evidence.pii,
        evidence.title,
    )
    if pii_decision.resolved:
        return replace(
            analysis,
            outcome=_resolved_outcome(outcome, pii_decision.resolved),
        )
    if pii_decision.reason:
        rejection_notes.append(pii_decision.reason)

    title_decision = await _resolve_title(
        providers,
        evidence.title,
        pdf_path.name,
    )
    if title_decision.resolved:
        return replace(
            analysis,
            outcome=_resolved_outcome(outcome, title_decision.resolved),
        )
    if title_decision.outcome == "conflicting_provider_records":
        provider_conflicts.append(title_decision.reason)
        return replace(
            analysis,
            outcome=replace(
                outcome,
                status="conflicting_provider_records",
                note=title_decision.reason,
            ),
        )

    if (
        evidence.arxiv_doi
        and not any(
            pdf_evidence.dois_equal(item.doi, evidence.arxiv_doi)
            for item in evidence.dois
        )
    ):
        inferred = pdf_evidence.DoiCandidate(
            doi=evidence.arxiv_doi,
            source="filename-arxiv",
            score=75,
        )
        inferred_decision = await _resolve_doi(
            providers,
            inferred,
            evidence.title,
        )
        if inferred_decision.resolved:
            return replace(
                analysis,
                outcome=_resolved_outcome(outcome, inferred_decision.resolved),
            )
        if inferred_decision.outcome == "conflicting_provider_records":
            provider_conflicts.append(
                f"{evidence.arxiv_doi} (filename-arxiv): "
                f"{inferred_decision.reason}"
            )
        elif inferred_decision.reason:
            rejection_notes.append(
                f"{evidence.arxiv_doi} (filename-arxiv): "
                f"{inferred_decision.reason}"
            )

    if provider_conflicts:
        outcome = replace(
            outcome,
            status="conflicting_provider_records",
            note="; ".join(provider_conflicts[:3]),
        )
    elif evidence.dois or evidence.pii or evidence.arxiv_doi:
        outcome = replace(
            outcome,
            status="no_confident_metadata_found",
            note=(
                "; ".join(rejection_notes[:3])
                or title_decision.reason
            ),
        )
    else:
        outcome = replace(
            outcome,
            status="no_confident_title_match",
            note=title_decision.reason,
        )
    return replace(analysis, outcome=outcome)


def _fatal_analysis(pdf_path: Path, error: Exception) -> _Analysis:
    message = str(error)
    classification = document_classification.classify_failed_pdf(pdf_path)
    return _Analysis(
        OrganizationOutcome(
            source=pdf_path,
            new_name=pdf_path.name,
            status=f"fatal_error:{type(error).__name__}",
            note=message,
            pdf_status="pdf_processing_error",
            pdf_note=message,
            failure_category=classification.category,
            classification_note=classification.reason,
        )
    )


def _source_order(path: Path) -> str:
    return str(path).casefold()


def _is_in_holding_directory(
    root: Path,
    path: Path,
    *,
    include_failed: bool = False,
) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    holding = {"duplicated", "superseded"}
    if not include_failed:
        holding.add("failed")
    return bool(
        {
            part.casefold()
            for part in relative.parts[:-1]
        }
        & holding
    )


def _discover_pdfs(options: RunOptions) -> list[Path]:
    root = options.root
    if options.retry_failed:
        failed = root / "Failed"
        if not failed.exists():
            return []
        return sorted(
            (
                path
                for path in failed.rglob("*.pdf")
                if not pdf_rewrite.is_rewrite_artifact(path)
                and not collection_mutation.is_replacement_artifact(path)
            ),
            key=_source_order,
        )
    return [
        path
        for path in sorted(root.rglob("*.pdf"), key=_source_order)
        if not pdf_rewrite.is_rewrite_artifact(path)
        and not collection_mutation.is_replacement_artifact(path)
        and not _is_in_holding_directory(
            root,
            path,
            include_failed=options.include_failed,
        )
        and (
            options.newer_than is None
            or path.stat().st_mtime > options.newer_than
        )
    ]


async def _analyze_all(
    files: list[Path],
    options: RunOptions,
    providers: ProviderSet,
    pdf_runner: pdf_process.PdfProcessRunner,
    progress: ProgressAdapter | None,
    identity_cache: identity_cache_storage.IdentityCache | None,
) -> list[_Analysis]:
    semaphore = asyncio.Semaphore(PDF_WORKER_CONCURRENCY)

    async def safe_analysis(path: Path) -> _Analysis:
        try:
            async with semaphore:
                return await _analyze_pdf(
                    path,
                    options.pdf_password,
                    providers,
                    pdf_runner,
                    identity_cache,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return _fatal_analysis(path, exc)

    tasks = [asyncio.create_task(safe_analysis(path)) for path in files]
    completed: list[_Analysis] = []
    try:
        for future in asyncio.as_completed(tasks):
            analysis = await future
            completed.append(analysis)
            if progress:
                progress(len(completed), len(files), analysis.outcome)
    finally:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    return sorted(completed, key=lambda item: _source_order(item.outcome.source))


def _apply_publication_audit(
    outcome: OrganizationOutcome,
    audit: publication_audit.PublicationAudit,
) -> OrganizationOutcome:
    values = audit.report_values()
    return replace(
        outcome,
        retraction_status=str(values["retraction_status"]),
        retraction_evidence=str(values["retraction_evidence"]),
        retraction_note=str(values["retraction_note"]),
        commentary_status=str(values["commentary_status"]),
        pubmed_comment_links=str(values["pubmed_comment_links"]),
        scite_total=values["scite_total"],
        scite_supporting=values["scite_supporting"],
        scite_contradicting=values["scite_contradicting"],
        scite_mentioning=values["scite_mentioning"],
        scite_unclassified=values["scite_unclassified"],
        scite_citing_publications=values["scite_citing_publications"],
        commentary_note=str(values["commentary_note"]),
        audit_checked_at=str(values["audit_checked_at"]),
    )


async def _audit_analyses(
    analyses: list[_Analysis],
    auditor: publication_audit.AsyncAuditor | None,
) -> list[_Analysis]:
    if auditor is None:
        return analyses

    grouped: dict[publication_audit.AuditIdentity, list[int]] = {}
    updated = list(analyses)
    for index, analysis in enumerate(analyses):
        outcome = analysis.outcome
        if not _is_resolved_status(outcome.status):
            continue
        key = publication_audit.AuditIdentity.from_values(
            doi=outcome.doi,
            pmid=outcome.pmid,
            arxiv_id=outcome.arxiv_doi,
        )
        grouped.setdefault(key, []).append(index)
    valid = tuple(
        identity
        for identity in grouped
        if any((identity.doi, identity.pmid, identity.arxiv_id))
    )
    try:
        results = await auditor.audit_many(valid)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        results = {
            identity: publication_audit.incomplete_audit(
                doi=identity.doi or None,
                pmid=identity.pmid or None,
                arxiv_id=identity.arxiv_id or None,
                reason=(
                    "Publication audit failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            for identity in valid
        }

    for identity, indexes in grouped.items():
        audit = results.get(identity)
        if audit is None:
            reason = (
                "Accepted record had no DOI, PMID, or arXiv identifier"
                if not any((identity.doi, identity.pmid, identity.arxiv_id))
                else "Publication audit returned no result for the identity"
            )
            audit = publication_audit.incomplete_audit(reason=reason)
        for index in indexes:
            updated[index] = replace(
                updated[index],
                outcome=_apply_publication_audit(
                    updated[index].outcome,
                    audit,
                ),
            )
    return updated


@dataclass(frozen=True, slots=True)
class _ProductionServices:
    providers: metadata_providers.MetadataProviders
    auditor: publication_audit.AsyncPublicationAuditor


@asynccontextmanager
async def _production_services(
    options: RunOptions,
) -> AsyncIterator[_ProductionServices]:
    cache = metadata_providers.MetadataCache(
        options.root / ".pdf_renamer_cache.sqlite"
    )
    cache.open()
    try:
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=50,
        )
        async with httpx.AsyncClient(
            timeout=30,
            limits=limits,
            verify=metadata_providers.build_http_verify_context(),
            follow_redirects=True,
        ) as client:
            ncbi_limiter = metadata_providers.AsyncRateLimiter(
                10 if options.api_key else 3,
                1.0,
            )
            metadata_limiter = metadata_providers.AsyncRateLimiter(50, 1.0)
            metadata_semaphore = asyncio.Semaphore(METADATA_CONCURRENCY)
            run_payloads = metadata_providers.RunProviderPayloads()
            yield _ProductionServices(
                providers=metadata_providers.MetadataProviders.build(
                    client,
                    cache,
                    api_key=options.api_key,
                    email=options.email,
                    ncbi_limiter=ncbi_limiter,
                    metadata_limiter=metadata_limiter,
                    metadata_semaphore=metadata_semaphore,
                    headers_json=metadata_providers.build_headers(
                        options.email,
                        accept_xml=False,
                    ),
                    headers_xml=metadata_providers.build_headers(
                        options.email,
                        accept_xml=True,
                    ),
                    run_payloads=run_payloads,
                ),
                auditor=publication_audit.AsyncPublicationAuditor(
                    client,
                    api_key=options.api_key,
                    email=options.email,
                    ncbi_limiter=ncbi_limiter,
                    metadata_limiter=metadata_limiter,
                    semaphore=metadata_semaphore,
                    run_payloads=run_payloads,
                    scite=publication_audit.AsyncSciteTalliesClient(
                        client,
                        cache=cache,
                        api_key=options.scite_api_key,
                        headers=metadata_providers.build_headers(options.email),
                    ),
                ),
            )
    finally:
        cache.close()


async def retry_incomplete_audits(
    options: RunOptions,
    rows: list[dict[str, object]],
    *,
    auditor: publication_audit.AsyncAuditor | None = None,
    scite: publication_audit.AsyncSciteTallies | None = None,
) -> tuple[list[dict[str, object]], int]:
    """Refresh only incomplete audit fields in an existing organization report."""
    root = options.root.expanduser().resolve()
    if not root.is_dir():
        raise OrganizationRunError(
            f"Collection root is not a directory: {root}"
        )
    full_groups: dict[publication_audit.AuditIdentity, list[int]] = {}
    scite_groups: dict[publication_audit.AuditIdentity, list[int]] = {}
    updated = [dict(row) for row in rows]
    for index, row in enumerate(rows):
        if not publication_audit.compact_audit_needs_retry(row):
            continue
        identity = publication_audit.AuditIdentity.from_values(
            doi=str(row.get("doi") or ""),
            pmid=str(row.get("pmid") or ""),
            arxiv_id=str(row.get("arxiv_doi") or ""),
        )
        if not any((identity.doi, identity.pmid, identity.arxiv_id)):
            continue
        groups = (
            scite_groups
            if publication_audit.compact_audit_needs_only_scite(row)
            else full_groups
        )
        groups.setdefault(identity, []).append(index)
    selected_count = sum(
        len(indexes)
        for groups in (full_groups, scite_groups)
        for indexes in groups.values()
    )
    if not selected_count:
        return updated, 0

    async def refresh(
        selected_auditor: publication_audit.AsyncAuditor,
        selected_scite: publication_audit.AsyncSciteTallies | None,
    ) -> None:
        if selected_scite is None:
            for identity, indexes in scite_groups.items():
                full_groups.setdefault(identity, []).extend(indexes)
            scite_groups.clear()
        else:
            for identity in tuple(scite_groups):
                if identity in full_groups:
                    full_groups[identity].extend(scite_groups.pop(identity))

        results: Mapping[
            publication_audit.AuditIdentity,
            publication_audit.PublicationAudit,
        ] = {}
        if full_groups:
            try:
                results = await selected_auditor.audit_many(full_groups)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                results = {
                    identity: publication_audit.incomplete_audit(
                        doi=identity.doi or None,
                        pmid=identity.pmid or None,
                        arxiv_id=identity.arxiv_id or None,
                        reason=(
                            "Publication audit failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                    for identity in full_groups
                }
        for identity, indexes in full_groups.items():
            audit = results.get(identity)
            if audit is None:
                audit = publication_audit.incomplete_audit(
                    reason="Publication audit returned no result for the identity"
                )
            for index in indexes:
                updated[index] = publication_audit.merge_compact_retry(
                    updated[index],
                    audit,
                )

        if selected_scite is None or not scite_groups:
            return
        try:
            scite_results = await selected_scite.fetch_many(
                identity.doi for identity in scite_groups
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            scite_results = {
                identity.doi: publication_audit.SciteLookup(
                    note=f"Scite batch failed: {type(exc).__name__}: {exc}"
                )
                for identity in scite_groups
            }
        for identity, indexes in scite_groups.items():
            lookup = scite_results.get(
                identity.doi,
                publication_audit.SciteLookup(
                    note="Scite returned no result for the identity"
                ),
            )
            for index in indexes:
                updated[index] = publication_audit.merge_scite_retry(
                    updated[index],
                    lookup,
                )

    if auditor is not None:
        await refresh(auditor, scite)
    else:
        options = replace(options, root=root)
        async with _production_services(options) as production:
            await refresh(production.auditor, production.auditor.scite)
    return updated, selected_count


async def _complete_isolated_transaction(
    operation: Callable[..., T],
    *args: object,
    **kwargs: object,
) -> T:
    task = asyncio.create_task(
        asyncio.to_thread(operation, *args, **kwargs)
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.shield(task)
        raise


async def _complete_isolated_async_transaction(
    operation: Callable[..., Awaitable[T]],
    *args: object,
    **kwargs: object,
) -> T:
    task = asyncio.create_task(operation(*args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.shield(task)
        raise


async def _apply_rewrites(
    analyses: list[_Analysis],
    options: RunOptions,
    pdf_runner: pdf_process.PdfProcessRunner,
) -> list[OrganizationOutcome]:
    outcomes: list[OrganizationOutcome] = []
    for analysis in analyses:
        outcome = analysis.outcome
        if (
            outcome.source_token is not None
            and file_metadata.file_token(outcome.source) != outcome.source_token
        ):
            outcomes.append(replace(
                outcome, status="pdf_source_changed", pdf_status="pdf_source_changed",
                pdf_note="Source changed since analysis",
                note="Source changed since analysis; rerun analysis",
            ))
            continue
        # Analysis failures retain their status for normal failed placement.
        # Only actionable normalization or compression starts a transaction.
        if (
            not analysis.needs_rewrite
            or outcome.pdf_status in BLOCKING_PDF_STATUSES
        ):
            outcomes.append(outcome)
            continue
        try:
            rewrite = await _complete_isolated_async_transaction(
                pdf_runner.rewrite,
                outcome.source,
                pdf_password=options.pdf_password,
                remove_page_indices=analysis.remove_page_indices,
                compress_if_large=analysis.compress_if_large,
                expected_source_token=outcome.source_token,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            rewrite = pdf_rewrite.PdfRewriteOutcome(
                (
                    "pdf_processing_error"
                    if isinstance(exc, BrokenProcessPool)
                    else "pdf_repair_failed"
                    if analysis.requires_normalization
                    else "pdf_compression_failed"
                ),
                f"PDF rewrite transaction failed: {type(exc).__name__}: {exc}",
            )
        outcome = replace(
            outcome,
            pdf_status=rewrite.status,
            pdf_note=rewrite.note,
            source_token=(
                rewrite.source_token
                if rewrite.source_token is not None
                else outcome.source_token
            ),
            placement_blocked=rewrite.status in BLOCKING_PDF_STATUSES,
        )
        if rewrite.status in BLOCKING_PDF_STATUSES:
            outcome = replace(
                outcome,
                status=rewrite.status,
                note=rewrite.note,
            )
        outcomes.append(outcome)
    return outcomes


def _commit_priority(
    root: Path,
    outcome: OrganizationOutcome,
) -> tuple[int, str, int]:
    source = outcome.source
    base_name = outcome.new_name or source.name
    if not _is_resolved_status(outcome.status):
        return 2, _source_order(source), 0
    root_target = root / base_name
    try:
        modification_time = source.stat().st_mtime_ns
    except OSError:
        modification_time = -1
    return (
        0 if source == root_target else 1,
        _source_order(root_target),
        -modification_time,
    )


def _mutate_analyzed_source(
    operation: Callable[..., T], outcome: OrganizationOutcome, *args: object,
) -> T:
    """Check the analyzed state again inside the placement transaction task."""
    if (
        outcome.source_token is not None
        and file_metadata.file_token(outcome.source) != outcome.source_token
    ):
        raise collection_mutation.CollectionMutationError(
            "Source changed since analysis; rerun analysis"
        )
    return operation(outcome.source, *args)


async def _existing_target_matches(
    path: Path, outcome: OrganizationOutcome,
    pdf_runner: pdf_process.PdfProcessRunner | None,
) -> bool:
    """Require fresh DOI and title corroboration before displacing a target."""
    if pdf_runner is None or not outcome.doi:
        return False
    before = file_metadata.file_token(path)
    try:
        evidence = await pdf_runner.inspect(path)
    except Exception:
        return False
    if (
        before is None or file_metadata.file_token(path) != before
        or evidence.pdf_status in BLOCKING_PDF_STATUSES or evidence.needs_rewrite
    ):
        return False
    identifiers = {candidate.doi.casefold() for candidate in evidence.dois if candidate.score >= 80}
    return (
        identifiers == {outcome.doi.casefold()}
        and bool(evidence.title and outcome.title)
        and record_matching.title_similarity(evidence.title, outcome.title)
        >= record_matching.LOW_SOURCE_TITLE_CONFIRM_SCORE
    )


def _place_verified_source(
    source: Path, target: Path, duplicate_directory: Path,
    expected_target_token: file_metadata.FileToken | None,
) -> collection_mutation.PlacementOutcome:
    if file_metadata.file_token(target) != expected_target_token:
        raise collection_mutation.CollectionMutationError("Canonical target changed before placement")
    return collection_mutation.place_canonical(source, target, duplicate_directory)


async def _apply_collection_mutations(
    root: Path,
    outcomes: list[OrganizationOutcome],
    pdf_runner: pdf_process.PdfProcessRunner | None = None,
) -> list[OrganizationOutcome]:
    duplicate_directory = root / "Duplicated"
    failed_directory = root / "Failed"
    updated = list(outcomes)
    canonical_owners: dict[Path, int] = {}
    indexes = sorted(
        range(len(updated)),
        key=lambda index: _commit_priority(root, updated[index]),
    )

    for index in indexes:
        outcome = updated[index]
        source = outcome.source
        base_name = outcome.new_name or source.name
        if outcome.placement_blocked:
            continue
        if outcome.pdf_status in {
            "pdf_source_changed", "pdf_processing_error", "pdf_processing_timeout",
        } or outcome.status.startswith("fatal_error:"):
            continue
        if (
            outcome.source_token is not None
            and file_metadata.file_token(source) != outcome.source_token
        ):
            updated[index] = replace(
                outcome, status="pdf_source_changed", pdf_status="pdf_source_changed",
                note="Source changed before placement; rerun analysis",
            )
            continue
        if not source.exists():
            updated[index] = replace(
                outcome,
                status="error_rename:source file no longer exists",
            )
            continue

        if not _is_resolved_status(outcome.status):
            category = (
                outcome.failure_category
                if outcome.failure_category
                in document_classification.FAILED_CATEGORIES
                else "Others"
            )
            category_directory = failed_directory / category
            final_target = category_directory / base_name
            if source == final_target or (
                final_target.exists() and source.samefile(final_target)
            ):
                updated[index] = replace(outcome, target_path=source)
                continue
            try:
                final_target = await _complete_isolated_transaction(
                    _mutate_analyzed_source,
                    collection_mutation.move_to_unique,
                    outcome,
                    category_directory,
                    base_name,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                updated[index] = replace(
                    outcome,
                    status=f"error_rename:{exc}",
                )
            else:
                updated[index] = replace(
                    outcome,
                    target_path=final_target,
                )
            continue

        root_target = root / base_name
        owner_index = canonical_owners.get(root_target)
        unique_collision = False
        target_token = file_metadata.file_token(root_target)
        if root_target.exists() and not source.samefile(root_target):
            owner = updated[owner_index] if owner_index is not None else None
            same_work = bool(
                owner is not None and outcome.doi and owner.doi
                and pdf_evidence.dois_equal(outcome.doi, owner.doi)
                and owner.source_token == target_token
            )
            if not same_work:
                same_work = await _existing_target_matches(root_target, outcome, pdf_runner)
            if not same_work:
                base_name = filename_policy.disambiguate_pdf_name(
                    base_name,
                    outcome.doi or str(source),
                )
                root_target = root / base_name
                unique_collision = root_target.exists() or root_target.is_symlink()
                target_token = file_metadata.file_token(root_target)
                outcome = _append_note(
                    replace(outcome, new_name=base_name),
                    "Generated-name collision had a different or unverified identity; added a "
                    "stable DOI-derived suffix",
                )
        try:
            if unique_collision:
                placed = await _complete_isolated_transaction(
                    _mutate_analyzed_source, collection_mutation.move_to_unique,
                    outcome, root, base_name,
                )
                placement = collection_mutation.PlacementOutcome("canonical", placed)
            else:
                placement = await _complete_isolated_transaction(
                    _mutate_analyzed_source, _place_verified_source,
                    outcome, root_target, duplicate_directory, target_token,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            updated[index] = replace(
                outcome,
                status=f"error_rename:{exc}",
            )
            continue

        outcome = replace(outcome, target_path=placement.target)
        if placement.action == "duplicate":
            updated[index] = _mark_duplicate(
                outcome,
                "Existing canonical copy is newer or has the same "
                "modification time",
            )
            continue

        if placement.action == "canonical_replaced":
            displaced_path = placement.displaced
            if displaced_path is None:
                updated[index] = replace(
                    outcome,
                    status="error_rename:missing displaced canonical path",
                )
                continue
            displaced_index = canonical_owners.get(root_target)
            if displaced_index is not None:
                displaced = replace(
                    updated[displaced_index],
                    target_path=displaced_path,
                )
                updated[displaced_index] = _mark_duplicate(
                    displaced,
                    "Newer copy retained as canonical; this older copy "
                    "was preserved",
                )
            outcome = _append_note(
                outcome,
                "Newer copy retained as canonical; prior copy preserved "
                "in Duplicated",
            )

        updated[index] = outcome
        canonical_owners[root_target] = index

    return sorted(updated, key=lambda item: _source_order(item.source))


async def _update_identity_cache_after_commit(
    analyses: list[_Analysis],
    outcomes: list[OrganizationOutcome],
    options: RunOptions,
    pdf_runner: pdf_process.PdfProcessRunner,
    identity_cache: identity_cache_storage.IdentityCache | None,
) -> None:
    """Move reusable evidence to committed paths and refresh rewritten PDFs."""
    cache = (
        identity_cache
        if identity_cache is not None
        and identity_cache.usable
        and not options.pdf_password
        else None
    )
    if cache is None:
        return
    analyses_by_source = {
        analysis.outcome.source: analysis for analysis in analyses
    }

    for outcome in outcomes:
        if not cache.usable:
            return
        analysis = analyses_by_source.get(outcome.source)
        evidence = analysis.evidence if analysis is not None else None
        target = outcome.target_path or outcome.source
        try:
            if not target.is_file() or evidence is None:
                cache.discard(outcome.source)
                continue
            if evidence.pdf_status in BLOCKING_PDF_STATUSES:
                cache.discard(outcome.source)
                cache.discard(target)
                continue
            if analysis is not None and (
                analysis.requires_normalization
                or (analysis.compress_if_large and outcome.pdf_status == "pdf_compressed")
                or (evidence.source_token is not None and file_metadata.file_token(target) != evidence.source_token)
            ):
                try:
                    before = file_metadata.file_token(target)
                    evidence = await pdf_runner.inspect(
                        target,
                        pdf_password=options.pdf_password,
                    )
                    if before is None or file_metadata.file_token(target) != before:
                        cache.discard(outcome.source)
                        cache.discard(target)
                        continue
                    evidence = replace(evidence, source_token=before)
                except Exception:
                    cache.discard(outcome.source)
                    cache.discard(target)
                    continue
                if evidence.pdf_status in BLOCKING_PDF_STATUSES:
                    cache.discard(outcome.source)
                    cache.discard(target)
                    continue
            cache.put(
                target,
                pdf_evidence.evidence_to_cache_payload(evidence),
            )
            if target != outcome.source:
                cache.discard(outcome.source)
        except Exception as exc:
            _disable_identity_cache(cache, "post-commit cache update", exc)
            return


async def run(
    options: RunOptions,
    *,
    providers: ProviderSet | None = None,
    auditor: publication_audit.AsyncAuditor | None = None,
    pdf_runner: pdf_process.PdfProcessRunner | None = None,
    identity_cache: identity_cache_storage.IdentityCache | None = None,
    progress: ProgressAdapter | None = None,
) -> list[OrganizationOutcome]:
    root = options.root.expanduser().resolve()
    options = replace(options, root=root)
    if not root.is_dir():
        raise OrganizationRunError(
            f"Collection root is not a directory: {root}"
        )
    if options.retry_failed and options.newer_than is not None:
        raise OrganizationRunError(
            "retry_failed and newer_than cannot be combined"
        )
    if options.retry_failed and options.include_failed:
        raise OrganizationRunError(
            "retry_failed and include_failed cannot be combined"
        )
    if options.include_failed and options.commit:
        raise OrganizationRunError(
            "include_failed is read-only and cannot be combined with commit"
        )

    owned_pdf_runner: pdf_process.SpawnPdfProcessRunner | None = None
    if pdf_runner is None:
        owned_pdf_runner = pdf_process.SpawnPdfProcessRunner(
            inspection_timeout=options.inspection_timeout,
        )
        pdf_runner = owned_pdf_runner

    try:
        files = _discover_pdfs(options)
        if providers is None:
            async with _production_services(options) as production:
                analyses = await _analyze_all(
                    files,
                    options,
                    production.providers,
                    pdf_runner,
                    progress,
                    identity_cache,
                )
                analyses = await _audit_analyses(
                    analyses,
                    auditor or production.auditor,
                )
        else:
            analyses = await _analyze_all(
                files,
                options,
                providers,
                pdf_runner,
                progress,
                identity_cache,
            )
            analyses = await _audit_analyses(analyses, auditor)

        outcomes = [analysis.outcome for analysis in analyses]
        if not options.commit:
            return outcomes

        outcomes = await _apply_rewrites(analyses, options, pdf_runner)
        outcomes = await _apply_collection_mutations(root, outcomes, pdf_runner)
        await _update_identity_cache_after_commit(
            analyses,
            outcomes,
            options,
            pdf_runner,
            identity_cache,
        )
        return outcomes
    finally:
        if owned_pdf_runner is not None:
            await owned_pdf_runner.aclose()


@dataclass(frozen=True, slots=True)
class AuditOptions:
    root: Path
    api_key: str = ""
    email: str = ""
    pdf_password: str = ""
    scite_api_key: str = ""
    inspection_timeout: float = pdf_process.DEFAULT_INSPECTION_TIMEOUT

    def __post_init__(self) -> None:
        object.__setattr__(self, "inspection_timeout", pdf_process.inspection_timeout_seconds(self.inspection_timeout))


async def run_audit(
    options: AuditOptions,
    *,
    providers: ProviderSet | None = None,
    auditor: publication_audit.AsyncAuditor | None = None,
    pdf_runner: pdf_process.PdfProcessRunner | None = None,
    identity_cache: identity_cache_storage.IdentityCache | None = None,
    progress: ProgressAdapter | None = None,
) -> list[dict[str, object]]:
    """Audit active and Failed PDFs without authorizing mutation."""
    root = options.root.expanduser().resolve()
    outcomes = await run(
        RunOptions(
            root=root,
            commit=False,
            api_key=options.api_key,
            email=options.email,
            include_failed=True,
            pdf_password=options.pdf_password,
            scite_api_key=options.scite_api_key,
            inspection_timeout=options.inspection_timeout,
        ),
        providers=providers,
        auditor=auditor,
        pdf_runner=pdf_runner,
        identity_cache=identity_cache,
        progress=progress,
    )
    return [audit_report.report_row(root, outcome) for outcome in outcomes]
