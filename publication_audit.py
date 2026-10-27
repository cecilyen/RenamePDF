"""Fetch publication-status evidence and format audit results."""

from __future__ import annotations

import asyncio
import json
import random
import time
import urllib.parse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol, TypeVar

import httpx

import metadata_providers
import pdf_evidence
import publication_status


SCITE_TALLIES = "https://api.scite.ai/tallies"
SCITE_BATCH_SIZE = 500
SCITE_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
SCITE_ATTEMPTS = 3
SCITE_REQUEST_BUDGET_SECONDS = 180.0
T = TypeVar("T")

AUDIT_FIELD_SUFFIXES = (
    "retraction_status",
    "retraction_evidence",
    "retraction_note",
    "commentary_status",
    "pubmed_comment_links",
    "scite_total",
    "scite_supporting",
    "scite_contradicting",
    "scite_mentioning",
    "scite_unclassified",
    "scite_citing_publications",
    "commentary_note",
    "audit_checked_at",
)
COMPACT_AUDIT_FIELD_SUFFIXES = (
    "retraction_status",
    "commentary_status",
    "audit_evidence",
    "audit_note",
    "audit_checked_at",
)
SCITE_REPORT_FIELDS = (
    "scite_total",
    "scite_supporting",
    "scite_contradicting",
    "scite_mentioning",
    "scite_unclassified",
    "scite_citing_publications",
)


def report_field_names(prefix: str = "") -> tuple[str, ...]:
    return tuple(f"{prefix}{name}" for name in AUDIT_FIELD_SUFFIXES)


def compact_report_field_names(prefix: str = "") -> tuple[str, ...]:
    """Return the compact audit columns used by the organization report."""
    return tuple(f"{prefix}{name}" for name in COMPACT_AUDIT_FIELD_SUFFIXES)


def _compact_json(values: dict[str, object]) -> str:
    if not values:
        return ""
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def compact_report_values(
    detailed: Mapping[str, object],
    prefix: str = "",
) -> dict[str, object]:
    """Consolidate details without merging formal and commentary status."""
    evidence: dict[str, object] = {}
    retraction_evidence = str(detailed.get("retraction_evidence") or "").strip()
    if retraction_evidence:
        evidence["retraction"] = retraction_evidence
    pubmed_comments = str(detailed.get("pubmed_comment_links") or "").strip()
    if pubmed_comments:
        evidence["pubmed_comments"] = pubmed_comments

    scite = {
        name.removeprefix("scite_"): detailed[name]
        for name in SCITE_REPORT_FIELDS
        if detailed.get(name) not in (None, "")
    }
    if scite:
        evidence["scite"] = scite

    retraction_note = str(detailed.get("retraction_note") or "").strip()
    commentary_note = str(detailed.get("commentary_note") or "").strip()
    notes: dict[str, object] = {}
    if retraction_note and retraction_note == commentary_note:
        notes["all"] = retraction_note
    else:
        if retraction_note:
            notes["retraction"] = retraction_note
        if commentary_note:
            notes["commentary"] = commentary_note

    values: dict[str, object] = {
        "retraction_status": str(
            detailed.get("retraction_status") or "identity_unresolved"
        ),
        "commentary_status": str(
            detailed.get("commentary_status") or "identity_unresolved"
        ),
        "audit_evidence": _compact_json(evidence),
        "audit_note": _compact_json(notes),
        "audit_checked_at": str(detailed.get("audit_checked_at") or ""),
    }
    return {f"{prefix}{key}": value for key, value in values.items()}


def _json_object(value: object) -> dict[str, object]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def compact_audit_needs_retry(row: Mapping[str, object]) -> bool:
    """Return whether a compact organization audit has an unfinished side."""
    return (
        str(row.get("retraction_status") or "") == "check_incomplete"
        or str(row.get("commentary_status") or "") == "check_incomplete"
        or "scite tally was unavailable"
        in str(row.get("audit_note") or "").casefold()
    )


def compact_audit_needs_only_scite(row: Mapping[str, object]) -> bool:
    """Return whether Scite is the only unfinished audit source."""
    note = str(row.get("audit_note") or "").casefold()
    return (
        "scite tally was unavailable" in note
        and "pubmed record was unavailable" not in note
        and str(row.get("retraction_status") or "") != "check_incomplete"
    )


def merge_compact_retry(
    existing: Mapping[str, object],
    audit: PublicationAudit,
) -> dict[str, object]:
    """Replace only the incomplete sides of a compact audit result."""
    retry_retraction = (
        str(existing.get("retraction_status") or "") == "check_incomplete"
    )
    retry_commentary = (
        str(existing.get("commentary_status") or "") == "check_incomplete"
        or "scite tally was unavailable"
        in str(existing.get("audit_note") or "").casefold()
    )
    if not (retry_retraction or retry_commentary):
        return dict(existing)

    replacement = compact_report_values(audit.report_values())
    old_evidence = _json_object(existing.get("audit_evidence"))
    new_evidence = _json_object(replacement.get("audit_evidence"))
    evidence: dict[str, object] = {}
    retraction_evidence = (
        new_evidence.get("retraction")
        if retry_retraction
        else old_evidence.get("retraction")
    )
    if retraction_evidence:
        evidence["retraction"] = retraction_evidence
    for key in ("pubmed_comments", "scite"):
        value = (
            new_evidence.get(key)
            if retry_commentary
            else old_evidence.get(key)
        )
        if value:
            evidence[key] = value

    old_notes = _json_object(existing.get("audit_note"))
    new_notes = _json_object(replacement.get("audit_note"))

    def side_note(notes: Mapping[str, object], side: str) -> str:
        return str(notes.get(side) or notes.get("all") or "").strip()

    retraction_note = side_note(
        new_notes if retry_retraction else old_notes,
        "retraction",
    )
    commentary_note = side_note(
        new_notes if retry_commentary else old_notes,
        "commentary",
    )
    notes: dict[str, object] = {}
    if retraction_note and retraction_note == commentary_note:
        notes["all"] = retraction_note
    else:
        if retraction_note:
            notes["retraction"] = retraction_note
        if commentary_note:
            notes["commentary"] = commentary_note

    merged = dict(existing)
    if retry_retraction:
        merged["retraction_status"] = replacement["retraction_status"]
    if retry_commentary:
        merged["commentary_status"] = replacement["commentary_status"]
    merged["audit_evidence"] = _compact_json(evidence)
    merged["audit_note"] = _compact_json(notes)
    merged["audit_checked_at"] = replacement["audit_checked_at"]
    return merged


def merge_scite_retry(
    existing: Mapping[str, object],
    lookup: SciteLookup,
    *,
    checked_at: str | None = None,
) -> dict[str, object]:
    """Merge one Scite retry without re-querying completed formal sources."""
    merged = dict(existing)
    evidence = _json_object(existing.get("audit_evidence"))
    notes = _json_object(existing.get("audit_note"))
    pubmed_comments = str(evidence.get("pubmed_comments") or "").strip()
    assessment = publication_status.assess_commentary(
        scite_payload=lookup.payload,
        scite_checked=lookup.checked,
        scite_expected=True,
        known_comment_links=publication_status.comment_links_from_report(
            pubmed_comments
        ),
    )
    tallies = assessment.scite

    if lookup.checked:
        scite = {
            key: value
            for key, value in (
                ("total", tallies.total),
                ("supporting", tallies.supporting),
                ("contradicting", tallies.contradicting),
                ("mentioning", tallies.mentioning),
                ("unclassified", tallies.unclassified),
                ("citing_publications", tallies.citing_publications),
            )
            if value is not None
        }
        if scite:
            evidence["scite"] = scite
        else:
            evidence.pop("scite", None)

    commentary_note = "; ".join(
        value for value in (assessment.note, lookup.note) if value
    )
    retraction_note = str(
        notes.get("retraction") or notes.get("all") or ""
    ).strip()
    updated_notes: dict[str, object] = {}
    if retraction_note and retraction_note == commentary_note:
        updated_notes["all"] = retraction_note
    else:
        if retraction_note:
            updated_notes["retraction"] = retraction_note
        if commentary_note:
            updated_notes["commentary"] = commentary_note

    merged["commentary_status"] = assessment.status
    merged["audit_evidence"] = _compact_json(evidence)
    merged["audit_note"] = _compact_json(updated_notes)
    merged["audit_checked_at"] = checked_at or _checked_at(
        lambda: datetime.now(timezone.utc)
    )
    return merged


def empty_report_values(
    prefix: str = "",
    *,
    status: str = "identity_unresolved",
) -> dict[str, object]:
    values: dict[str, object] = {
        "retraction_status": status,
        "retraction_evidence": "",
        "retraction_note": "",
        "commentary_status": status,
        "pubmed_comment_links": "",
        "scite_total": "",
        "scite_supporting": "",
        "scite_contradicting": "",
        "scite_mentioning": "",
        "scite_unclassified": "",
        "scite_citing_publications": "",
        "commentary_note": "",
        "audit_checked_at": "",
    }
    return {f"{prefix}{key}": value for key, value in values.items()}


@dataclass(frozen=True, slots=True)
class PublicationAudit:
    retraction: publication_status.RetractionAssessment
    commentary: publication_status.CommentaryAssessment
    checked_at: str

    def report_values(self, prefix: str = "") -> dict[str, object]:
        scite = self.commentary.scite
        values: dict[str, object] = {
            "retraction_status": self.retraction.status,
            "retraction_evidence": self.retraction.evidence_text,
            "retraction_note": self.retraction.note,
            "commentary_status": self.commentary.status,
            "pubmed_comment_links": ",".join(
                link.report_text()
                for link in self.commentary.pubmed_comment_links
            ),
            "scite_total": _report_number(scite.total),
            "scite_supporting": _report_number(scite.supporting),
            "scite_contradicting": _report_number(scite.contradicting),
            "scite_mentioning": _report_number(scite.mentioning),
            "scite_unclassified": _report_number(scite.unclassified),
            "scite_citing_publications": _report_number(
                scite.citing_publications
            ),
            "commentary_note": self.commentary.note,
            "audit_checked_at": self.checked_at,
        }
        return {f"{prefix}{key}": value for key, value in values.items()}


def _report_number(value: int | None) -> int | str:
    return value if value is not None else ""


def _checked_at(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00",
        "Z",
    )


def incomplete_audit(
    *,
    doi: str | None = None,
    pmid: str | None = None,
    arxiv_id: str | None = None,
    reason: str = "Publication audit failed before all sources were checked",
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> PublicationAudit:
    return PublicationAudit(
        retraction=publication_status.RetractionAssessment(
            status="check_incomplete",
            note=reason,
        ),
        commentary=publication_status.CommentaryAssessment(
            status="check_incomplete",
            note=reason,
        ),
        checked_at=_checked_at(clock),
    )


@dataclass(frozen=True, slots=True)
class AuditIdentity:
    doi: str = ""
    pmid: str = ""
    arxiv_id: str = ""

    @classmethod
    def from_values(
        cls,
        *,
        doi: str | None = None,
        pmid: str | None = None,
        arxiv_id: str | None = None,
    ) -> AuditIdentity:
        return cls(
            pdf_evidence.clean_doi(doi or "").casefold(),
            (pmid or "").strip(),
            (arxiv_id or "").strip().casefold(),
        )


@dataclass(frozen=True, slots=True)
class SciteLookup:
    payload: dict[str, Any] | None = None
    checked: bool = False
    note: str = ""


class AsyncSciteTallies(Protocol):
    async def fetch_many(
        self,
        dois: Iterable[str],
    ) -> Mapping[str, SciteLookup]: ...


class AsyncSciteTalliesClient:
    """Fetch public Scite tallies in cached, rate-limit-aware batches."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        cache: metadata_providers.MetadataCache | None = None,
        api_key: str = "",
        headers: Mapping[str, str] | None = None,
        cache_ttl_seconds: float = SCITE_CACHE_TTL_SECONDS,
        attempts: int = SCITE_ATTEMPTS,
        request_budget_seconds: float = SCITE_REQUEST_BUDGET_SECONDS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.client = client
        self.cache = cache
        self.cache_ttl_seconds = cache_ttl_seconds
        self.attempts = max(1, attempts)
        self.request_budget_seconds = max(0.0, request_budget_seconds)
        self.sleep = sleep
        self.monotonic = monotonic
        self.wall_clock = wall_clock
        self.jitter = jitter
        self.headers = dict(headers or {})
        self.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        if api_key.strip():
            self.headers["Authorization"] = f"Bearer {api_key.strip()}"
        self._request_lock = asyncio.Lock()
        self._cooldown_until = 0.0

    def _now(self) -> datetime:
        value = self.wall_clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _header_delay(
        self,
        headers: Mapping[str, str],
        fallback: float,
    ) -> float:
        retry_after = headers.get("Retry-After")
        if retry_after:
            return metadata_providers.retry_after_seconds(
                retry_after,
                fallback,
                now=self._now(),
                max_delay=None,
            )
        for name in (
            "RateLimit-Reset",
            "X-RateLimit-Reset-Minute",
            "X-RateLimit-Reset",
        ):
            raw = headers.get(name)
            if not raw:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            now_seconds = self._now().timestamp()
            if value > 10_000_000_000:
                value /= 1000.0
            if value > now_seconds:
                value -= now_seconds
            return max(0.0, value)
        return max(0.0, fallback)

    async def _wait_for_cooldown(self, deadline: float) -> bool:
        delay = max(0.0, self._cooldown_until - self.monotonic())
        if delay <= 0:
            return True
        if self.monotonic() + delay > deadline:
            return False
        await self.sleep(delay)
        return True

    def _set_header_cooldown(self, response: httpx.Response) -> None:
        remaining = (
            response.headers.get("RateLimit-Remaining")
            or response.headers.get("X-RateLimit-Remaining-Minute")
            or response.headers.get("X-RateLimit-Remaining")
        )
        if str(remaining or "").strip() != "0":
            return
        delay = self._header_delay(response.headers, 0.0)
        self._cooldown_until = max(
            self._cooldown_until,
            self.monotonic() + delay,
        )

    async def _request_chunk(
        self,
        dois: list[str],
    ) -> tuple[dict[str, dict[str, Any]], str]:
        deadline = self.monotonic() + self.request_budget_seconds
        backoff = 1.0
        async with self._request_lock:
            for attempt in range(self.attempts):
                if not await self._wait_for_cooldown(deadline):
                    return {}, "Scite cooldown exceeded the request time budget"
                remaining = deadline - self.monotonic()
                if remaining <= 0:
                    return {}, "Scite request time budget was exhausted"
                try:
                    response = await self.client.post(
                        SCITE_TALLIES,
                        json=dois,
                        headers=self.headers,
                        timeout=min(30.0, remaining),
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt == self.attempts - 1:
                        return {}, f"Scite transport failed: {type(exc).__name__}"
                    delay = backoff
                else:
                    if response.status_code == 200:
                        self._set_header_cooldown(response)
                        try:
                            body = response.json()
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            return {}, "Scite returned invalid JSON"
                        tallies = (
                            body.get("tallies")
                            if isinstance(body, dict)
                            else None
                        )
                        if not isinstance(tallies, dict):
                            return {}, "Scite returned an unexpected tally response"
                        parsed: dict[str, dict[str, Any]] = {}
                        for raw_doi, payload in tallies.items():
                            key = pdf_evidence.clean_doi(str(raw_doi)).casefold()
                            if key and isinstance(payload, dict):
                                parsed[key] = payload
                        return parsed, ""
                    if response.status_code not in {429, 500, 502, 503, 504}:
                        return {}, (
                            "Scite request failed with HTTP "
                            f"{response.status_code}"
                        )
                    if attempt == self.attempts - 1:
                        return {}, (
                            "Scite remained rate-limited"
                            if response.status_code == 429
                            else (
                                "Scite request failed with HTTP "
                                f"{response.status_code}"
                            )
                        )
                    delay = self._header_delay(response.headers, backoff)

                delay += self.jitter() * 0.25
                if self.monotonic() + delay > deadline:
                    return {}, "Scite retry delay exceeded the request time budget"
                self._cooldown_until = max(
                    self._cooldown_until,
                    self.monotonic() + delay,
                )
                await self.sleep(delay)
                backoff *= 2
        return {}, "Scite request did not complete"

    async def fetch_many(
        self,
        dois: Iterable[str],
    ) -> Mapping[str, SciteLookup]:
        normalized = tuple(
            dict.fromkeys(
                cleaned
                for doi in dois
                if (cleaned := pdf_evidence.clean_doi(doi).casefold())
            )
        )
        results: dict[str, SciteLookup] = {}
        now = self._now().timestamp()
        fresh_after = now - self.cache_ttl_seconds
        missing: list[str] = []
        for doi in normalized:
            payload = (
                self.cache.get_scite_tally(doi, fresh_after=fresh_after)
                if self.cache is not None
                else None
            )
            if payload is None:
                missing.append(doi)
            else:
                results[doi] = SciteLookup(payload=payload, checked=True)

        for start in range(0, len(missing), SCITE_BATCH_SIZE):
            chunk = missing[start : start + SCITE_BATCH_SIZE]
            payloads, note = await self._request_chunk(chunk)
            checked_at = self._now().timestamp()
            for doi in chunk:
                payload = payloads.get(doi)
                if payload is None:
                    results[doi] = SciteLookup(
                        note=note or "Scite returned no tally for the DOI",
                    )
                    continue
                if self.cache is not None:
                    self.cache.put_scite_tally(
                        doi,
                        payload,
                        checked_at=checked_at,
                    )
                results[doi] = SciteLookup(payload=payload, checked=True)
        return results


class AsyncAuditor(Protocol):
    async def audit_many(
        self,
        identities: Iterable[AuditIdentity],
    ) -> Mapping[AuditIdentity, PublicationAudit]: ...

class JsonClient(Protocol):
    email: str

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...


class AsyncPublicationAuditor:
    """Reuse current-run provider payloads and fetch missing audit records."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str = "",
        email: str = "",
        ncbi_limiter: metadata_providers.AsyncRateLimiter | None = None,
        metadata_limiter: metadata_providers.AsyncRateLimiter | None = None,
        semaphore: asyncio.Semaphore | None = None,
        run_payloads: metadata_providers.RunProviderPayloads | None = None,
        scite: AsyncSciteTallies | None = None,
        scite_api_key: str = "",
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.client = client
        self.api_key = api_key
        self.email = email
        self.ncbi_limiter = ncbi_limiter
        self.metadata_limiter = metadata_limiter
        self.semaphore = semaphore
        self.run_payloads = run_payloads
        self.clock = clock
        self.headers_json = metadata_providers.build_headers(email)
        self.headers_xml = metadata_providers.build_headers(email, accept_xml=True)
        self.scite = scite or AsyncSciteTalliesClient(
            client,
            api_key=scite_api_key,
            headers=self.headers_json,
        )
        self._cache: dict[AuditIdentity, PublicationAudit] = {}

    async def _bounded(self, operation: Callable[[], Awaitable[T]]) -> T:
        if self.semaphore is None:
            return await operation()
        async with self.semaphore:
            return await operation()

    async def _crossref_message(self, doi: str) -> dict[str, Any] | None:
        if self.run_payloads is not None:
            message = self.run_payloads.crossref_message(doi)
            if message is not None:
                return message
        payload = await self._bounded(
            lambda: metadata_providers.fetch_http(
                self.client,
                metadata_providers.CROSSREF_WORKS
                + urllib.parse.quote(doi, safe=""),
                {"mailto": self.email} if self.email else None,
                False,
                3,
                self.metadata_limiter,
                self.headers_json,
            )
        )
        message = payload.get("message") if isinstance(payload, dict) else None
        if isinstance(message, dict):
            if self.run_payloads is not None:
                self.run_payloads.remember_crossref_message(message)
            return message
        return None

    async def _pubmed_xml(self, pmid: str, doi: str) -> str | None:
        if self.run_payloads is not None:
            xml_text = self.run_payloads.pubmed_xml(pmid=pmid, doi=doi)
            if xml_text is not None:
                return xml_text
        if not pmid:
            return None
        params: dict[str, Any] = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
        }
        if self.api_key.strip():
            params["api_key"] = self.api_key.strip()
        if self.email.strip():
            params["email"] = self.email.strip()
        payload = await self._bounded(
            lambda: metadata_providers.fetch_http(
                self.client,
                metadata_providers.PUBMED_EFETCH,
                params,
                True,
                3,
                self.ncbi_limiter,
                self.headers_xml,
            )
        )
        if not isinstance(payload, str):
            return None
        if self.run_payloads is not None:
            meta = metadata_providers.parse_pubmed_full(payload)
            self.run_payloads.remember_pubmed_xml(
                pmid,
                payload,
                doi=meta.doi if meta is not None else None,
            )
        return payload

    async def _audit_identity(
        self,
        identity: AuditIdentity,
        scite_lookup: SciteLookup,
    ) -> PublicationAudit:
        crossref_message, pubmed_xml = await asyncio.gather(
            self._crossref_message(identity.doi)
            if identity.doi
            else _async_none(),
            self._pubmed_xml(identity.pmid, identity.doi)
            if identity.pmid or (self.run_payloads is not None and identity.doi)
            else _async_none(),
        )
        pubmed_checked = publication_status.pubmed_record_available(pubmed_xml)
        pubmed_expected = bool(identity.pmid or pubmed_xml is not None)
        commentary = publication_status.assess_commentary(
            pubmed_xml=pubmed_xml,
            scite_payload=scite_lookup.payload,
            pubmed_checked=pubmed_checked,
            scite_checked=scite_lookup.checked,
            pubmed_expected=pubmed_expected,
            scite_expected=bool(identity.doi),
        )
        if scite_lookup.note:
            commentary = replace(
                commentary,
                note="; ".join(
                    value
                    for value in (commentary.note, scite_lookup.note)
                    if value
                ),
            )
        return PublicationAudit(
            retraction=publication_status.assess_retractions(
                crossref_message=crossref_message,
                pubmed_xml=pubmed_xml,
                crossref_checked=isinstance(crossref_message, dict),
                pubmed_checked=pubmed_checked,
                crossref_expected=bool(identity.doi),
                pubmed_expected=pubmed_expected,
            ),
            commentary=commentary,
            checked_at=_checked_at(self.clock),
        )

    async def audit_many(
        self,
        identities: Iterable[AuditIdentity],
    ) -> Mapping[AuditIdentity, PublicationAudit]:
        normalized = tuple(
            dict.fromkeys(
                AuditIdentity.from_values(
                    doi=identity.doi,
                    pmid=identity.pmid,
                    arxiv_id=identity.arxiv_id,
                )
                for identity in identities
            )
        )
        pending = tuple(
            identity for identity in normalized if identity not in self._cache
        )
        if not pending:
            return {identity: self._cache[identity] for identity in normalized}
        try:
            scite_results = await self.scite.fetch_many(
                identity.doi for identity in pending if identity.doi
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            scite_results = {
                identity.doi: SciteLookup(
                    note=f"Scite batch failed: {type(exc).__name__}: {exc}",
                )
                for identity in pending
                if identity.doi
            }

        async def check(
            identity: AuditIdentity,
        ) -> tuple[AuditIdentity, PublicationAudit]:
            lookup = scite_results.get(identity.doi, SciteLookup())
            try:
                result = await self._audit_identity(identity, lookup)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result = incomplete_audit(
                    doi=identity.doi or None,
                    pmid=identity.pmid or None,
                    arxiv_id=identity.arxiv_id or None,
                    reason=(
                        "Publication audit failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    clock=self.clock,
                )
            return identity, result

        checked = await asyncio.gather(*(check(identity) for identity in pending))
        self._cache.update(checked)
        return {identity: self._cache[identity] for identity in normalized}

    async def audit(
        self,
        *,
        doi: str | None,
        pmid: str | None = None,
        arxiv_id: str | None = None,
    ) -> PublicationAudit:
        identity = AuditIdentity.from_values(
            doi=doi,
            pmid=pmid,
            arxiv_id=arxiv_id,
        )
        return (await self.audit_many((identity,)))[identity]


async def _async_none() -> None:
    return None


class PublicationAuditor:
    """Compatibility adapter for isolated synchronous callers and tests."""

    def __init__(
        self,
        client: JsonClient,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.client = client
        self.clock = clock
        self._cache: dict[tuple[str, str, str], PublicationAudit] = {}

    def _get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return self.client.get_json(url, params)

    def audit(
        self,
        *,
        doi: str | None,
        pmid: str | None = None,
        arxiv_id: str | None = None,
        crossref_message: dict[str, Any] | None = None,
    ) -> PublicationAudit:
        cleaned_doi = pdf_evidence.clean_doi(doi or "")
        cleaned_pmid = (pmid or "").strip()
        cleaned_arxiv = (arxiv_id or "").strip()
        key = (cleaned_doi.casefold(), cleaned_pmid, cleaned_arxiv.casefold())
        if key in self._cache:
            return self._cache[key]

        scite_payload = None
        if cleaned_doi:
            if not isinstance(crossref_message, dict):
                crossref_payload = self._get_json(
                    metadata_providers.CROSSREF_WORKS
                    + urllib.parse.quote(cleaned_doi, safe=""),
                    {"mailto": self.client.email} if self.client.email else None,
                )
                candidate = (
                    crossref_payload.get("message")
                    if isinstance(crossref_payload, dict)
                    else None
                )
                if isinstance(candidate, dict):
                    crossref_message = candidate
            scite_payload = self._get_json(
                f"{SCITE_TALLIES}/{urllib.parse.quote(cleaned_doi, safe='')}",
                None,
            )

        result = PublicationAudit(
            retraction=publication_status.assess_retractions(
                crossref_message=crossref_message,
                crossref_checked=isinstance(crossref_message, dict),
                crossref_expected=bool(cleaned_doi),
            ),
            commentary=publication_status.assess_commentary(
                scite_payload=scite_payload,
                scite_checked=isinstance(scite_payload, dict),
                scite_expected=bool(cleaned_doi),
            ),
            checked_at=_checked_at(self.clock),
        )
        self._cache[key] = result
        return result
