"""Bibliographic provider adapters for PubMed, Crossref, and DataCite."""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import sqlite3
import ssl
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Awaitable, TypeVar

import httpx

import pdf_evidence
import record_matching
from record_matching import Author, PaperMeta, ProviderRecord

try:
    import truststore
except Exception:
    truststore = None


PMC_IDCONV = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CROSSREF_WORKS = "https://api.crossref.org/works/"
CROSSREF_WORKS_QUERY = "https://api.crossref.org/works"
DATACITE_DOIS = "https://api.datacite.org/dois/"
TITLE_SEARCH_ROWS = 5

class MetadataCache:
    def __init__(self, path: Path):
        self.path = str(path)
        self.con: sqlite3.Connection | None = None
        self.hot_doi2pmid: dict[str, str] = {}
        self.hot_crossref: dict[str, dict[str, Any]] = {}
        self.hot_scite: dict[str, tuple[dict[str, Any], float]] = {}

    def open(self) -> None:
        self.con = sqlite3.connect(self.path, isolation_level=None)
        self.con.execute("PRAGMA journal_mode=WAL;")
        self.con.execute("PRAGMA synchronous=NORMAL;")
        self.con.execute("PRAGMA temp_store=MEMORY;")
        self.con.execute("PRAGMA cache_size=-20000;")
        self.con.execute("PRAGMA busy_timeout=5000;")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS doi2pmid "
            "(doi TEXT PRIMARY KEY, pmid TEXT)"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS crossref "
            "(doi TEXT PRIMARY KEY, payload TEXT)"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS scite_tallies "
            "(doi TEXT PRIMARY KEY, payload TEXT NOT NULL, "
            "checked_at REAL NOT NULL)"
        )

    def close(self) -> None:
        if self.con is not None:
            self.con.close()
            self.con = None

    def _conn(self) -> sqlite3.Connection:
        if self.con is None:
            raise RuntimeError("MetadataCache is not open")
        return self.con

    def get_doi2pmid(self, doi: str) -> str | None:
        cached = self.hot_doi2pmid.get(doi)
        if cached is not None:
            return cached
        row = self._conn().execute(
            "SELECT pmid FROM doi2pmid WHERE doi=?",
            (doi,),
        ).fetchone()
        if row:
            self.hot_doi2pmid[doi] = row[0]
            return row[0]
        return None

    def put_doi2pmid(self, doi: str, pmid: str) -> None:
        self.hot_doi2pmid[doi] = pmid
        self._conn().execute(
            "INSERT INTO doi2pmid(doi,pmid) VALUES(?,?) "
            "ON CONFLICT(doi) DO UPDATE SET pmid=excluded.pmid",
            (doi, pmid),
        )

    def delete_doi2pmid(self, doi: str) -> None:
        self.hot_doi2pmid.pop(doi, None)
        self._conn().execute(
            "DELETE FROM doi2pmid WHERE doi=?",
            (doi,),
        )

    def get_crossref(self, doi: str) -> dict[str, Any] | None:
        cached = self.hot_crossref.get(doi)
        if cached is not None:
            return cached
        row = self._conn().execute(
            "SELECT payload FROM crossref WHERE doi=?",
            (doi,),
        ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row[0])
        except Exception:
            return None
        self.hot_crossref[doi] = payload
        return payload

    def put_crossref(self, doi: str, payload: dict[str, Any]) -> None:
        self.hot_crossref[doi] = payload
        self._conn().execute(
            "INSERT INTO crossref(doi,payload) VALUES(?,?) "
            "ON CONFLICT(doi) DO UPDATE SET payload=excluded.payload",
            (doi, json.dumps(payload, ensure_ascii=False)),
        )

    def get_scite_tally(
        self,
        doi: str,
        *,
        fresh_after: float,
    ) -> dict[str, Any] | None:
        cached = self.hot_scite.get(doi)
        if cached is not None:
            payload, checked_at = cached
            return payload if checked_at >= fresh_after else None
        row = self._conn().execute(
            "SELECT payload, checked_at FROM scite_tallies WHERE doi=?",
            (doi,),
        ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row[0])
            checked_at = float(row[1])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        self.hot_scite[doi] = (payload, checked_at)
        return payload if checked_at >= fresh_after else None

    def put_scite_tally(
        self,
        doi: str,
        payload: dict[str, Any],
        *,
        checked_at: float,
    ) -> None:
        self.hot_scite[doi] = (payload, checked_at)
        self._conn().execute(
            "INSERT INTO scite_tallies(doi,payload,checked_at) VALUES(?,?,?) "
            "ON CONFLICT(doi) DO UPDATE SET "
            "payload=excluded.payload, checked_at=excluded.checked_at",
            (
                doi,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                checked_at,
            ),
        )


CacheDB = MetadataCache


@dataclass(slots=True)
class RunProviderPayloads:
    """Live provider payloads retained only for the current run.

    Persistent bibliographic-cache values are deliberately excluded so a
    publication audit never mistakes an undated cached Crossref record for a
    fresh check.
    """

    _pubmed_by_pmid: dict[str, str] = field(default_factory=dict)
    _pubmed_by_doi: dict[str, str] = field(default_factory=dict)
    _crossref_by_doi: dict[str, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def _doi_key(doi: str | None) -> str:
        return pdf_evidence.clean_doi(doi or "").casefold()

    @staticmethod
    def _pmid_key(pmid: str | None) -> str:
        return str(pmid or "").strip()

    def remember_pubmed_xml(
        self,
        pmid: str,
        xml_text: str,
        *,
        doi: str | None = None,
    ) -> None:
        if not xml_text:
            return
        if pmid_key := self._pmid_key(pmid):
            self._pubmed_by_pmid[pmid_key] = xml_text
        if doi_key := self._doi_key(doi):
            self._pubmed_by_doi[doi_key] = xml_text

    def pubmed_xml(
        self,
        *,
        pmid: str | None = None,
        doi: str | None = None,
    ) -> str | None:
        if pmid_key := self._pmid_key(pmid):
            if xml_text := self._pubmed_by_pmid.get(pmid_key):
                return xml_text
        if doi_key := self._doi_key(doi):
            return self._pubmed_by_doi.get(doi_key)
        return None

    def remember_crossref_message(self, message: dict[str, Any]) -> None:
        doi_key = self._doi_key(str(message.get("DOI") or ""))
        if doi_key:
            self._crossref_by_doi[doi_key] = message

    def crossref_message(self, doi: str | None) -> dict[str, Any] | None:
        return self._crossref_by_doi.get(self._doi_key(doi))


class AsyncRateLimiter:
    def __init__(self, rate: int, per_seconds: float):
        from collections import deque

        self.rate = rate
        self.per = per_seconds
        self.times = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self.lock:
                now = time.monotonic()
                while self.times and (now - self.times[0]) >= self.per:
                    self.times.popleft()
                if len(self.times) < self.rate:
                    self.times.append(now)
                    return
                wait = self.per - (now - self.times[0])
            await asyncio.sleep(max(0.001, wait))


def build_headers(email: str, accept_xml: bool = False) -> dict[str, str]:
    user_agent = "pdf-renamer-doi-only/2.3"
    if email:
        user_agent += f" (mailto:{email})"
    return {
        "User-Agent": user_agent,
        "Accept": "application/xml" if accept_xml else "application/json",
    }


def build_http_verify_context() -> ssl.SSLContext | bool:
    if truststore is None:
        return True
    try:
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        return True


def retry_after_seconds(
    value: str | None,
    fallback: float,
    *,
    now: datetime | None = None,
    max_delay: float | None = 10.0,
) -> float:
    """Return a nonnegative delay from Retry-After seconds or an HTTP date."""
    def bounded(delay: float) -> float:
        value = max(0.0, float(delay))
        return min(value, max_delay) if max_delay is not None else value

    if not value:
        return bounded(fallback)
    try:
        delay = float(value)
        if not math.isfinite(delay):
            raise ValueError("Retry-After delay must be finite")
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except ValueError:
            delay = fallback
        else:
            if retry_at is None:
                delay = fallback
            else:
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                reference = now or datetime.now(timezone.utc)
                if reference.tzinfo is None:
                    reference = reference.replace(tzinfo=timezone.utc)
                delay = (retry_at - reference).total_seconds()
    return bounded(delay)


async def fetch_http(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | None,
    accept_xml: bool,
    attempts: int,
    limiter: AsyncRateLimiter | None,
    headers: dict[str, str] | None,
) -> Any:
    backoff = 0.5
    for attempt in range(attempts):
        try:
            if limiter:
                await limiter.acquire()
            response = await client.get(
                url,
                params=params,
                headers=headers,
                timeout=30,
            )
            if response.status_code == 404:
                return None
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt == attempts - 1:
                    return None
                retry_after = response.headers.get("Retry-After")
                wait = retry_after_seconds(retry_after, backoff)
                await asyncio.sleep(min(wait + random.random() * 0.25, 10.0))
                backoff *= 2
                continue
            if 400 <= response.status_code < 500:
                return None
            response.raise_for_status()
            return response.text if accept_xml else response.json()
        except (
            httpx.TimeoutException,
            httpx.TransportError,
            httpx.DecodingError,
        ):
            if attempt == attempts - 1:
                return None
            await asyncio.sleep(backoff + random.random() * 0.15)
            backoff *= 2
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    return None


def _dedupe_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in ids:
        value = str(item or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _ncbi_request_params(
    params: dict[str, Any],
    *,
    api_key: str = "",
    email: str = "",
) -> dict[str, Any]:
    result = dict(params)
    if cleaned_api_key := api_key.strip():
        result["api_key"] = cleaned_api_key
    if cleaned_email := email.strip():
        result["email"] = cleaned_email
    return result


def pubmed_doi_terms(doi: str) -> list[str]:
    cleaned = pdf_evidence.clean_doi(doi)
    values = list(dict.fromkeys((cleaned, cleaned.lower())))
    return [
        f'{quote}{value}{quote}[{field}]'
        for field in ("DOI", "AID")
        for value in values
        for quote in ("", '"')
    ]




def _et_text(node: ET.Element | None) -> str | None:
    return (node.text or "").strip() if node is not None and node.text else None


def _et_all_text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    text = " ".join(part.strip() for part in node.itertext() if part and part.strip())
    return re.sub(r"\s+", " ", text).strip() or None


def parse_pubmed_full(xml_text: str) -> PaperMeta | None:
    if not xml_text:
        return None
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return None
    article = root.find(".//PubmedArticle")
    if article is None:
        article = root.find(".//PubmedBookArticle")
    if article is None:
        return None

    citation = article.find("./MedlineCitation")
    article_node = citation.find("./Article") if citation is not None else None
    pmid = _et_text(citation.find("./PMID")) if citation is not None else None

    doi = None
    for article_id in article.findall(".//ArticleIdList/ArticleId"):
        if (article_id.get("IdType") or "").lower() == "doi":
            doi = pdf_evidence.clean_doi(_et_text(article_id) or "") or None
            break

    title = (
        _et_all_text(article_node.find("./ArticleTitle"))
        if article_node is not None
        else ""
    )
    title = pdf_evidence.clean_title_candidate(title or "")

    year_text = (
        _et_text(article_node.find("./Journal/JournalIssue/PubDate/Year"))
        if article_node is not None
        else None
    )
    if not year_text and article_node is not None:
        medline_date = _et_text(
            article_node.find("./Journal/JournalIssue/PubDate/MedlineDate")
        )
        match = re.search(r"(19|20)\d{2}", medline_date or "")
        if match:
            year_text = match.group(0)
    if not year_text:
        match = re.search(r"(19|20)\d{2}", xml_text)
        year_text = match.group(0) if match else "1900"
    try:
        year = int(year_text)
    except Exception:
        year = 1900

    journal = (
        _et_text(citation.find("./MedlineJournalInfo/MedlineTA"))
        if citation is not None
        else None
    )
    if not journal and article_node is not None:
        journal = _et_text(article_node.find("./Journal/ISOAbbreviation"))

    authors: list[Author] = []
    if article_node is not None:
        for node in article_node.findall("./AuthorList/Author"):
            family = _et_text(node.find("./LastName")) or _et_text(
                node.find("./CollectiveName")
            )
            if family:
                authors.append(
                    Author(
                        given=_et_text(node.find("./ForeName")) or "",
                        family=family,
                    )
                )
    if not authors:
        authors = [Author(given="", family="Unknown")]

    corresponding: Author | None = None
    if article_node is not None:
        for node in article_node.findall("./AuthorList/Author"):
            affiliation = " ".join(
                _et_text(item) or ""
                for item in node.findall("./AffiliationInfo/Affiliation")
            )
            if re.search(
                r"corresponding|corresp\.|@[A-Z0-9.-]+\.[A-Z]{2,}",
                affiliation,
                re.IGNORECASE,
            ):
                corresponding = Author(
                    given=_et_text(node.find("./ForeName")) or "",
                    family=(
                        _et_text(node.find("./LastName"))
                        or _et_text(node.find("./CollectiveName"))
                        or ""
                    ),
                )
                break
    if corresponding is None:
        corresponding = authors[-1]

    author_keywords: list[str] = []
    if citation is not None:
        for keyword_list in citation.findall("./KeywordList"):
            if (keyword_list.get("Owner") or "").upper() == "NLM":
                continue
            for keyword in keyword_list.findall("./Keyword"):
                value = _et_text(keyword)
                if value:
                    author_keywords.append(value)

    mesh_keywords: list[str] = []
    if citation is not None:
        for heading in citation.findall("./MeshHeadingList/MeshHeading"):
            value = _et_text(heading.find("./DescriptorName"))
            if value:
                mesh_keywords.append(value)

    return PaperMeta(
        title=title,
        year=year,
        journal_abbrev=journal,
        first_author=authors[0],
        authors=authors,
        corresponding_author=corresponding,
        doi=doi,
        pmid=pmid,
        author_keywords=list(dict.fromkeys(author_keywords))[:8],
        mesh_keywords=list(dict.fromkeys(mesh_keywords))[:8],
    )


def parse_crossref_message(message: dict[str, Any]) -> PaperMeta | None:
    if not message:
        return None
    title = pdf_evidence.clean_title_candidate(
        str((message.get("title") or [""])[0])
    )

    def year_from_date_parts(value: Any) -> int | None:
        parts = value.get("date-parts") if isinstance(value, dict) else None
        if not parts or not isinstance(parts, list) or not parts[0]:
            return None
        try:
            return int(parts[0][0])
        except Exception:
            return None

    year = (
        year_from_date_parts(message.get("issued"))
        or year_from_date_parts(message.get("published-print"))
        or year_from_date_parts(message.get("published-online"))
        or year_from_date_parts(message.get("created"))
        or 1900
    )
    short_container = message.get("short-container-title") or []
    journal = (short_container[0] if short_container else None) or (
        (message.get("container-title") or [""])[0]
    )

    authors: list[Author] = []
    for contributor in message.get("author") or message.get("editor") or []:
        family = contributor.get("family") or ""
        given = contributor.get("given") or ""
        name = contributor.get("name") or ""
        if not family and name:
            parts = name.split()
            family = parts[-1] if parts else ""
            given = " ".join(parts[:-1]) if len(parts) > 1 else ""
        if family or name:
            authors.append(Author(given=given, family=family or "Unknown"))
    if not authors:
        authors = [Author(given="", family="Unknown")]

    return PaperMeta(
        title=title,
        year=year,
        journal_abbrev=journal,
        first_author=authors[0],
        authors=authors,
        corresponding_author=authors[-1],
        doi=pdf_evidence.clean_doi(str(message.get("DOI") or "")) or None,
        pmid=None,
        author_keywords=[
            str(subject) for subject in (message.get("subject") or [])
        ][:8],
        mesh_keywords=[],
    )


def parse_datacite_attributes(attributes: dict[str, Any]) -> PaperMeta | None:
    if not attributes:
        return None
    title = ""
    for item in attributes.get("titles") or []:
        if isinstance(item, dict) and item.get("title"):
            title = pdf_evidence.clean_title_candidate(str(item["title"]))
            break
    try:
        year = int(attributes.get("publicationYear") or 1900)
    except Exception:
        year = 1900

    authors: list[Author] = []
    for creator in attributes.get("creators") or []:
        if not isinstance(creator, dict):
            continue
        family = str(creator.get("familyName") or "").strip()
        given = str(creator.get("givenName") or "").strip()
        name = str(creator.get("name") or "").strip()
        if not family and name:
            if "," in name:
                family, given = (
                    part.strip() for part in name.split(",", 1)
                )
            else:
                parts = name.split()
                family = parts[-1] if parts else ""
                given = " ".join(parts[:-1])
        if family:
            authors.append(Author(given=given, family=family))
    if not authors:
        authors = [Author(given="", family="Unknown")]

    container = (
        attributes.get("container")
        if isinstance(attributes.get("container"), dict)
        else {}
    )
    journal = str(container.get("title") or "").strip()
    publisher = str(attributes.get("publisher") or "").strip()
    types = (
        attributes.get("types")
        if isinstance(attributes.get("types"), dict)
        else {}
    )
    resource_type = str(
        types.get("resourceType") or types.get("resourceTypeGeneral") or ""
    ).strip()
    if not journal:
        journal = (
            publisher
            if publisher and publisher.lower() != "unpublished"
            else resource_type
        )

    subjects = [
        str(item.get("subject")).strip()
        for item in (attributes.get("subjects") or [])
        if isinstance(item, dict) and item.get("subject")
    ]
    return PaperMeta(
        title=title,
        year=year,
        journal_abbrev=journal or None,
        first_author=authors[0],
        authors=authors,
        corresponding_author=authors[-1],
        doi=pdf_evidence.clean_doi(str(attributes.get("doi") or "")) or None,
        pmid=None,
        author_keywords=subjects[:8],
        mesh_keywords=[],
    )


@dataclass(frozen=True, slots=True)
class ProviderLookup:
    records: tuple[ProviderRecord, ...] = ()
    notes: tuple[str, ...] = ()


T = TypeVar("T")


async def _with_semaphore(
    semaphore: asyncio.Semaphore | None,
    operation: Awaitable[T],
) -> T:
    if semaphore is None:
        return await operation
    async with semaphore:
        return await operation


@dataclass(slots=True)
class PubMedAdapter:
    client: httpx.AsyncClient
    cache: MetadataCache
    api_key: str
    email: str
    limiter: AsyncRateLimiter
    headers_json: dict[str, str]
    headers_xml: dict[str, str]
    run_payloads: RunProviderPayloads | None = None

    async def doi_to_pmids(self, doi: str) -> list[str]:
        cleaned = pdf_evidence.clean_doi(doi)
        cached = self.cache.get_doi2pmid(cleaned)
        if cached:
            return [cached]
        payload = await fetch_http(
            self.client,
            PMC_IDCONV,
            _ncbi_request_params(
                {
                    "format": "json",
                    "ids": cleaned,
                    "tool": "pdf_renamer",
                },
                email=self.email,
            ),
            False,
            3,
            self.limiter,
            self.headers_json,
        )
        records = payload.get("records") if payload else None
        if isinstance(records, list) and records:
            pmid = records[0].get("pmid")
            if pmid:
                return [str(pmid)]
        for term in pubmed_doi_terms(cleaned):
            payload = await fetch_http(
                self.client,
                PUBMED_ESEARCH,
                _ncbi_request_params(
                    {
                        "db": "pubmed",
                        "term": term,
                        "retmode": "json",
                        "retmax": 5,
                    },
                    api_key=self.api_key,
                    email=self.email,
                ),
                False,
                3,
                self.limiter,
                self.headers_json,
            )
            found = (
                payload.get("esearchresult", {}).get("idlist", [])
                if payload
                else []
            )
            if found:
                return _dedupe_ids([str(item) for item in found])
        return []

    async def ids_by_title(self, title: str) -> list[str]:
        cleaned = pdf_evidence.clean_title_candidate(title)
        if not pdf_evidence.looks_like_title(cleaned):
            return []
        for term in (f'"{cleaned}"[Title]', f"{cleaned}[Title]"):
            payload = await fetch_http(
                self.client,
                PUBMED_ESEARCH,
                _ncbi_request_params(
                    {
                        "db": "pubmed",
                        "term": term,
                        "retmode": "json",
                        "retmax": TITLE_SEARCH_ROWS,
                        "sort": "relevance",
                    },
                    api_key=self.api_key,
                    email=self.email,
                ),
                False,
                3,
                self.limiter,
                self.headers_json,
            )
            ids = (
                payload.get("esearchresult", {}).get("idlist", [])
                if payload
                else []
            )
            if ids:
                return _dedupe_ids([str(item) for item in ids])
        return []

    async def fetch_meta(
        self,
        pmid: str,
    ) -> tuple[PaperMeta | None, str | None]:
        xml = (
            self.run_payloads.pubmed_xml(pmid=pmid)
            if self.run_payloads is not None
            else None
        )
        if xml is None:
            xml = await fetch_http(
                self.client,
                PUBMED_EFETCH,
                _ncbi_request_params(
                    {
                        "db": "pubmed",
                        "id": pmid,
                        "retmode": "xml",
                    },
                    api_key=self.api_key,
                    email=self.email,
                ),
                True,
                3,
                self.limiter,
                self.headers_xml,
            )
        if not xml:
            return None, None
        if self.run_payloads is not None:
            self.run_payloads.remember_pubmed_xml(pmid, xml)
        # A notice and its linked original are different publication identities.
        # Preserve the requested PMID; formal-update links belong to the audit.
        meta = parse_pubmed_full(xml)
        if meta:
            meta.pmid = pmid
            if self.run_payloads is not None:
                self.run_payloads.remember_pubmed_xml(
                    pmid,
                    xml,
                    doi=meta.doi,
                )
        return meta, pmid if meta else None

    async def records_by_doi(self, doi: str) -> ProviderLookup:
        records: list[ProviderRecord] = []
        notes: list[str] = []
        cleaned = pdf_evidence.clean_doi(doi)
        cached_pmid = self.cache.get_doi2pmid(cleaned)
        if cached_pmid:
            meta, resolved_pmid = await self.fetch_meta(cached_pmid)
            if (
                meta is not None
                and meta.doi
                and pdf_evidence.dois_equal(meta.doi, cleaned)
            ):
                return ProviderLookup(
                    (
                        ProviderRecord(
                            "pubmed",
                            meta,
                            str(resolved_pmid or cached_pmid),
                        ),
                    )
                )
            self.cache.delete_doi2pmid(cleaned)
            returned_doi = meta.doi if meta and meta.doi else "no DOI"
            notes.append(
                "Discarded cached PubMed mapping because PMID "
                f"{cached_pmid} returned {returned_doi}, not {cleaned}"
            )

        for pmid in await self.doi_to_pmids(cleaned):
            meta, resolved_pmid = await self.fetch_meta(pmid)
            if meta is None:
                notes.append(f"PubMed PMID {pmid} returned no record")
                continue
            if (
                meta.doi
                and not pdf_evidence.dois_equal(meta.doi, cleaned)
            ):
                notes.append(
                    f"PubMed PMID {pmid} returned DOI {meta.doi}, "
                    f"not requested DOI {cleaned}"
                )
                continue
            records.append(
                ProviderRecord(
                    "pubmed",
                    meta,
                    str(resolved_pmid or pmid),
                )
            )
            if not record_matching.record_has_enough_metadata(meta):
                notes.append(f"PubMed PMID {pmid} lacked enough metadata")
        return ProviderLookup(tuple(records), tuple(notes))

    async def records_by_title(self, title: str) -> ProviderLookup:
        records: list[ProviderRecord] = []
        for pmid in await self.ids_by_title(title):
            meta, resolved_pmid = await self.fetch_meta(pmid)
            if meta is not None:
                records.append(
                    ProviderRecord(
                        "pubmed",
                        meta,
                        str(resolved_pmid or pmid),
                    )
                )
        return ProviderLookup(tuple(records))

    def remember_doi_pmid(self, doi: str, pmid: str) -> None:
        self.cache.put_doi2pmid(pdf_evidence.clean_doi(doi), pmid)


@dataclass(slots=True)
class CrossrefAdapter:
    client: httpx.AsyncClient
    cache: MetadataCache
    limiter: AsyncRateLimiter
    headers: dict[str, str]
    semaphore: asyncio.Semaphore | None = None
    run_payloads: RunProviderPayloads | None = None

    async def message_by_doi(self, doi: str) -> dict[str, Any] | None:
        cleaned = pdf_evidence.clean_doi(doi)
        fresh = (
            self.run_payloads.crossref_message(cleaned)
            if self.run_payloads is not None
            else None
        )
        if fresh is not None:
            return fresh
        cached = self.cache.get_crossref(cleaned)
        if cached:
            return cached
        payload = await _with_semaphore(
            self.semaphore,
            fetch_http(
                self.client,
                CROSSREF_WORKS + urllib.parse.quote(cleaned, safe=""),
                None,
                False,
                3,
                self.limiter,
                self.headers,
            ),
        )
        message = payload.get("message") if payload else None
        if isinstance(message, dict):
            self.cache.put_crossref(cleaned, message)
            if self.run_payloads is not None:
                self.run_payloads.remember_crossref_message(message)
            return message
        return None

    async def messages_by_title(self, title: str) -> list[dict[str, Any]]:
        cleaned = pdf_evidence.clean_title_candidate(title)
        if not pdf_evidence.looks_like_title(cleaned):
            return []
        payload = await _with_semaphore(
            self.semaphore,
            fetch_http(
                self.client,
                CROSSREF_WORKS_QUERY,
                {
                    "query.title": cleaned,
                    "rows": TITLE_SEARCH_ROWS,
                    "sort": "score",
                    "order": "desc",
                },
                False,
                3,
                self.limiter,
                self.headers,
            ),
        )
        items = payload.get("message", {}).get("items", []) if payload else []
        messages = [item for item in items if isinstance(item, dict)]
        if self.run_payloads is not None:
            for message in messages:
                self.run_payloads.remember_crossref_message(message)
        return messages

    async def messages_by_alternative_id(
        self,
        alternative_id: str,
    ) -> list[dict[str, Any]]:
        cleaned = pdf_evidence.clean_pii(alternative_id)
        if not cleaned:
            return []
        payload = await _with_semaphore(
            self.semaphore,
            fetch_http(
                self.client,
                CROSSREF_WORKS_QUERY,
                {
                    "filter": f"alternative-id:{cleaned}",
                    "rows": 3,
                },
                False,
                3,
                self.limiter,
                self.headers,
            ),
        )
        items = payload.get("message", {}).get("items", []) if payload else []
        messages = [item for item in items if isinstance(item, dict)]
        if self.run_payloads is not None:
            for message in messages:
                self.run_payloads.remember_crossref_message(message)
        return messages

    async def records_by_doi(self, doi: str) -> ProviderLookup:
        message = await self.message_by_doi(doi)
        meta = parse_crossref_message(message or {})
        if meta is not None:
            notes = (
                ("Crossref DOI lookup returned incomplete metadata",)
                if not record_matching.record_has_enough_metadata(meta)
                else ()
            )
            return ProviderLookup(
                (ProviderRecord("crossref", meta),),
                notes,
            )
        if message:
            return ProviderLookup(
                notes=("Crossref DOI lookup returned unparseable metadata",)
            )
        return ProviderLookup(
            notes=("DOI lookup returned no Crossref record",)
        )

    async def records_by_title(self, title: str) -> ProviderLookup:
        records = tuple(
            ProviderRecord("crossref", meta)
            for message in await self.messages_by_title(title)
            if (meta := parse_crossref_message(message)) is not None
        )
        return ProviderLookup(records)

    async def records_by_alternative_id(
        self,
        alternative_id: str,
    ) -> ProviderLookup:
        records: list[ProviderRecord] = []
        expected = pdf_evidence.clean_pii(alternative_id)
        if not expected:
            return ProviderLookup(
                notes=("Publisher alternative identifier was invalid",)
            )
        for message in await self.messages_by_alternative_id(expected):
            raw_identifiers = message.get("alternative-id") or []
            if isinstance(raw_identifiers, str):
                raw_identifiers = [raw_identifiers]
            identifiers = {
                cleaned
                for value in raw_identifiers
                if (cleaned := pdf_evidence.clean_pii(str(value)))
            }
            if expected not in identifiers:
                continue
            meta = parse_crossref_message(message)
            if meta is not None:
                records.append(ProviderRecord("crossref", meta))
        return ProviderLookup(tuple(records))


@dataclass(slots=True)
class DataCiteAdapter:
    client: httpx.AsyncClient
    limiter: AsyncRateLimiter
    headers: dict[str, str]
    semaphore: asyncio.Semaphore | None = None

    async def attributes_by_doi(self, doi: str) -> dict[str, Any] | None:
        payload = await _with_semaphore(
            self.semaphore,
            fetch_http(
                self.client,
                DATACITE_DOIS
                + urllib.parse.quote(pdf_evidence.clean_doi(doi), safe=""),
                None,
                False,
                3,
                self.limiter,
                self.headers,
            ),
        )
        data = payload.get("data") if payload else None
        attributes = data.get("attributes") if isinstance(data, dict) else None
        return attributes if isinstance(attributes, dict) else None

    async def attributes_by_title(self, title: str) -> list[dict[str, Any]]:
        cleaned = pdf_evidence.clean_title_candidate(title)
        if not pdf_evidence.looks_like_title(cleaned):
            return []
        payload = await _with_semaphore(
            self.semaphore,
            fetch_http(
                self.client,
                DATACITE_DOIS.rstrip("/"),
                {
                    "query": f'titles.title:"{cleaned}"',
                    "page[size]": TITLE_SEARCH_ROWS,
                },
                False,
                3,
                self.limiter,
                self.headers,
            ),
        )
        data = payload.get("data") if payload else []
        return [
            item.get("attributes")
            for item in data or []
            if isinstance(item, dict)
            and isinstance(item.get("attributes"), dict)
        ]

    async def records_by_doi(self, doi: str) -> ProviderLookup:
        attributes = await self.attributes_by_doi(doi)
        meta = parse_datacite_attributes(attributes or {})
        if meta is not None:
            notes = (
                ("DataCite DOI lookup returned incomplete metadata",)
                if not record_matching.record_has_enough_metadata(meta)
                else ()
            )
            return ProviderLookup(
                (ProviderRecord("datacite", meta),),
                notes,
            )
        if attributes:
            return ProviderLookup(
                notes=("DataCite DOI lookup returned unparseable metadata",)
            )
        return ProviderLookup(
            notes=("DOI lookup returned no DataCite record",)
        )

    async def records_by_title(self, title: str) -> ProviderLookup:
        records = tuple(
            ProviderRecord("datacite", meta)
            for attributes in await self.attributes_by_title(title)
            if (meta := parse_datacite_attributes(attributes)) is not None
        )
        return ProviderLookup(records)


@dataclass(slots=True)
class MetadataProviders:
    pubmed: PubMedAdapter
    crossref: CrossrefAdapter
    datacite: DataCiteAdapter

    @classmethod
    def build(
        cls,
        client: httpx.AsyncClient,
        cache: MetadataCache,
        *,
        api_key: str,
        email: str,
        ncbi_limiter: AsyncRateLimiter,
        metadata_limiter: AsyncRateLimiter,
        metadata_semaphore: asyncio.Semaphore,
        headers_json: dict[str, str],
        headers_xml: dict[str, str],
        run_payloads: RunProviderPayloads | None = None,
    ) -> MetadataProviders:
        return cls(
            pubmed=PubMedAdapter(
                client,
                cache,
                api_key,
                email,
                ncbi_limiter,
                headers_json,
                headers_xml,
                run_payloads,
            ),
            crossref=CrossrefAdapter(
                client,
                cache,
                metadata_limiter,
                headers_json,
                metadata_semaphore,
                run_payloads,
            ),
            datacite=DataCiteAdapter(
                client,
                metadata_limiter,
                headers_json,
                metadata_semaphore,
            ),
        )

    async def records_by_doi(self, doi: str) -> ProviderLookup:
        results = await asyncio.gather(
            self.pubmed.records_by_doi(doi),
            self.crossref.records_by_doi(doi),
            self.datacite.records_by_doi(doi),
        )
        return ProviderLookup(
            tuple(
                record
                for result in results
                for record in result.records
            ),
            tuple(note for result in results for note in result.notes),
        )

    async def records_by_title(self, title: str) -> ProviderLookup:
        results = await asyncio.gather(
            self.pubmed.records_by_title(title),
            self.crossref.records_by_title(title),
            self.datacite.records_by_title(title),
        )
        return ProviderLookup(
            tuple(
                record
                for result in results
                for record in result.records
            ),
            tuple(note for result in results for note in result.notes),
        )
