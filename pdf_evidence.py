"""Read-only bibliographic evidence extraction from PDF files."""

from __future__ import annotations

import html
import re
import tempfile
import unicodedata
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pymupdf

import pdf_rewrite
import file_metadata


RAW_SCAN_CHUNK = 1024 * 1024
RAW_SCAN_OVERLAP = 256
DOI_BODY_CHARS = r"A-Z0-9._;()/:+\-%<>"
DOI_REGEX = re.compile(
    rf"\b10\.\d{{4,9}}/[{DOI_BODY_CHARS}]+",
    re.IGNORECASE,
)
DOI_URL_REGEX = re.compile(
    rf"https?://(?:dx\.)?doi\.org/(10\.\d{{4,9}}/[{DOI_BODY_CHARS}]+)",
    re.IGNORECASE,
)
DOI_LOOSE_REGEX = re.compile(
    rf"\b10\s*\.\s*\d{{4,9}}\s*/\s*(?:[{DOI_BODY_CHARS}]+\s*){{1,30}}",
    re.IGNORECASE,
)
DOI_LABEL_REGEX = re.compile(
    rf"\bdoi\s*[:=]\s*(10\s*\.\s*\d{{4,9}}\s*/\s*"
    rf"(?:[{DOI_BODY_CHARS}]+\s*){{1,30}})",
    re.IGNORECASE,
)
DOI_REGEX_BYTES = re.compile(
    rb"\b10\.\d{4,9}/[-._;()/:+%<>A-Z0-9]+",
    re.IGNORECASE,
)
DOI_HARD_DELIMITERS = "\"'|\\"
PII_FORMATTED_BODY = (
    r"(?:S\s*)?"
    r"(?:[A-Z0-9]\s*){4}-\s*"
    r"(?:[A-Z0-9]\s*){4}"
    r"\(\s*(?:[A-Z0-9]\s*){2}\)\s*"
    r"(?:[A-Z0-9]\s*){5}-\s*[A-Z0-9]"
)
PII_LABELED_REGEX = re.compile(
    rf"\bPII\b\s*[:=]?\s*("
    rf"{PII_FORMATTED_BODY}|(?:S\s*)?(?:[A-Z0-9]\s*){{16}}"
    rf")",
    re.IGNORECASE,
)
PII_FORMATTED_REGEX = re.compile(
    rf"\b({PII_FORMATTED_BODY})\b",
    re.IGNORECASE,
)
PII_COMPACT_REGEX = re.compile(r"S?[A-Z0-9]{16,20}", re.IGNORECASE)
ARXIV_ID_BODY = r"(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})"
ARXIV_ID_REGEXES = (
    re.compile(
        rf"\barxiv\s*:\s*({ARXIV_ID_BODY})(?:v\d+)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\barxiv\.org/(?:abs|pdf)/({ARXIV_ID_BODY})(?:v\d+)?"
        r"(?:\.pdf)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b10\.48550/arxiv\.({ARXIV_ID_BODY})(?:v\d+)?\b",
        re.IGNORECASE,
    ),
)
TITLE_STOPWORDS = {
    "the", "and", "with", "for", "from", "into", "onto", "upon", "over",
    "under", "between", "among", "about", "of", "in", "on", "at", "by",
    "as", "to", "an", "a", "or", "nor", "but", "is", "are", "was",
    "were", "be", "being", "been", "this", "that", "these", "those",
    "using", "via",
}
PREPRINT_DOI_PREFIXES = (
    "10.1101/",
    "10.21203/",
    "10.31219/osf.io/",
    "10.48550/arxiv.",
)
COVER_PAGE_SCAN_LIMIT = pdf_rewrite.COVER_PAGE_SCAN_LIMIT
UNREPAIRABLE_PDF_STATUSES = frozenset({"pdf_unrepairable"})
TITLE_BOILERPLATE_COMPACT_PREFIXES = (
    "attribution40internationalccby40license",
    "contentslistsavailableatsciencedirect",
)
TITLE_BOILERPLATE_COMPACT_FRAGMENTS = (
    "forpersonaluseonly",
    "scanneddocinpdfformat",
    "scanneddocumentfromhp9100c",
)
TITLE_BOILERPLATE_COMPACT_VALUES = frozenset(
    {
        "editorial",
        "originalarticle",
        "originalcontribution",
        "researcharticle",
        "reviewarticle",
        "technicalnote",
    }
)
TITLE_BOILERPLATE_OCR_MARKERS = frozenset({"1", "i", "l"})
SCANNER_METADATA_MARKERS = (
    "capture plug-in",
    "digital sender",
    "paper capture",
    "scanner",
)
LEADING_ARTICLE_TITLE_MIN_Y_RATIO = 0.33
LEADING_ARTICLE_MIN_WORDS = 80
LEADING_ARTICLE_DOI_SCORE = 40
# Bump this marker whenever cached evidence extraction semantics change.
EVIDENCE_CACHE_VERSION = f"pdf-evidence-v7;pymupdf={pymupdf.VersionBind}"
CLASSIFICATION_TEXT_LIMIT = 150_000
CLASSIFICATION_FRONT_MATTER_LIMIT = 50_000


DISSERTATION_TEXT_PATTERNS = (
    re.compile(
        r"\b(?:doctoral|ph\.?\s*d\.?|masters?|m\.?\s*sc\.?)\s+"
        r"(?:dissertation|thesis)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:dissertation|thesis)\s+submitted\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:a\s+)?(?:dissertation|thesis)\s+presented\s+to\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsubmitted\s+(?:to|in)\b.{0,120}"
        r"\b(?:degree|fulfilment|fulfillment)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\binaugural[- ]?dissertation\b", re.IGNORECASE),
)
PREPRINT_TEXT_PATTERNS = (
    re.compile(r"\b(?:bioRxiv|medRxiv|ChemRxiv)\s+preprint\b", re.IGNORECASE),
    re.compile(r"\bthis\s+preprint\b", re.IGNORECASE),
    re.compile(
        r"\bnot\s+(?:yet\s+)?certified\s+by\s+peer\s+review\b",
        re.IGNORECASE,
    ),
)
PROCEEDING_TEXT_PATTERNS = (
    re.compile(r"\bproceedings\s+of\b", re.IGNORECASE),
    re.compile(r"\bconference\s+proceedings\b", re.IGNORECASE),
    re.compile(r"\bconference\s+paper\b", re.IGNORECASE),
    re.compile(r"\bannual\s+meeting\b", re.IGNORECASE),
    re.compile(r"\bmeeting\s+abstract\b", re.IGNORECASE),
    re.compile(r"\bextended\s+abstract\b", re.IGNORECASE),
    re.compile(r"\bISMRM\b.{0,80}\babstract\b", re.IGNORECASE | re.DOTALL),
)
ISMRM_PROCEEDINGS_FOOTER_PATTERN = re.compile(
    r"\bProc\.?\s+Intl\.?\s+Soc\.?\s+Mag\.?\s+Reson\.?\s+Med\.?"
    r"\s+\d+\s*\(\d{4}\)",
    re.IGNORECASE,
)
DIGITAL_POSTER_PATTERN = re.compile(
    r"\bProgram\s+No\s*:\s*\d+\b.{0,120}"
    r"\bPoster\s+Type\s*:\s*(?:Digital\s+)?Poster\b",
    re.IGNORECASE | re.DOTALL,
)
EUCAP_PROCEEDINGS_PATTERN = re.compile(
    r"\bEuropean\s+Conference\s+on\s+Antennas\s+and\s+Propagation\b"
    r".{0,80}\bEuCAP\s+\d{4}\b",
    re.IGNORECASE | re.DOTALL,
)

ISMRM_CITATION_PATTERN = re.compile(
    r"\bProc\.?\s+ISMRM\b",
    re.IGNORECASE,
)
LEGACY_ABSTRACT_HEADING_PATTERNS = (
    re.compile(r"\bintroduction\b", re.IGNORECASE),
    re.compile(r"\bmethods?(?:\s+and\s+materials?)?\b", re.IGNORECASE),
    re.compile(r"\bresults?\b", re.IGNORECASE),
    re.compile(r"\breferences\b", re.IGNORECASE),
)
STRUCTURED_ABSTRACT_HEADING_PATTERNS = (
    re.compile(r"\bintroduction\b", re.IGNORECASE),
    re.compile(r"\bresults?\b", re.IGNORECASE),
    re.compile(r"\b(?:discussion|conclusions?)\b", re.IGNORECASE),
    re.compile(r"\breferences\b", re.IGNORECASE),
)
VENDOR_NAME_PATTERN = re.compile(
    r"\b(?:Siemens(?:\s+Healthineers)?|Philips(?:\s+Healthcare)?|"
    r"GE\s+HealthCare|Canon\s+Medical|Toshiba\s+Medical|"
    r"Bruker(?:\s+BioSpin)?|Agilent|Varian)\b",
    re.IGNORECASE,
)
VENDOR_DOCUMENT_PATTERN = re.compile(
    r"\b(?:white\s+paper|application\s+note|product\s+brochure|"
    r"user\s+manual|operator(?:s|'s)?\s+manual|service\s+manual|"
    r"technical\s+guide)\b",
    re.IGNORECASE,
)

try:
    pymupdf.TOOLS.mupdf_display_errors(False)  # type: ignore[attr-defined]
except Exception:
    pass


@dataclass(frozen=True, slots=True)
class DoiCandidate:
    doi: str
    source: str
    score: int


InsertedCoverPage = pdf_rewrite.InsertedCoverPage


@dataclass(frozen=True, slots=True)
class ClassificationMarkers:
    ismrm_footer: bool = False
    numbered_poster: bool = False
    eucap: bool = False
    ismrm_citation: bool = False
    structured_abstract: bool = False
    legacy_abstract: bool = False
    dissertation: bool = False
    preprint: bool = False
    proceedings: bool = False
    vendor_document: bool = False
    vendor_name: bool = False
    white_paper: bool = False


def extract_classification_markers(
    title: str, text_samples: Sequence[str],
) -> ClassificationMarkers:
    """Extract path-independent facts without deciding a document category."""
    sample = text_samples[0].split("\f", 1)[0] if text_samples else ""
    front = "\n".join(value for value in (title, sample) if value)[:CLASSIFICATION_FRONT_MATTER_LIMIT]
    document = "\n".join(text_samples)[:CLASSIFICATION_TEXT_LIMIT]
    return ClassificationMarkers(
        ismrm_footer=bool(ISMRM_PROCEEDINGS_FOOTER_PATTERN.search(document)),
        numbered_poster=bool(DIGITAL_POSTER_PATTERN.search(front)),
        eucap=bool(EUCAP_PROCEEDINGS_PATTERN.search(front)),
        ismrm_citation=bool(ISMRM_CITATION_PATTERN.search(front)),
        structured_abstract=all(pattern.search(front) for pattern in STRUCTURED_ABSTRACT_HEADING_PATTERNS),
        legacy_abstract=all(pattern.search(front) for pattern in LEGACY_ABSTRACT_HEADING_PATTERNS),
        dissertation=any(pattern.search(front) for pattern in DISSERTATION_TEXT_PATTERNS),
        preprint=any(pattern.search(front) for pattern in PREPRINT_TEXT_PATTERNS) or bool(arxiv_source_markers(sample)),
        proceedings=any(pattern.search(front) for pattern in PROCEEDING_TEXT_PATTERNS),
        vendor_document=bool(VENDOR_DOCUMENT_PATTERN.search(front)),
        vendor_name=bool(VENDOR_NAME_PATTERN.search(front)),
        white_paper=bool(re.search(r"\bwhite\s+paper\b", front, re.IGNORECASE)),
    )


@dataclass(frozen=True, slots=True)
class PdfEvidence:
    dois: list[DoiCandidate]
    title: str | None
    pii: str | None
    arxiv_doi: str | None
    pdf_status: str = "ok"
    pdf_note: str = ""
    needs_repair: bool = False
    needs_decryption: bool = False
    has_signatures: bool = False
    text_samples: tuple[str, ...] = ()
    inserted_cover_pages: tuple[InsertedCoverPage, ...] = ()
    source_token: file_metadata.FileToken | None = None
    classification_markers: ClassificationMarkers | None = None

    @property
    def needs_rewrite(self) -> bool:
        return bool(
            self.needs_repair
            or self.needs_decryption
            or self.inserted_cover_pages
        )


def _classification_markers(evidence: PdfEvidence) -> ClassificationMarkers:
    return evidence.classification_markers or extract_classification_markers(
        evidence.title or "", evidence.text_samples,
    )


def _classification_markers_from_payload(value: object) -> ClassificationMarkers:
    expected = ClassificationMarkers.__dataclass_fields__.keys()
    if (
        not isinstance(value, dict) or value.keys() != expected
        or any(type(item) is not bool for item in value.values())
    ):
        raise ValueError("cached classification markers are invalid")
    return ClassificationMarkers(**value)


def without_text_samples(evidence: PdfEvidence) -> PdfEvidence:
    """Drop worker text while preserving compact classification facts."""
    return replace(evidence, text_samples=(), classification_markers=_classification_markers(evidence))


def evidence_to_cache_payload(
    evidence: PdfEvidence,
) -> dict[str, object]:
    """Serialize durable evidence without retaining large extracted text."""
    return {
        "_source_token": list(evidence.source_token) if evidence.source_token is not None else None,
        "classification_markers": asdict(_classification_markers(evidence)),
        "dois": [
            {
                "doi": candidate.doi,
                "source": candidate.source,
                "score": candidate.score,
            }
            for candidate in evidence.dois
        ],
        "title": evidence.title,
        "pii": evidence.pii,
        "arxiv_doi": evidence.arxiv_doi,
        "pdf_status": evidence.pdf_status,
        "pdf_note": evidence.pdf_note,
        "needs_repair": evidence.needs_repair,
        "needs_decryption": evidence.needs_decryption,
        "has_signatures": evidence.has_signatures,
        "inserted_cover_pages": [
            {
                "page_index": page.page_index,
                "kind": page.kind,
                "reason": page.reason,
            }
            for page in evidence.inserted_cover_pages
        ],
    }


def _cached_optional_string(
    payload: Mapping[str, object],
    field: str,
) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"cached evidence field {field!r} is not a string")
    return value


def _cached_boolean(
    payload: Mapping[str, object],
    field: str,
) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"cached evidence field {field!r} is not boolean")
    return value


def evidence_from_cache_payload(
    payload: Mapping[str, object],
) -> PdfEvidence:
    """Rebuild validated evidence from an identity-cache payload."""
    raw_dois = payload.get("dois")
    if not isinstance(raw_dois, list):
        raise ValueError("cached DOI evidence is not a list")
    dois: list[DoiCandidate] = []
    for item in raw_dois:
        if not isinstance(item, dict):
            raise ValueError("cached DOI candidate is not an object")
        doi = item.get("doi")
        source = item.get("source")
        score = item.get("score")
        if not isinstance(doi, str) or not clean_doi(doi):
            raise ValueError("cached DOI candidate is invalid")
        if not isinstance(source, str) or not source:
            raise ValueError("cached DOI source is invalid")
        if type(score) is not int or score < 0:
            raise ValueError("cached DOI score is invalid")
        dois.append(DoiCandidate(clean_doi(doi), source, score))

    raw_covers = payload.get("inserted_cover_pages")
    if not isinstance(raw_covers, list):
        raise ValueError("cached inserted-cover evidence is not a list")
    inserted_cover_pages: list[InsertedCoverPage] = []
    for item in raw_covers:
        if not isinstance(item, dict):
            raise ValueError("cached inserted-cover item is not an object")
        page_index = item.get("page_index")
        kind = item.get("kind")
        reason = item.get("reason")
        if type(page_index) is not int or page_index < 0:
            raise ValueError("cached inserted-cover page index is invalid")
        if not isinstance(kind, str) or not kind:
            raise ValueError("cached inserted-cover kind is invalid")
        if not isinstance(reason, str) or not reason:
            raise ValueError("cached inserted-cover reason is invalid")
        inserted_cover_pages.append(
            InsertedCoverPage(page_index, kind, reason)
        )

    pdf_status = payload.get("pdf_status")
    pdf_note = payload.get("pdf_note")
    if not isinstance(pdf_status, str) or not pdf_status:
        raise ValueError("cached PDF status is invalid")
    if not isinstance(pdf_note, str):
        raise ValueError("cached PDF note is invalid")
    raw_token = payload.get("_source_token")
    if raw_token is not None and (
        not isinstance(raw_token, list) or len(raw_token) != 4
        or any(type(part) is not int or part < 0 for part in raw_token)
    ):
        raise ValueError("cached source-state token is invalid")
    return PdfEvidence(
        dois=dois,
        title=_cached_optional_string(payload, "title"),
        pii=_cached_optional_string(payload, "pii"),
        arxiv_doi=_cached_optional_string(payload, "arxiv_doi"),
        pdf_status=pdf_status,
        pdf_note=pdf_note,
        needs_repair=_cached_boolean(payload, "needs_repair"),
        needs_decryption=_cached_boolean(payload, "needs_decryption"),
        has_signatures=_cached_boolean(payload, "has_signatures"),
        classification_markers=_classification_markers_from_payload(payload.get("classification_markers")),
        source_token=tuple(raw_token) if raw_token is not None else None,
        inserted_cover_pages=tuple(inserted_cover_pages),
    )


# Compatibility name retained for callers that construct the previous type.
PdfHints = PdfEvidence


@dataclass(slots=True)
class _PdfAccess:
    document: Any | None
    needs_repair: bool = False
    needs_decryption: bool = False
    has_signatures: bool = False
    status: str = "ok"
    note: str = ""


detect_inserted_cover_pages = pdf_rewrite.detect_inserted_cover_pages


def _html_unescape(value: str) -> str:
    try:
        return html.unescape(value or "")
    except Exception:
        return value or ""


def _trim_doi_tail(doi: str) -> str:
    doi = (doi or "").strip().strip(DOI_HARD_DELIMITERS)
    doi = re.split(r"<<|>>", doi, maxsplit=1)[0]
    doi = re.split(
        r"</?[A-Z_][A-Z0-9_.:-]*(?:\s[^>]*)?>",
        doi,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    pairs = {")": "(", "]": "[", "}": "{"}

    while True:
        previous = doi
        doi = re.sub(
            r"(?:>>?\s*)?(?:endobj|endstream|stream|obj)\d*$",
            "",
            doi,
            flags=re.IGNORECASE,
        ).rstrip()
        while doi and doi[-1] in ".,;:":
            doi = doi[:-1]
        if doi.endswith(">") and doi.count(">") > doi.count("<"):
            doi = doi[:-1].rstrip(".,;:")
        while (
            doi
            and doi[-1] in pairs
            and doi.count(doi[-1]) > doi.count(pairs[doi[-1]])
        ):
            doi = doi[:-1].rstrip(".,;:")
        if doi == previous:
            break

    return doi.strip().strip(DOI_HARD_DELIMITERS)


def clean_doi(doi: str) -> str:
    doi = _html_unescape(str(doi or "")).strip()
    try:
        doi = urllib.parse.unquote(doi)
    except Exception:
        pass
    doi = re.sub(r"^doi\s*[:=]\s*", "", doi, flags=re.IGNORECASE)
    doi = re.sub(
        r"^https?://(?:dx\.)?doi\s*\.\s*org\s*/\s*",
        "",
        doi,
        flags=re.IGNORECASE,
    )
    doi = re.sub(r"^doi\s+org\s*/\s*", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", doi)
    doi = re.sub(r"\s+", "", doi)
    doi = re.sub(
        r"^10\.\s*(\d{4,9})\s*/",
        r"10.\1/",
        doi,
        flags=re.IGNORECASE,
    )
    return _trim_doi_tail(doi)


def clean_pii(pii: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(pii or ""))
    normalized = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", normalized)
    compact = re.sub(r"[^A-Z0-9]", "", normalized.upper())
    return compact if PII_COMPACT_REGEX.fullmatch(compact) else ""


def extract_piis_from_string(text: str) -> list[str]:
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text)

    def collect(expression: re.Pattern[str]) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        for match in expression.finditer(normalized):
            pii = clean_pii(match.group(1))
            if pii and pii not in seen:
                seen.add(pii)
                found.append(pii)
        return found

    labeled = collect(PII_LABELED_REGEX)
    return labeled or collect(PII_FORMATTED_REGEX)


def _unique_pii_from_texts(texts: list[str] | tuple[str, ...]) -> str | None:
    candidates = {
        pii
        for text in texts
        for pii in extract_piis_from_string(text)
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _is_plausible_doi(doi: str) -> bool:
    if not doi or not re.fullmatch(
        rf"10\.\d{{4,9}}/[{DOI_BODY_CHARS}]+",
        doi,
        re.IGNORECASE,
    ):
        return False
    suffix = doi.split("/", 1)[1]
    if len(suffix) < 3 or suffix[-1] in "-_/:(":
        return False
    if re.match(r"^\(?issn\)?\d{4}-?\d{3}[\dX]", suffix, re.IGNORECASE):
        return False
    return not re.search(
        r"(?:abstract|introduction|references|copyright|received|accepted)$",
        suffix,
        re.IGNORECASE,
    )


def _doi_candidate_variants(raw: str) -> list[str]:
    raw = _html_unescape(raw or "")
    variants: list[str] = []
    tokens = [token for token in re.split(r"\s+", raw.strip()) if token]
    for index in range(1, min(len(tokens), 8) + 1):
        variants.append("".join(tokens[:index]))
        current = clean_doi(variants[-1])
        if (
            _is_plausible_doi(current)
            and not tokens[index - 1].endswith(
                ("-", "_", "/", ":", ";", "+", "%", "(")
            )
        ):
            break
    first_token = clean_doi(tokens[0]) if tokens else ""
    first_token_is_complete = bool(
        _is_plausible_doi(first_token)
        and not tokens[0].endswith(("-", "_", "/", ":", ";", "+", "%", "("))
    )
    if not first_token_is_complete:
        variants.extend(
            (
                raw,
                re.sub(
                    r"(?<=[A-Z0-9._;()/:+\-%])[\r\n\f]+"
                    r"(?=[A-Z0-9._;()/:+\-%])",
                    "",
                    raw,
                    flags=re.IGNORECASE,
                ),
            )
        )

    boundary_words = (
        "Received", "Accepted", "Available", "Abstract", "Keywords",
        "Introduction", "Frontiers", "CITATION", "CITATIONS",
        "DigitalObject", "Digital", "Objective", "Inorder", "Designof",
        "Viewonline", "View", "ORIGINALRESEARCH", "ORIGINAL", "Advance",
        "Access", "Publication", "Date", "brought", "doi:bioRxiv", "www",
    )
    preferred: list[str] = []
    for variant in variants:
        cleaned = clean_doi(variant)
        if not cleaned:
            continue
        alternatives = (
            re.sub(
                r"(?<=\d)(?:" + "|".join(boundary_words) + r").*$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ),
            re.sub(
                r"/(?:full|abstract|pdf|epdf|article|xml)$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ),
            re.sub(r"/-/DC\d+$", "", cleaned, flags=re.IGNORECASE),
        )
        preferred.extend(item for item in alternatives if item != cleaned)

    result: list[str] = []
    seen: set[str] = set()
    for variant in preferred + variants:
        doi = clean_doi(variant)
        key = doi.lower()
        if _is_plausible_doi(doi) and key not in seen:
            seen.add(key)
            result.append(doi)
    return result


def _doi_scan_variants(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", _html_unescape(text or ""))
    text = text.replace("\u00ad", "")
    compact_prefixes = re.sub(
        r"\b10\s*\.\s*(\d{4,9})\s*/\s*",
        r"10.\1/",
        text,
        flags=re.IGNORECASE,
    )
    compact_urls = re.sub(
        r"https?://(?:dx\.)?doi\s*\.\s*org\s*/\s*",
        "https://doi.org/",
        text,
        flags=re.IGNORECASE,
    )
    return list(
        dict.fromkeys(
            item
            for item in (text, compact_prefixes, compact_urls)
            if item
        )
    )


def extract_dois_from_string(text: str, limit: int = 20) -> list[str]:
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for variant in _doi_scan_variants(text):
        for expression in (
            DOI_LABEL_REGEX,
            DOI_URL_REGEX,
            DOI_REGEX,
            DOI_LOOSE_REGEX,
        ):
            for match in expression.finditer(variant):
                raw = match.group(1) if match.lastindex else match.group(0)
                for doi in _doi_candidate_variants(raw):
                    key = doi.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    found.append(doi)
                    if len(found) >= limit:
                        return found
    return found


def extract_doi_from_string(text: str) -> str | None:
    dois = extract_dois_from_string(text, limit=1)
    return dois[0] if dois else None


def dois_equal(left: str | None, right: str | None) -> bool:
    return clean_doi(left or "").lower() == clean_doi(right or "").lower()


def _flatten_text_dict(value: Any) -> str:
    chunks: list[str] = []
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text)
        for key in ("blocks", "lines", "spans"):
            nested = value.get(key)
            if nested:
                chunks.append(_flatten_text_dict(nested))
    elif isinstance(value, list):
        for item in value:
            chunks.append(_flatten_text_dict(item))
    return "\n".join(chunk for chunk in chunks if chunk)


def page_text_variants(page: Any) -> list[str]:
    variants: list[str] = []
    get_text = getattr(page, "get_text", None) or getattr(page, "getText", None)
    if not get_text:
        return variants
    for mode in ("text", "blocks", "dict"):
        try:
            data = get_text(mode)
        except Exception:
            continue
        if isinstance(data, str):
            text = data
        elif mode == "blocks" and isinstance(data, list):
            text = "\n".join(
                str(block[4])
                for block in data
                if isinstance(block, (list, tuple)) and len(block) > 4
            )
        else:
            text = _flatten_text_dict(data)
        if text and text.strip():
            variants.append(text)
    return list(dict.fromkeys(variants))


_page_text_variants = page_text_variants


def _page_plain_text(page: Any) -> str:
    variants = page_text_variants(page)
    return variants[0] if variants else ""


def _page_has_dominant_raster_image(
    page: Any,
    *,
    minimum_coverage: float = 0.5,
) -> bool:
    """Return whether a raster image covers most of the visible page."""
    try:
        page_rect = page.rect
        page_area = float(page_rect.get_area())
        if page_area <= 0:
            return False
        for image in page.get_images(full=True):
            xref = image[0]
            for image_rect in page.get_image_rects(xref):
                visible_rect = image_rect & page_rect
                if float(visible_rect.get_area()) / page_area >= minimum_coverage:
                    return True
    except Exception:
        return False
    return False


def _document_is_image_only(
    document: Any,
    page_numbers: list[int],
    page_text_cache: dict[int, list[str]],
) -> bool:
    if not page_numbers:
        return False
    if any(
        text.strip()
        for page_number in page_numbers
        for text in page_text_cache.get(page_number, ())
    ):
        return False
    return any(
        _page_has_dominant_raster_image(document[page_number])
        for page_number in page_numbers[:3]
    )


def extract_dois_from_bytes(pdf_path: Path, limit: int = 20) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    try:
        with pdf_path.open("rb") as handle:
            tail = b""
            while True:
                chunk = handle.read(RAW_SCAN_CHUNK)
                if not chunk:
                    break
                data = tail + chunk
                for match in DOI_REGEX_BYTES.finditer(data):
                    doi = clean_doi(
                        match.group(0).decode("latin-1", errors="ignore")
                    )
                    key = doi.lower()
                    if _is_plausible_doi(doi) and key not in seen:
                        seen.add(key)
                        found.append(doi)
                        if len(found) >= limit:
                            return found
                text = data.decode("latin-1", errors="ignore")
                for doi in extract_dois_from_string(text, limit=limit):
                    key = doi.lower()
                    if key not in seen:
                        seen.add(key)
                        found.append(doi)
                        if len(found) >= limit:
                            return found
                tail = (
                    data[-RAW_SCAN_OVERLAP:]
                    if len(data) > RAW_SCAN_OVERLAP
                    else data
                )
    except Exception:
        pass
    return found


def extract_doi_from_bytes(pdf_path: Path) -> str | None:
    dois = extract_dois_from_bytes(pdf_path, limit=1)
    return dois[0] if dois else None


def clean_title_candidate(title: str) -> str:
    title = _html_unescape(title or "")
    title = re.sub(r"<[^>]+>", " ", title)
    title = re.sub(
        r"^(?:r\s+e\s+s\s+e\s+a\s+r\s+c\s+h\s+a\s+r\s+t\s+i\s+c\s+l\s+e|"
        r"a\s+r\s+t\s+i\s+c\s+l\s+e)\s*[:.-]?\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\b(?:doi|https?://doi\.org|https?://dx\.doi\.org)\S+",
        " ",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\s+\bConference\s+Paper\b[^\d]{0,30}(?:19|20)\d{2}\b.*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\s+\b(?:CITATIONS?|READS)\b.*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    # PDF text extraction commonly separates subscripted scientific symbols.
    title = re.sub(r"\b([A-Za-z])\s+(?=\d+\b)", r"\1", title)
    title = re.sub(r"\s+([,;:)*])", r"\1", title)
    return re.sub(r"\s+", " ", title).strip(" .,:;|-")


_clean_title_candidate = clean_title_candidate


def _is_title_boilerplate(title: str) -> bool:
    normalized = title.casefold().strip()
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    exact_candidates = {compact}
    words = re.findall(r"[a-z0-9]+", normalized)
    if len(words) > 1 and words[0] in TITLE_BOILERPLATE_OCR_MARKERS:
        # Scanned bullets commonly appear as a leading "l", "I", or "1" token.
        exact_candidates.add("".join(words[1:]))
    return bool(
        exact_candidates & TITLE_BOILERPLATE_COMPACT_VALUES
        or any(
            compact.startswith(prefix)
            for prefix in TITLE_BOILERPLATE_COMPACT_PREFIXES
        )
        or any(
            fragment in compact
            for fragment in TITLE_BOILERPLATE_COMPACT_FRAGMENTS
        )
    )


def _looks_like_title(title: str) -> bool:
    candidate = clean_title_candidate(title)
    if not (20 <= len(candidate) <= 300):
        return False
    if len(re.findall(r"[A-Za-z]", candidate)) < 12:
        return False
    if _is_title_boilerplate(candidate):
        return False
    if re.search(r"<\s*img\b|data:image|base64", candidate, re.IGNORECASE):
        return False
    tokens = candidate.split()
    if (
        len(candidate) >= 80
        and tokens
        and max(len(token) for token in tokens) >= 40
        and all(re.fullmatch(r"[A-Za-z0-9+/=]+", token) for token in tokens)
    ):
        return False
    if re.search(
        r"^oup_[a-z0-9_]+\b|\b\d+\.\.\d+\s*\+*$|\.indd\b|\.qxd\b",
        candidate,
        re.IGNORECASE,
    ):
        return False
    bad = (
        r"^(?:abstract|introduction|references|keywords|article|research article|"
        r"original article)$|downloaded from|all rights reserved|copyright|"
        r"^microsoft word|^untitled|^unknown$|^view pdf$|^table of contents$|"
        r"^art\s*-\s*.*\(r\)$|^pii\s*:|\.pdf$|\.tmp$|\.docx?$|"
        r"https?://|www\.|\b(?:see profile|publications?|citations?|reads)\b|"
        r"\b(?:18|19|20)\d{2}\s*;\s*\d+\s*:\s*\d+|"
        r"\b(?:18|19|20)\d{2}\s*;\s*\d+\s*;\s*\d+|"
        r"\bpublished\s+online\b|"
        r"\b(?:18|19|20)\d{2}\)?\s+\d+\s*[-–]\s*\d+|^\d+\.\s+|"
        r"journal of|volume \d+|vol\.|issue \d+"
    )
    return not re.search(bad, candidate, re.IGNORECASE)


looks_like_title = _looks_like_title


def _title_quality(title: str | None) -> float:
    candidate = clean_title_candidate(title or "")
    if not _looks_like_title(candidate):
        return -1.0
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", candidate)
    if len(words) > 28 or re.search(r"[.!?]\s+[A-Z]", candidate):
        return -1.0
    content = [word for word in words if word.lower() not in TITLE_STOPWORDS]
    score = min(len(content), 8) * 0.6 + min(len(candidate), 180) / 120
    if 4 <= len(words) <= 20:
        score += 2.0
    if len(words) > 20:
        score -= (len(words) - 20) * 0.7
    capitalized = sum(1 for word in content if word[0].isupper())
    if content:
        score += 2.5 * capitalized / len(content)
    if candidate[:1].islower():
        score -= 3.0
    if (
        candidate.count("(") != candidate.count(")")
        or candidate.count("[") != candidate.count("]")
    ):
        score -= 3.0
    if len(words) >= 16 and re.search(
        r"\b(?:is|are|was|were|has|have)\b",
        candidate,
        re.IGNORECASE,
    ):
        score -= 4.0
    if re.search(
        r"\b(?:has become|have become|we (?:describe|present|report)|"
        r"here we|this (?:study|paper|work)|is described|are described)\b",
        candidate,
        re.IGNORECASE,
    ):
        score -= 4.0
    if re.search(
        r"\b(?:university|institute|department|laboratory|center|centre|"
        r"correspondence|received|accepted|copyright|keywords?|abstract)\b",
        candidate,
        re.IGNORECASE,
    ):
        score -= 5.0
    capitalized_ratio = capitalized / len(content) if content else 0.0
    if (
        candidate.count(",") >= 3
        or (candidate.count(",") >= 2 and capitalized_ratio >= 0.75)
        or re.search(r"\bet\s+al\.?\b", candidate, re.IGNORECASE)
    ):
        score -= 5.0
    return score


def _prefer_title(
    current: str | None,
    candidate: str | None,
) -> str | None:
    return candidate if _title_quality(candidate) > _title_quality(current) else current


def _title_from_plain_text(text: str) -> str | None:
    indexed_lines: list[tuple[int, str, float]] = []
    for index, raw_line in enumerate((text or "").splitlines()[:50]):
        source_bonus = (
            5.0
            if re.search(
                r"\bConference\s+Paper\b[^\d]{0,30}(?:19|20)\d{2}\b",
                raw_line,
                re.IGNORECASE,
            )
            else 0.0
        )
        line = clean_title_candidate(raw_line.replace("\ufffd", ""))
        if not line:
            continue
        if re.match(
            r"^(?:abstract|introduction|summary|keywords?)\b",
            line,
            re.IGNORECASE,
        ):
            break
        if re.search(
            r"https?://|www\.|copyright|all rights reserved|printed in|"
            r"\b(?:original contribution|technical note|PhD|MD|MSc|BS|BA|DDS)\b|"
            r"\b(?:PII|ISSN)\s*[:A-Z0-9-]|\b\d{4}-\d{3}[0-9X]\b|"
            r"^(?:received|accepted|correspondence|address|departments?|"
            r"affiliations?)\b|\b(?:university|institute|laboratory|center|"
            r"centre)\b|\b(?:18|19|20)\d{2}\s*;\s*\d+\s*:\s*\d+|"
            r"\b(?:18|19|20)\d{2}\)?\s+\d+\s*[-–]\s*\d+",
            line,
            re.IGNORECASE,
        ):
            continue
        if line.count(",") >= 3:
            continue
        if re.fullmatch(
            r"(?:[A-Z][A-Z'-]*\s+){1,4}[A-Z][A-Z'-]*"
            r"\s+(?:AND|&)\s+"
            r"(?:[A-Z][A-Z'-]*\s+){1,4}[A-Z][A-Z'-]*",
            line,
        ):
            continue
        indexed_lines.append((index, line, source_bonus))

    scored: list[tuple[float, int, str]] = []
    for start in range(len(indexed_lines)):
        combined: list[str] = []
        previous_index = indexed_lines[start][0]
        source_bonus = 0.0
        for end in range(start, min(start + 3, len(indexed_lines))):
            line_index, line, line_bonus = indexed_lines[end]
            if end > start and line_index - previous_index > 2:
                break
            previous_index = line_index
            source_bonus = max(source_bonus, line_bonus)
            combined.append(line)
            candidate = clean_title_candidate(" ".join(combined))
            quality = _title_quality(candidate)
            if quality >= 0:
                scored.append(
                    (
                        quality + source_bonus - indexed_lines[start][0] * 0.18,
                        start,
                        candidate,
                    )
                )
            if line.rstrip().endswith(("'", '"')):
                break
    if not scored:
        return None
    scored.sort(
        key=lambda item: (item[0], -item[1], len(item[2])),
        reverse=True,
    )
    return scored[0][2]


def combine_page_text_variants(
    page_variants: list[list[str]],
) -> tuple[str, ...]:
    """Combine cached PyMuPDF page representations into document samples."""
    sample_count = max((len(variants) for variants in page_variants), default=0)
    samples: list[str] = []
    for variant_index in range(sample_count):
        pages = [
            variants[variant_index]
            for variants in page_variants
            if len(variants) > variant_index
            and variants[variant_index].strip()
        ]
        if pages:
            samples.append("\f".join(pages))
    return tuple(dict.fromkeys(samples))


def _span_lines_for_title(page: Any) -> list[tuple[float, float, float, str]]:
    try:
        data = page.get_text("dict")
    except Exception:
        return []
    lines: list[tuple[float, float, float, str]] = []
    for block in data.get("blocks", []):
        if not isinstance(block, dict):
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", []) if isinstance(line, dict) else []
            texts = [
                str(span.get("text", ""))
                for span in spans
                if isinstance(span, dict)
            ]
            text = clean_title_candidate(
                " ".join(item for item in texts if item.strip())
            )
            if not text:
                continue
            sizes = [
                float(span.get("size", 0))
                for span in spans
                if isinstance(span, dict) and span.get("size")
            ]
            size = max(sizes) if sizes else 0.0
            bbox = line.get("bbox") or block.get("bbox") or (0, 0, 0, 0)
            try:
                x0, y0 = float(bbox[0]), float(bbox[1])
            except Exception:
                x0, y0 = 0.0, 0.0
            lines.append((y0, x0, size, text))
    return lines


def _title_from_first_page(
    page: Any,
    text_variants: list[str] | None = None,
) -> str | None:
    try:
        page_height = float(page.rect.height)
    except Exception:
        page_height = 1000.0
    visible_lines = [
        item
        for item in _span_lines_for_title(page)
        if item[0] <= page_height * 0.62
    ]
    lines = [
        item
        for item in visible_lines
        if _looks_like_title(item[3])
        and not _is_title_boilerplate(item[3])
    ]
    if lines:
        max_size = max(item[2] for item in lines)
        threshold = max(max_size * 0.78, max_size - 1.5)
        line_gap_limit = max(max_size * 2.0, page_height * 0.025)
        prominent = [
            item
            for item in lines
            if item[2] >= threshold
            and not _is_title_boilerplate(item[3])
        ]
        prominent = sorted(prominent, key=lambda item: (item[0], item[1]))
        selected = []
        for item in prominent:
            if selected and item[0] - selected[-1][0] > line_gap_limit:
                break
            selected.append(item)
            if len(selected) >= 4:
                break
        if selected:
            last_y = selected[-1][0]
            for item in sorted(
                visible_lines,
                key=lambda value: (value[0], value[1]),
            ):
                if item[0] <= last_y:
                    continue
                if item[0] - last_y > line_gap_limit:
                    break
                if item[2] < threshold or _is_title_boilerplate(item[3]):
                    break
                # A final wrapped title line can be too short to qualify as a
                # standalone title, while still sharing the title font.
                if len(item[3]) >= 20:
                    break
                selected.append(item)
                last_y = item[0]
                if len(selected) >= 6:
                    break
        title = clean_title_candidate(
            " ".join(item[3] for item in selected)
        )
        if _looks_like_title(title):
            return title
    variants = text_variants if text_variants is not None else page_text_variants(page)
    for text in variants[:2]:
        title = _title_from_plain_text(text)
        if title:
            return title
    return None


def _leading_article_dois(page: Any, title: str | None) -> frozenset[str]:
    normalized_title = re.sub(
        r"[^a-z0-9]+",
        " ",
        (title or "").casefold(),
    ).strip()
    if not normalized_title:
        return frozenset()
    lines = _span_lines_for_title(page)
    title_positions = [
        y0
        for y0, _x0, _size, text in lines
        if (
            len(
                normalized_line := re.sub(
                    r"[^a-z0-9]+",
                    " ",
                    text.casefold(),
                ).strip()
            )
            >= 12
            and normalized_line in normalized_title
        )
    ]
    if not title_positions:
        return frozenset()
    title_y = min(title_positions)
    try:
        page_height = float(page.rect.height)
    except Exception:
        return frozenset()
    if title_y < page_height * LEADING_ARTICLE_TITLE_MIN_Y_RATIO:
        return frozenset()
    leading_text = " ".join(
        text for y0, _x0, _size, text in lines if y0 < title_y
    )
    if len(re.findall(r"[A-Za-z]+", leading_text)) < LEADING_ARTICLE_MIN_WORDS:
        return frozenset()
    return frozenset(
        doi.casefold() for doi in extract_dois_from_string(leading_text)
    )


def _metadata_texts(document: Any) -> dict[str, str]:
    texts = {
        (key or "").lower(): (value or "")
        for key, value in (document.metadata or {}).items()
    }
    if hasattr(document, "get_xml_metadata"):
        try:
            value = document.get_xml_metadata()
        except Exception:
            value = ""
        if value:
            texts["get_xml_metadata"] = value
    if hasattr(document, "xref_xml_metadata") and hasattr(
        document,
        "xref_stream",
    ):
        try:
            xref = document.xref_xml_metadata()
            if xref:
                raw = document.xref_stream(xref)
                if isinstance(raw, bytes):
                    texts["xmp"] = raw.decode("utf-8", errors="ignore")
        except Exception:
            pass
    return texts


def _pdf_title_from_metadata(metadata: dict[str, str]) -> str | None:
    for key in ("title", "dc:title"):
        title = clean_title_candidate(metadata.get(key, ""))
        if _looks_like_title(title):
            return title
    scanner_signature = " ".join(
        metadata.get(key, "").casefold()
        for key in ("creator", "producer")
    )
    if any(marker in scanner_signature for marker in SCANNER_METADATA_MARKERS):
        return None
    for key in ("description", "subject"):
        title = clean_title_candidate(metadata.get(key, ""))
        if _looks_like_title(title):
            return title
    return None


def _rank_doi_candidates(
    candidates: dict[str, DoiCandidate],
) -> list[DoiCandidate]:
    return sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.score,
            clean_doi(candidate.doi).lower().startswith(PREPRINT_DOI_PREFIXES),
            candidate.source,
            candidate.doi.lower(),
        ),
    )


def _doi_suffix_matches_pdf_filename(pdf_path: Path, doi: str) -> bool:
    stem = pdf_path.stem.lower()
    if len(stem) < 8:
        return False
    suffix = clean_doi(doi).split("/", 1)[-1].lower()
    return stem in suffix


def _pii_from_filename(pdf_path: Path) -> str | None:
    match = re.fullmatch(
        r"1-s2\.0-(S?[A-Z0-9]{16,20})-main",
        pdf_path.stem,
        re.IGNORECASE,
    )
    if not match:
        return None
    return clean_pii(match.group(1)) or None


def _arxiv_doi_from_filename(pdf_path: Path) -> str | None:
    match = re.fullmatch(
        r"(\d{4}\.\d{4,5})(?:v\d+)?",
        pdf_path.stem,
        re.IGNORECASE,
    )
    return f"10.48550/arXiv.{match.group(1)}" if match else None


def extract_arxiv_dois_from_string(text: str) -> list[str]:
    """Return canonical arXiv DOIs from explicit arXiv identifiers."""
    results: list[str] = []
    seen: set[str] = set()
    for regex in ARXIV_ID_REGEXES:
        for match in regex.finditer(text or ""):
            identifier = match.group(1)
            doi = f"10.48550/arXiv.{identifier}"
            key = doi.casefold()
            if key not in seen:
                seen.add(key)
                results.append(doi)
    return results


def arxiv_source_markers(text: str) -> tuple[str, ...]:
    """Accept standalone source stamps before references, not inline citations."""
    markers = []
    prefix = (
        r"(?:arxiv\s*:\s*|https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/|"
        r"(?:https?://doi\.org/)?10\.48550/arxiv\.)"
    )
    stamp = re.compile(
        prefix + ARXIV_ID_BODY + r"(?:v\d+)?(?:\.pdf)?"
        r"(?:\s*\[[^\]\n]+\])?(?:\s+\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})?",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"(?:references|bibliography|references and notes)\s*[:.]?", line, re.IGNORECASE):
            break
        if stamp.fullmatch(line):
            markers.append(line)
    return tuple(markers)


def _unique_arxiv_doi_from_texts(texts: Iterable[object]) -> str | None:
    candidates = {
        doi.casefold(): doi
        for text in texts
        for doi in extract_arxiv_dois_from_string(str(text or ""))
    }
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _encryption_label(document: Any) -> str:
    try:
        metadata = document.metadata or {}
        return str(metadata.get("encryption") or "").strip()
    except Exception:
        return ""


def _open_pdf(pdf_path: Path, password: str) -> _PdfAccess:
    try:
        document = pymupdf.open(str(pdf_path))
    except Exception as exc:
        return _PdfAccess(
            document=None,
            status="pdf_unreadable",
            note=f"PyMuPDF could not open the PDF: {type(exc).__name__}: {exc}",
        )
    needs_repair = bool(getattr(document, "is_repaired", False))
    needs_decryption = bool(
        getattr(document, "needs_pass", False)
        or getattr(document, "is_encrypted", False)
        or _encryption_label(document)
    )
    if getattr(document, "needs_pass", False):
        authenticated = False
        for candidate in dict.fromkeys((password, "")):
            try:
                if document.authenticate(candidate) > 0:
                    authenticated = True
                    break
            except Exception:
                continue
        if not authenticated or getattr(document, "is_encrypted", False):
            document.close()
            return _PdfAccess(
                document=None,
                needs_repair=needs_repair,
                needs_decryption=True,
                status="pdf_password_required",
                note="PDF authentication failed; set RENAMEPDF_PDF_PASSWORD "
                "to a valid user or owner password and retry",
            )
    return _PdfAccess(
        document=document,
        needs_repair=needs_repair,
        needs_decryption=needs_decryption or bool(_encryption_label(document)),
        has_signatures=pdf_rewrite.pdf_has_signatures(document),
    )


def _required_rewrite_status(
    needs_repair: bool,
    needs_decryption: bool,
    inserted_cover_pages: tuple[InsertedCoverPage, ...] = (),
) -> str:
    if inserted_cover_pages:
        return "pdf_cover_page_removal_needed"
    if needs_repair and needs_decryption:
        return "pdf_repair_and_decryption_needed"
    if needs_repair:
        return "pdf_repair_needed"
    if needs_decryption:
        return "pdf_decryption_needed"
    return "ok"


def inspect_pdf(
    pdf_path: Path,
    *,
    pdf_password: str = "",
    max_text_pages: int = 5,
) -> PdfEvidence:
    """Collect DOI and title evidence without modifying the PDF."""
    source_token = file_metadata.file_token(pdf_path)
    candidates: dict[str, DoiCandidate] = {}
    title: str | None = None
    page_text_cache: dict[int, list[str]] = {}
    text_samples: tuple[str, ...] = ()
    inserted_cover_pages: tuple[InsertedCoverPage, ...] = ()
    first_content_page_title: str | None = None
    pii = _pii_from_filename(pdf_path)
    arxiv_evidence_texts: list[str] = []
    filename_arxiv_doi = _arxiv_doi_from_filename(pdf_path)
    if filename_arxiv_doi:
        arxiv_evidence_texts.append(filename_arxiv_doi)
    preprocessing_directory: tempfile.TemporaryDirectory[str] | None = None
    evidence_pdf_path = pdf_path
    extraction_blocked = False

    def add_dois(
        text: str,
        source: str,
        score: int,
        demoted_dois: frozenset[str] = frozenset(),
    ) -> None:
        for doi in extract_dois_from_string(text, limit=20):
            key = doi.lower()
            candidate_score = (
                min(score, LEADING_ARTICLE_DOI_SCORE)
                if key in demoted_dois
                else score
            )
            current = candidates.get(key)
            if current is None or candidate_score > current.score:
                candidates[key] = DoiCandidate(
                    doi,
                    source,
                    candidate_score,
                )

    def page_texts(document: Any, page_number: int) -> list[str]:
        if page_number not in page_text_cache:
            page_text_cache[page_number] = page_text_variants(
                document[page_number]
            )
        return page_text_cache[page_number]

    access = _open_pdf(pdf_path, pdf_password)
    document = access.document
    pdf_status = access.status
    pdf_note = access.note
    try:
        if document is None:
            raise ValueError(pdf_note or "PyMuPDF could not access the PDF")
        inserted_cover_pages = detect_inserted_cover_pages(
            document,
            text_for_page=lambda page_number: (
                page_texts(document, page_number)[0]
                if page_texts(document, page_number)
                else ""
            ),
        )
        if (
            access.needs_repair
            or access.needs_decryption
            or inserted_cover_pages
        ):
            if access.has_signatures:
                pdf_status = "pdf_signed_rewrite_blocked"
                pdf_note = (
                    "PDF rewrite was skipped because rewriting would "
                    "invalidate a digital signature"
                )
                raise ValueError(pdf_note)
            preprocessing_directory = tempfile.TemporaryDirectory(
                prefix="renamepdf-evidence-"
            )
            evidence_pdf_path = (
                Path(preprocessing_directory.name) / pdf_path.name
            )
            document.close()
            document = None
            try:
                pdf_rewrite.write_validated_copy(
                    pdf_path,
                    evidence_pdf_path,
                    pdf_password=pdf_password,
                    remove_page_indices=tuple(
                        item.page_index for item in inserted_cover_pages
                    ),
                    allow_signed_source=True,
                )
                document = pymupdf.open(str(evidence_pdf_path))
                page_text_cache.clear()
            except Exception as exc:
                extraction_blocked = True
                pdf_status = "pdf_unrepairable"
                pdf_note = (
                    "PyMuPDF could not create a validated temporary copy "
                    "before evidence extraction: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise

        metadata = _metadata_texts(document)
        for key, value in metadata.items():
            if value:
                add_dois(value, f"metadata:{key}", 120)
        title = _pdf_title_from_metadata(metadata)
        arxiv_evidence_texts.extend(metadata.values())
        add_dois(pdf_path.stem.replace("_", "/"), "filename", 85)

        content_page_numbers = list(range(len(document)))
        leading_article_dois: frozenset[str] = frozenset()

        if content_page_numbers:
            first_content_page = content_page_numbers[0]
            arxiv_evidence_texts.extend(
                marker for text in page_texts(document, first_content_page)
                for marker in arxiv_source_markers(text)
            )
            first_page_pii = _unique_pii_from_texts(
                page_texts(document, first_content_page)
            )
            if pii is None:
                pii = first_page_pii
            elif first_page_pii and first_page_pii != pii:
                pii = None
            first_content_page_title = _title_from_first_page(
                document[first_content_page],
                page_texts(document, first_content_page),
            )
            title = _prefer_title(
                title,
                first_content_page_title,
            )
            leading_article_dois = _leading_article_dois(
                document[first_content_page],
                first_content_page_title,
            )

        for page_number in content_page_numbers:
            try:
                for link in document[page_number].get_links():
                    link_score = (
                        115
                        if page_number == content_page_numbers[0]
                        else 75
                        if page_number in content_page_numbers[:5]
                        else 55
                    )
                    add_dois(
                        link.get("uri") or "",
                        f"link:p{page_number + 1}",
                        link_score,
                        (
                            leading_article_dois
                            if page_number == content_page_numbers[0]
                            else frozenset()
                        ),
                    )
            except Exception:
                continue

        if content_page_numbers:
            seen_pages: set[int] = set()
            priority_pages = content_page_numbers[:5]
            if len(content_page_numbers) > 5:
                priority_pages.append(content_page_numbers[-1])
            for content_index, page_number in enumerate(priority_pages):
                if page_number in seen_pages:
                    continue
                seen_pages.add(page_number)
                score = (
                    110
                    if content_index == 0
                    else 75
                    if content_index < 5
                    else 55
                )
                for text in page_texts(document, page_number):
                    add_dois(
                        text,
                        f"text:p{page_number + 1}",
                        score,
                        (
                            leading_article_dois
                            if page_number == first_content_page
                            else frozenset()
                        ),
                    )
            for page_number in content_page_numbers:
                if page_number in seen_pages:
                    continue
                for text in page_texts(document, page_number):
                    add_dois(text, f"text:p{page_number + 1}", 40)
            text_samples = combine_page_text_variants(
                [
                    page_texts(document, page_number)
                    for page_number in content_page_numbers[:max_text_pages]
                ]
            )
            for text in text_samples:
                add_dois(
                    text,
                    f"text:pymupdf:p1-{max_text_pages}",
                    75,
                    leading_article_dois,
                )
            # Combined samples omit empty pages. Recover first-page evidence
            # only from that page's own variants, never the sample's first chunk.
            for first_page_text in page_texts(document, first_content_page):
                add_dois(
                    first_page_text, f"text:p{first_content_page + 1}:sample",
                    100, leading_article_dois,
                )
                if first_content_page_title is None or title is None:
                    title = _prefer_title(
                        title,
                        _title_from_plain_text(first_page_text),
                    )

        if (
            access.needs_repair
            or access.needs_decryption
            or inserted_cover_pages
        ):
            pdf_status = _required_rewrite_status(
                access.needs_repair,
                access.needs_decryption,
                inserted_cover_pages,
            )
            actions: list[str] = []
            if access.needs_repair:
                actions.append("repair PDF structure")
            if access.needs_decryption:
                actions.append("remove PDF encryption")
            if inserted_cover_pages:
                page_numbers = ", ".join(
                    str(item.page_index + 1)
                    for item in inserted_cover_pages
                )
                kinds = ", ".join(
                    item.kind for item in inserted_cover_pages
                )
                actions.append(
                    f"remove inserted cover page(s) {page_numbers} "
                    f"({kinds})"
                )
            pdf_note = (
                f"PyMuPDF {' and '.join(actions)} in a validated temporary "
                "copy before evidence extraction; inspection left the source "
                "unchanged"
            )
        if _document_is_image_only(
            document,
            content_page_numbers,
            page_text_cache,
        ):
            image_only_note = (
                "PDF pages contain dominant raster images but no extractable "
                "text layer; OCR is required to detect visible DOI and title "
                "text"
            )
            if pdf_status == "ok":
                pdf_status = "pdf_image_only"
            pdf_note = (
                f"{pdf_note}; {image_only_note}"
                if pdf_note
                else image_only_note
            )
    except Exception as exc:
        extraction_blocked = True
        if pdf_status == "ok":
            pdf_status = "pdf_unreadable"
            pdf_note = (
                "PyMuPDF could not read the PDF: "
                f"{type(exc).__name__}: {exc}"
            )
    finally:
        if document is not None:
            try:
                document.close()
            except Exception:
                pass

    if not extraction_blocked:
        for doi in extract_dois_from_bytes(evidence_pdf_path):
            key = doi.lower()
            score = (
                85
                if _doi_suffix_matches_pdf_filename(pdf_path, doi)
                else 30
            )
            current = candidates.get(key)
            if current is None or score > current.score:
                candidates[key] = DoiCandidate(doi, "raw-bytes", score)
    if preprocessing_directory is not None:
        preprocessing_directory.cleanup()

    if file_metadata.file_token(pdf_path) != source_token:
        return PdfEvidence(
            dois=[], title=None, pii=None, arxiv_doi=None,
            pdf_status="pdf_source_changed", pdf_note="Source changed during inspection",
        )

    arxiv_doi = _unique_arxiv_doi_from_texts(arxiv_evidence_texts)
    for key, candidate in tuple(candidates.items()):
        if key.startswith("10.48550/arxiv.") and not dois_equal(candidate.doi, arxiv_doi):
            candidates[key] = replace(candidate, score=min(candidate.score, 75))

    return PdfEvidence(
        dois=_rank_doi_candidates(candidates),
        title=title,
        pii=pii,
        arxiv_doi=arxiv_doi,
        pdf_status=pdf_status,
        pdf_note=pdf_note,
        needs_repair=access.needs_repair,
        needs_decryption=access.needs_decryption,
        has_signatures=access.has_signatures,
        text_samples=text_samples,
        inserted_cover_pages=inserted_cover_pages,
        source_token=source_token,
    )


def inspect_pdf_with_rewrite(
    pdf_path: Path,
    *,
    commit: bool = False,
    pdf_password: str = "",
    max_text_pages: int = 5,
) -> PdfEvidence:
    """Inspect a PDF after optional isolated source normalization.

    Read-only inspection always extracts evidence from a validated temporary
    copy when preprocessing is needed. A committed caller installs that same
    class of normalized copy first, then performs a second inspection against
    the installed source. Failed or blocked rewrites return no identifier or
    title evidence so callers cannot match the unprocessed source.
    """
    evidence = inspect_pdf(
        pdf_path,
        pdf_password=pdf_password,
        max_text_pages=max_text_pages,
    )
    if evidence.pdf_status == "pdf_signed_rewrite_blocked":
        return evidence
    if not commit or not evidence.needs_rewrite:
        return evidence

    outcome = pdf_rewrite.rewrite_pdf(
        pdf_path,
        pdf_password=pdf_password,
        remove_page_indices=tuple(
            item.page_index for item in evidence.inserted_cover_pages
        ),
        expected_source_token=evidence.source_token,
    )
    if not outcome.changed:
        return replace(
            evidence,
            dois=[],
            title=None,
            text_samples=(),
            pdf_status=outcome.status,
            pdf_note=outcome.note,
        )

    rewritten = inspect_pdf(
        pdf_path,
        pdf_password=pdf_password,
        max_text_pages=max_text_pages,
    )
    if rewritten.pdf_status in UNREPAIRABLE_PDF_STATUSES:
        return rewritten
    if rewritten.pdf_status == "pdf_image_only":
        return replace(
            rewritten,
            pdf_note=f"{outcome.note}; {rewritten.pdf_note}",
        )
    return replace(
        rewritten,
        pdf_status=outcome.status,
        pdf_note=outcome.note,
    )


def sniff_doi_fast(pdf_path: Path) -> str | None:
    evidence = inspect_pdf(pdf_path)
    return evidence.dois[0].doi if evidence.dois else None
