"""Post-match document classification and evidence-backed audit labels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import pdf_evidence
import pdf_rewrite


FailedCategory = Literal[
    "Preprints",
    "Proceedings",
    "Dissertations",
    "VendorDocuments",
    "Others",
]
FAILED_CATEGORIES: tuple[FailedCategory, ...] = (
    "Preprints",
    "Proceedings",
    "Dissertations",
    "VendorDocuments",
    "Others",
)

PREPRINT_PATTERN = re.compile(
    r"(?:^|[_ .-])"
    r"(?:arxiv|biorxiv|medrxiv|chemrxiv|researchsquare|"
    r"research[-_ ]?square|ssrn|pre[-_ ]?print)"
    r"(?:\d|[_ .-]|$)",
    re.IGNORECASE,
)
PROCEEDING_PATTERNS = (
    re.compile(r"(?:^|_)ConferencePaper\d{2}(?:_|$)", re.IGNORECASE),
    re.compile(r"^ISMRM?\d", re.IGNORECASE),
    re.compile(
        r"(?:^|_)ISMRMAnnualMeeting\d{2}(?:_|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\d{4}IEEE.*(?:Conference|Symposium)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\d{4}.*(?:European|International).*(?:Conference|Symposium)",
        re.IGNORECASE,
    ),
    re.compile(r"\d{4}PhotonIcs.*Symposium", re.IGNORECASE),
    re.compile(
        r"(?:^|[_ .-])(?:conference|proceedings|symposium|workshop)"
        r"(?:[_ .-]|$)",
        re.IGNORECASE,
    ),
)
DISSERTATION_FILENAME_PATTERN = re.compile(
    r"(?:"
    r"(?:^|[_ .-])"
    r"(?:doctoral[_ .-]*|phd[_ .-]*|masters?[_ .-]*)?"
    r"(?:dissertation|thesis)"
    r"(?:[_ .-]|$)"
    r"|^Dis_.*(?:komplett|complete)"
    r")",
    re.IGNORECASE,
)
VENDOR_FILENAME_PATTERN = re.compile(
    r"(?:^|[_ .-])"
    r"(?:white[_ .-]*paper|whitepaper|product[_ .-]*brochure|"
    r"user[_ .-]*manual|operator[_ .-]*manual|service[_ .-]*manual|"
    r"magnetom|achieva|ingenia|intera|syngo|signa)"
    r"(?:[_ .-]|\d|$)",
    re.IGNORECASE,
)

NUMERIC_ABSTRACT_FILENAME_PATTERN = re.compile(r"^0?\d{3,5}$")
LEGACY_P_ABSTRACT_FILENAME_PATTERN = re.compile(r"^p\d{4,5}$", re.IGNORECASE)
PREPRINT_DOI_PREFIXES = (
    "10.1101/",
    "10.21203/",
    "10.31219/osf.io/",
    "10.48550/arxiv.",
)


@dataclass(frozen=True, slots=True)
class FailedClassification:
    category: FailedCategory
    reason: str


def classify_preliminary_path(path: Path) -> str | None:
    """Return the preliminary-source kind used by audit and replacement."""
    stem = path.stem
    if PREPRINT_PATTERN.search(stem):
        return "preprint"
    if any(pattern.search(stem) for pattern in PROCEEDING_PATTERNS):
        return "proceeding"
    return None


def _has_preprint_identifier(
    doi_candidates: Sequence[tuple[str, str, int]],
    arxiv_doi: str,
) -> bool:
    if arxiv_doi:
        return True
    for doi, source, score in doi_candidates:
        if score < 100:
            continue
        if not source.startswith(("metadata:", "filename", "link:p1", "text:p1")):
            continue
        if doi.casefold().startswith(PREPRINT_DOI_PREFIXES):
            return True
    return False


def _proceedings_source_reason(
    stem: str,
    markers: pdf_evidence.ClassificationMarkers,
) -> str | None:
    if markers.ismrm_footer:
        return "Document text contains an ISMRM proceedings footer"
    if markers.numbered_poster:
        return "Document front matter identifies a numbered conference poster"
    if markers.eucap:
        return "Document front matter identifies an EuCAP conference paper"
    if (
        NUMERIC_ABSTRACT_FILENAME_PATTERN.fullmatch(stem)
        and markers.ismrm_citation
        and markers.structured_abstract
    ):
        return (
            "Numeric one-page abstract contains structured sections and "
            "ISMRM proceedings citations"
        )
    if (
        LEGACY_P_ABSTRACT_FILENAME_PATTERN.fullmatch(stem)
        and markers.legacy_abstract
    ):
        return (
            "Legacy p-number filename and one-page structured abstract "
            "identify a proceedings document"
        )
    return None


def classify_failed_pdf(
    path: Path,
    *,
    title: str = "",
    text_samples: Sequence[str] = (),
    doi_candidates: Sequence[tuple[str, str, int]] = (),
    arxiv_doi: str = "",
    text_markers: pdf_evidence.ClassificationMarkers | None = None,
) -> FailedClassification:
    """Classify one unresolved PDF without affecting its match decision."""
    stem = path.stem
    if DISSERTATION_FILENAME_PATTERN.search(stem):
        return FailedClassification(
            "Dissertations",
            "Filename identifies a thesis or dissertation",
        )
    if _has_preprint_identifier(doi_candidates, arxiv_doi):
        return FailedClassification(
            "Preprints",
            "High-confidence preprint or arXiv identifier was extracted",
        )
    if PREPRINT_PATTERN.search(stem):
        return FailedClassification(
            "Preprints",
            "Filename identifies a preprint source",
        )
    if any(pattern.search(stem) for pattern in PROCEEDING_PATTERNS):
        return FailedClassification(
            "Proceedings",
            "Filename identifies a conference or proceedings document",
        )
    if VENDOR_FILENAME_PATTERN.search(stem):
        return FailedClassification(
            "VendorDocuments",
            "Filename identifies a vendor product document",
        )

    markers = text_markers or pdf_evidence.extract_classification_markers(title, text_samples)
    proceedings_reason = _proceedings_source_reason(
        stem,
        markers,
    )
    if proceedings_reason:
        return FailedClassification("Proceedings", proceedings_reason)
    if markers.dissertation:
        return FailedClassification(
            "Dissertations",
            "Document front matter identifies a thesis or dissertation",
        )
    if markers.preprint:
        return FailedClassification(
            "Preprints",
            "Document front matter identifies a preprint",
        )
    if markers.proceedings:
        return FailedClassification(
            "Proceedings",
            "Document front matter identifies conference proceedings",
        )
    if (
        markers.vendor_document
        and (markers.vendor_name or markers.white_paper)
    ):
        return FailedClassification(
            "VendorDocuments",
            "Document front matter identifies a vendor publication",
        )
    if (
        re.search(r"180000000\d{6,}", stem)
        and markers.vendor_name
    ):
        return FailedClassification(
            "VendorDocuments",
            "Vendor name corroborates the publisher document identifier",
        )
    return FailedClassification(
        "Others",
        "No reliable preprint, proceedings, dissertation, or vendor marker found",
    )


ACCEPTED_IDENTITY_STATUSES = frozenset(
    {
        "pubmed_resolved",
        "crossref_resolved",
        "datacite_resolved",
        "pubmed_title_resolved",
        "crossref_title_resolved",
        "datacite_title_resolved",
    }
)
FORMAL_REVIEW_STATUSES = frozenset(
    {
        "reported_retracted",
        "expression_of_concern",
        "correction_or_partial_retraction",
        "retracted_then_reinstated_or_republished",
        "republished_or_reinstated",
        "retraction_notice_document",
        "expression_of_concern_notice_document",
    }
)
FIND_PUBLISHED_VERSION = "find_published_version"
FIND_EXACT_CLEAN_COPY = "find_exact_clean_copy"
AUTOMATED_REPLACEMENT_ACTIONS = frozenset(
    {FIND_PUBLISHED_VERSION, FIND_EXACT_CLEAN_COPY}
)


@dataclass(frozen=True, slots=True)
class AuditClassification:
    identity_status: str
    publication_stage: str
    stage_evidence: str
    review_status: str
    review_note: str
    replacement_action: str
    replacement_eligible: bool
    replacement_reason: str
    labels: tuple[str, ...]


def _identity_status(status: str) -> str:
    if status in ACCEPTED_IDENTITY_STATUSES:
        return "accepted"
    if status == "conflicting_provider_records":
        return "provider_conflict"
    if "ambiguous" in status:
        return "ambiguous"
    return "unresolved"


def _holding_category(path: Path) -> str:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part.casefold() != "failed" or index + 1 >= len(parts) - 1:
            continue
        return parts[index + 1].casefold()
    return ""


def _publication_stage(
    *,
    path: Path,
    doi: str,
    arxiv_doi: str,
    journal: str,
    failure_category: str,
    retraction_status: str,
) -> tuple[str, str]:
    if retraction_status in {
        "retraction_notice_document",
        "expression_of_concern_notice_document",
    }:
        return "formal_notice", "Formal-update metadata identifies a notice document"

    holding = _holding_category(path)
    holding_stages = {
        "preprints": "preprint",
        "proceedings": "proceeding",
        "dissertations": "dissertation",
        "vendordocuments": "vendor_document",
    }
    if holding in holding_stages:
        return (
            holding_stages[holding],
            f"Reviewed Failed/{holding} placement",
        )

    preliminary = classify_preliminary_path(path)
    if preliminary:
        return preliminary, "Filename identifies a preliminary source"
    if arxiv_doi or doi.casefold().startswith(
        PREPRINT_DOI_PREFIXES
    ):
        return "preprint", "Accepted identifier belongs to a preprint service"

    normalized_failure = failure_category.casefold()
    failure_stages = {
        "preprints": "preprint",
        "proceedings": "proceeding",
        "dissertations": "dissertation",
        "vendordocuments": "vendor_document",
    }
    if normalized_failure in failure_stages:
        return (
            failure_stages[normalized_failure],
            f"Unresolved-document classification is {failure_category}",
        )
    if journal:
        return "journal_article", "Accepted metadata contains a journal venue"
    return "unknown", "No reliable publication-stage evidence was available"


def _review_status(
    publication_stage: str,
    title_text: str,
) -> tuple[str, str]:
    if publication_stage == "preprint" and re.search(
        r"\breviewed\s+preprint\b",
        title_text,
        re.IGNORECASE,
    ):
        return (
            "reviewed_preprint",
            "The source explicitly identifies itself as a reviewed preprint",
        )
    if publication_stage == "preprint":
        return (
            "explicitly_preliminary",
            "Preprint status is known; peer-review status is not inferred",
        )
    return (
        "peer_review_not_established",
        "Record type or venue alone does not establish peer-review status",
    )


def classify_audit(
    *,
    path: Path,
    status: str,
    doi: str = "",
    arxiv_doi: str = "",
    title: str = "",
    pdf_title_hint: str = "",
    journal: str = "",
    pdf_status: str = "ok",
    failure_category: str = "",
    retraction_status: str = "identity_unresolved",
    commentary_status: str = "identity_unresolved",
) -> AuditClassification:
    """Combine existing evidence without turning absence into proof."""
    identity_status = _identity_status(status)
    publication_stage, stage_evidence = _publication_stage(
        path=path,
        doi=pdf_evidence.clean_doi(doi),
        arxiv_doi=arxiv_doi,
        journal=journal,
        failure_category=failure_category,
        retraction_status=retraction_status,
    )
    review_status, review_note = _review_status(
        publication_stage,
        " ".join((title, pdf_title_hint, path.stem)),
    )

    replacement_action = "none"
    replacement_eligible = False
    replacement_reason = "No evidence-backed replacement action was identified"
    usable_title = pdf_evidence.looks_like_title(title or pdf_title_hint)

    if pdf_status in pdf_rewrite.REWRITE_BLOCKING_STATUSES | {
        "pdf_processing_error",
        "pdf_processing_timeout",
    }:
        replacement_action = "manual_review"
        replacement_reason = (
            "PDF access or preprocessing is blocked; identity must be "
            "recovered before replacement"
        )
    elif retraction_status in FORMAL_REVIEW_STATUSES:
        replacement_action = "review_formal_update"
        replacement_reason = (
            "Formal-update evidence requires direction-aware review before "
            "a successor can be selected"
        )
    elif pdf_status in {
        "pdf_decryption_needed",
        "pdf_repair_needed",
        "pdf_repair_and_decryption_needed",
        "pdf_cover_page_removal_needed",
    }:
        replacement_action = "normalize_pdf"
        replacement_reason = (
            "The existing lossless preprocessing transaction is preferred "
            "over publication replacement"
        )
    elif publication_stage in {"preprint", "proceeding"} and usable_title:
        replacement_action = FIND_PUBLISHED_VERSION
        replacement_eligible = True
        replacement_reason = (
            "Preliminary source has enough title evidence for conservative "
            "published-version resolution"
        )
    elif (
        pdf_status == "pdf_image_only"
        and identity_status == "accepted"
        and bool(pdf_evidence.clean_doi(doi))
    ):
        replacement_action = FIND_EXACT_CLEAN_COPY
        replacement_eligible = True
        replacement_reason = (
            "Accepted DOI can constrain a replacement to the same publication"
        )
    labels = [
        f"identity:{identity_status}",
        f"stage:{publication_stage}",
        f"review:{review_status}",
        f"formal:{retraction_status}",
        f"commentary:{commentary_status}",
        f"document:{pdf_status}",
    ]
    if replacement_action != "none":
        labels.append(f"action:{replacement_action}")
    return AuditClassification(
        identity_status=identity_status,
        publication_stage=publication_stage,
        stage_evidence=stage_evidence,
        review_status=review_status,
        review_note=review_note,
        replacement_action=replacement_action,
        replacement_eligible=replacement_eligible,
        replacement_reason=replacement_reason,
        labels=tuple(labels),
    )
