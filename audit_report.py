"""Versioned audit report contract shared by audit producers and consumers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import document_classification
import publication_audit

if TYPE_CHECKING:
    import organization_run


AUDIT_SCHEMA_VERSION = "1"
DEFAULT_REPORT_NAME = "0_audit_results.csv"
AUDIT_REPORT_FIELDS = (
    "audit_schema_version",
    "path",
    "relative_path",
    "file_size",
    "mtime_ns",
    "identity_status",
    "source_status",
    "doi",
    "pmid",
    "arxiv_doi",
    "title",
    "pdf_title_hint",
    "year",
    "journal",
    "match_source",
    "confidence",
    "pdf_status",
    "pdf_note",
    "publication_stage",
    "stage_evidence",
    "review_status",
    "review_note",
    *publication_audit.report_field_names(),
    "replacement_action",
    "replacement_eligible",
    "replacement_reason",
    "replacement_target_hint",
    "labels",
    "identity_note",
    "failure_category",
    "classification_note",
)


def _file_identity(root: Path, path: Path) -> tuple[str, int | str, int | str]:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        metadata = path.stat()
    except (OSError, ValueError):
        return "", "", ""
    return relative, metadata.st_size, metadata.st_mtime_ns


def report_row(
    root: Path,
    outcome: organization_run.OrganizationOutcome,
) -> dict[str, object]:
    """Return one versioned audit row without changing the source PDF."""
    relative, size, mtime_ns = _file_identity(root, outcome.source)
    classification = document_classification.classify_audit(
        path=outcome.source,
        status=outcome.status,
        doi=outcome.doi or "",
        arxiv_doi=outcome.arxiv_doi,
        title=outcome.title,
        pdf_title_hint=outcome.pdf_title_hint,
        journal=outcome.journal_abbrev,
        pdf_status=outcome.pdf_status,
        failure_category=outcome.failure_category,
        retraction_status=outcome.retraction_status,
        commentary_status=outcome.commentary_status,
    )
    audit_values: dict[str, object] = {
        "retraction_status": outcome.retraction_status,
        "retraction_evidence": outcome.retraction_evidence,
        "retraction_note": outcome.retraction_note,
        "commentary_status": outcome.commentary_status,
        "pubmed_comment_links": outcome.pubmed_comment_links,
        "scite_total": outcome.scite_total,
        "scite_supporting": outcome.scite_supporting,
        "scite_contradicting": outcome.scite_contradicting,
        "scite_mentioning": outcome.scite_mentioning,
        "scite_unclassified": outcome.scite_unclassified,
        "scite_citing_publications": outcome.scite_citing_publications,
        "commentary_note": outcome.commentary_note,
        "audit_checked_at": outcome.audit_checked_at,
    }
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "path": str(outcome.source),
        "relative_path": relative,
        "file_size": size,
        "mtime_ns": mtime_ns,
        "identity_status": classification.identity_status,
        "source_status": outcome.status,
        "doi": outcome.doi or "",
        "pmid": outcome.pmid or "",
        "arxiv_doi": outcome.arxiv_doi,
        "title": outcome.title,
        "pdf_title_hint": outcome.pdf_title_hint,
        "year": outcome.year,
        "journal": outcome.journal_abbrev,
        "match_source": outcome.match_source,
        "confidence": (
            f"{outcome.confidence:.2f}"
            if outcome.confidence is not None
            else ""
        ),
        "pdf_status": outcome.pdf_status,
        "pdf_note": outcome.pdf_note,
        "publication_stage": classification.publication_stage,
        "stage_evidence": classification.stage_evidence,
        "review_status": classification.review_status,
        "review_note": classification.review_note,
        **audit_values,
        "replacement_action": classification.replacement_action,
        "replacement_eligible": (
            "yes" if classification.replacement_eligible else "no"
        ),
        "replacement_reason": classification.replacement_reason,
        "replacement_target_hint": "",
        "labels": json.dumps(
            classification.labels,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "identity_note": outcome.note,
        "failure_category": outcome.failure_category,
        "classification_note": outcome.classification_note,
    }
