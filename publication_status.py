"""Interpret formal updates and commentary for accepted publications.

Pure assessments consume provider payloads; they do not fetch records,
identify PDFs, select replacements, or mutate collections.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class RetractionSignal:
    source: str
    kind: str
    related_id: str = ""
    detail: str = ""

    def report_text(self) -> str:
        value = f"{self.source}:{self.kind}"
        if self.related_id:
            value += f":{self.related_id}"
        if self.detail:
            value += f" ({self.detail})"
        return value


@dataclass(frozen=True, slots=True)
class RetractionAssessment:
    status: str
    evidence: tuple[RetractionSignal, ...] = ()
    note: str = ""

    @property
    def evidence_text(self) -> str:
        return "; ".join(signal.report_text() for signal in self.evidence[:12])


def _normalized_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _clean_report_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _crossref_update_kind(entry: dict[str, Any]) -> str | None:
    token = _normalized_token(
        " ".join(
            str(entry.get(key) or "")
            for key in ("type", "label")
        )
    )
    if "reinstat" in token or "restore" in token:
        return "reinstatement"
    if "expressionofconcern" in token:
        return "expression_of_concern"
    if "partialretract" in token:
        return "partial_retraction"
    if "retract" in token or "withdraw" in token:
        return "retraction"
    if "republish" in token:
        return "republication"
    if any(value in token for value in ("correct", "errat", "corrigend")):
        return "correction"
    return None


def _crossref_date(entry: dict[str, Any]) -> str:
    for key in ("updated", "published", "created"):
        value = entry.get(key)
        parts = value.get("date-parts") if isinstance(value, dict) else None
        if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
            continue
        date_parts = [str(part) for part in parts[0][:3] if part is not None]
        if date_parts:
            return "-".join(date_parts)
    return ""


def _crossref_detail(entry: dict[str, Any]) -> str:
    values: list[str] = []
    source = _clean_report_text(entry.get("source"))
    if source:
        values.append(f"source={source}")
    if date := _crossref_date(entry):
        values.append(f"date={date}")
    return ",".join(values)


def crossref_signals(message: dict[str, Any] | None) -> tuple[RetractionSignal, ...]:
    """Return formal update signals attached to one Crossref work."""
    if not isinstance(message, dict):
        return ()
    signals: list[RetractionSignal] = []
    for relation, current_is_notice in (("updated-by", False), ("update-to", True)):
        entries = message.get(relation)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            kind = _crossref_update_kind(entry)
            if kind is None:
                continue
            if current_is_notice:
                kind = {
                    "retraction": "retraction_notice",
                    "expression_of_concern": "expression_of_concern_notice",
                    "partial_retraction": "correction_notice",
                    "correction": "correction_notice",
                    "republication": "republished_version",
                }.get(kind, kind)
            related_id = _clean_report_text(
                entry.get("DOI") or entry.get("doi")
            )
            signals.append(
                RetractionSignal(
                    source="crossref",
                    kind=kind,
                    related_id=related_id,
                    detail=_crossref_detail(entry),
                )
            )
    return _dedupe_signals(signals)


def _pubmed_related_id(node: ET.Element) -> str:
    pmid = node.find("./PMID")
    if pmid is not None and (pmid.text or "").strip():
        return f"PMID:{_clean_report_text(pmid.text)}"
    source = node.find("./RefSource")
    return _clean_report_text(source.text) if source is not None else ""


def pubmed_signals(xml_text: str | None) -> tuple[RetractionSignal, ...]:
    """Return formal update signals attached to one PubMed record."""
    if not xml_text:
        return ()
    try:
        root = ET.fromstring(xml_text)
    except (ET.ParseError, ValueError):
        return ()

    signals: list[RetractionSignal] = []
    publication_types = {
        _normalized_token(node.text)
        for node in root.findall(".//PublicationTypeList/PublicationType")
        if (node.text or "").strip()
    }
    if "retractedpublication" in publication_types:
        signals.append(RetractionSignal("pubmed", "retraction"))
    if publication_types & {
        "retractionnotice",
        "retractionofpublication",
        "retraction",
    }:
        signals.append(RetractionSignal("pubmed", "retraction_notice"))
    if "expressionofconcern" in publication_types:
        signals.append(
            RetractionSignal("pubmed", "expression_of_concern_notice")
        )
    if "publishederratum" in publication_types:
        signals.append(RetractionSignal("pubmed", "correction_notice"))
    if "correctedandrepublishedarticle" in publication_types:
        signals.append(RetractionSignal("pubmed", "republished_version"))

    relation_kinds = {
        "retractionin": "retraction",
        "partialretractionin": "partial_retraction",
        "expressionofconcernin": "expression_of_concern",
        "erratumin": "correction",
        "updatein": "version_update_available",
        "correctedandrepublishedin": "correction",
        "retractedandrepublishedin": "retracted_and_republished",
        "republishedin": "republication",
        "retractionof": "retraction_notice",
        "partialretractionof": "correction_notice",
        "expressionofconcernfor": "expression_of_concern_notice",
        "erratumfor": "correction_notice",
        "updateof": "updated_version",
        "correctedandrepublishedfrom": "republished_version",
        "retractedandrepublishedfrom": "republished_version",
        "republishedfrom": "republished_version",
    }
    for node in root.findall(".//CommentsCorrectionsList/CommentsCorrections"):
        ref_type = _normalized_token(node.get("RefType"))
        kind = relation_kinds.get(ref_type)
        if kind:
            signals.append(
                RetractionSignal(
                    source="pubmed",
                    kind=kind,
                    related_id=_pubmed_related_id(node),
                )
            )
    return _dedupe_signals(signals)


def pubmed_record_available(xml_text: str | None) -> bool:
    if not xml_text:
        return False
    try:
        root = ET.fromstring(xml_text)
    except (ET.ParseError, ValueError):
        return False
    return (
        root.find(".//PubmedArticle") is not None
        or root.find(".//PubmedBookArticle") is not None
    )


def _dedupe_signals(
    signals: Iterable[RetractionSignal],
) -> tuple[RetractionSignal, ...]:
    return tuple(dict.fromkeys(signals))


def assess_retractions(
    *,
    crossref_message: dict[str, Any] | None = None,
    pubmed_xml: str | None = None,
    crossref_checked: bool = False,
    pubmed_checked: bool = False,
    crossref_expected: bool = False,
    pubmed_expected: bool = False,
) -> RetractionAssessment:
    """Classify formal update evidence without changing publication identity."""
    signals = _dedupe_signals(
        (*crossref_signals(crossref_message), *pubmed_signals(pubmed_xml))
    )
    kinds = {signal.kind for signal in signals}

    if "retracted_and_republished" in kinds or (
        "retraction" in kinds
        and kinds & {"reinstatement", "republication", "republished_version"}
    ):
        status = "retracted_then_reinstated_or_republished"
    elif "retraction" in kinds:
        status = "reported_retracted"
    elif "expression_of_concern" in kinds:
        status = "expression_of_concern"
    elif "retraction_notice" in kinds:
        status = "retraction_notice_document"
    elif "expression_of_concern_notice" in kinds:
        status = "expression_of_concern_notice_document"
    elif kinds & {"partial_retraction", "correction", "correction_notice"}:
        status = "correction_or_partial_retraction"
    elif kinds & {"republication", "republished_version", "reinstatement"}:
        status = "republished_or_reinstated"
    elif (crossref_expected and not crossref_checked) or (
        pubmed_expected and not pubmed_checked
    ):
        status = "check_incomplete"
    elif crossref_checked or pubmed_checked:
        status = "no_retraction_signal"
    else:
        status = "check_incomplete"

    version_kinds = {"version_update_available", "updated_version"}
    if kinds - version_kinds:
        note = (
            "Formal publication-update metadata was found; review the linked "
            "notice identifiers before acting."
        )
    elif kinds & version_kinds:
        note = (
            "Publication-version links were found; UpdateIn identifies a later "
            "version and UpdateOf an earlier version. These links alone do not "
            "establish a correction or retraction."
        )
        if status == "check_incomplete":
            note += " One or more applicable formal-status records were unavailable."
    elif status == "no_retraction_signal":
        note = (
            "No formal update signal was present in the checked PubMed or "
            "Crossref record; absence is not proof that no notice exists."
        )
    else:
        note = (
            "One or more applicable PubMed or Crossref records were unavailable "
            "for a complete formal-status check."
        )
    return RetractionAssessment(status=status, evidence=signals, note=note)


@dataclass(frozen=True, slots=True)
class SciteTallies:
    total: int | None = None
    supporting: int | None = None
    contradicting: int | None = None
    mentioning: int | None = None
    unclassified: int | None = None
    citing_publications: int | None = None


@dataclass(frozen=True, slots=True)
class PubMedCommentLink:
    relation: str
    pmid: str

    def report_text(self) -> str:
        return f"{self.relation}:PMID:{self.pmid}"


@dataclass(frozen=True, slots=True)
class CommentaryAssessment:
    status: str
    pubmed_comment_links: tuple[PubMedCommentLink, ...] = ()
    scite: SciteTallies = SciteTallies()
    note: str = ""


def pubmed_comment_links(xml_text: str | None) -> tuple[PubMedCommentLink, ...]:
    """Return PubMed records linked as comments on or by the current record."""
    if not xml_text:
        return ()
    try:
        root = ET.fromstring(xml_text)
    except (ET.ParseError, ValueError):
        return ()
    values: list[PubMedCommentLink] = []
    for node in root.findall(".//CommentsCorrectionsList/CommentsCorrections"):
        relation = _normalized_token(node.get("RefType"))
        if relation not in {"commentin", "commenton"}:
            continue
        pmid = node.find("./PMID")
        value = (pmid.text or "").strip() if pmid is not None else ""
        link = PubMedCommentLink(
            "CommentIn" if relation == "commentin" else "CommentOn",
            value,
        )
        if value.isdigit() and link not in values:
            values.append(link)
    return tuple(values)


def comment_links_from_report(value: str) -> tuple[PubMedCommentLink, ...]:
    """Parse the stable compact-report representation of PubMed comments."""
    links: list[PubMedCommentLink] = []
    for item in value.split(","):
        match = re.fullmatch(r"(CommentIn|CommentOn):PMID:(\d+)", item.strip())
        if match:
            link = PubMedCommentLink(match.group(1), match.group(2))
            if link not in links:
                links.append(link)
    return tuple(links)


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def parse_scite_tallies(payload: dict[str, Any] | None) -> SciteTallies:
    if not isinstance(payload, dict):
        return SciteTallies()
    return SciteTallies(
        total=_nonnegative_int(payload.get("total")),
        supporting=_nonnegative_int(payload.get("supporting")),
        contradicting=_nonnegative_int(payload.get("contradicting")),
        mentioning=_nonnegative_int(payload.get("mentioning")),
        unclassified=_nonnegative_int(payload.get("unclassified")),
        citing_publications=_nonnegative_int(payload.get("citingPublications")),
    )


def assess_commentary(
    *,
    pubmed_xml: str | None = None,
    scite_payload: dict[str, Any] | None = None,
    pubmed_checked: bool = False,
    scite_checked: bool = False,
    pubmed_expected: bool = False,
    scite_expected: bool = False,
    known_comment_links: Iterable[PubMedCommentLink] = (),
) -> CommentaryAssessment:
    """Classify linked comments and citation tallies without judging validity."""
    comment_links = tuple(
        dict.fromkeys(
            (*pubmed_comment_links(pubmed_xml), *known_comment_links)
        )
    )
    scite = parse_scite_tallies(scite_payload)

    if (scite.contradicting or 0) > 0:
        status = "critical_citation_found"
    elif comment_links:
        status = "linked_comment_found"
    elif (scite.total or 0) > 0:
        status = "citation_context_found"
    elif (pubmed_expected and not pubmed_checked) or (
        scite_expected and not scite_checked
    ):
        status = "check_incomplete"
    elif pubmed_checked or scite_checked:
        status = "no_commentary_signal"
    else:
        status = "check_incomplete"

    details: list[str] = []
    if comment_links:
        details.append(
            "PubMed comment links: "
            + ",".join(link.report_text() for link in comment_links)
        )
    if scite_checked:
        details.append(
            "Scite citation statements: "
            f"total={scite.total if scite.total is not None else 'unknown'},"
            f"supporting={scite.supporting if scite.supporting is not None else 'unknown'},"
            "contradicting="
            f"{scite.contradicting if scite.contradicting is not None else 'unknown'},"
            f"mentioning={scite.mentioning if scite.mentioning is not None else 'unknown'}"
        )
    elif scite_expected:
        details.append("Scite tally was unavailable")
    if pubmed_expected and not pubmed_checked:
        details.append("PubMed record was unavailable")
    details.append(
        "Comment and citation signals are not retraction or misconduct findings"
    )
    return CommentaryAssessment(
        status=status,
        pubmed_comment_links=comment_links,
        scite=scite,
        note="; ".join(details),
    )
