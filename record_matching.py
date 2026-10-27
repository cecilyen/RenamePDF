"""Conservative bibliographic record matching decisions.

Provider transport and parsing stay outside this module. The interface accepts
normalized records and owns completeness, agreement, conflict, ranking,
ambiguity, filename corroboration, confidence, and acceptance. Preliminary-to-
published decisions keep their own explicit thresholds and result type.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pdf_evidence


TITLE_ACCEPT_SCORE = 0.86
TITLE_AMBIGUITY_MARGIN = 0.04
VERY_HIGH_TITLE_SIMILARITY = 0.96
LOW_SOURCE_TITLE_CONFIRM_SCORE = 0.72
MATERIAL_TITLE_CONFLICT_SCORE = 0.72
IDENTIFIER_TITLE_CONTRADICTION_SCORE = 0.25
# Exact identifiers tolerate noisy extraction. PII additionally requires the
# sequence/token corroboration in decide_pii_match; provider agreement is separate.
PROVIDER_TRUST = {
    "pubmed": 3,
    "crossref": 2,
    "datacite": 1,
}


@dataclass(slots=True)
class Author:
    given: str
    family: str


@dataclass(slots=True)
class PaperMeta:
    title: str
    year: int
    journal_abbrev: str | None
    first_author: Author
    authors: list[Author]
    corresponding_author: Author | None
    doi: str | None
    pmid: str | None
    author_keywords: list[str]
    mesh_keywords: list[str]


@dataclass(frozen=True, slots=True)
class ProviderRecord:
    provider: str
    meta: PaperMeta
    pmid: str | None = None


@dataclass(slots=True)
class ResolvedRecord:
    status: str
    meta: PaperMeta
    pmid: str | None
    confidence: float
    source: str
    reason: str
    provider: str = ""


@dataclass(frozen=True, slots=True)
class MatchDecision:
    outcome: str
    resolved: ResolvedRecord | None
    reason: str


@dataclass(frozen=True, slots=True)
class _ScoredRecord:
    record: ProviderRecord
    confidence: float
    reason: str
    agreement: int
    completeness: int


@dataclass(frozen=True, slots=True)
class _TitleGroup:
    records: tuple[ProviderRecord, ...]
    representative: ProviderRecord
    similarity: float
    decision_score: float
    completeness: int


def _strip_diacritics(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value or "")
        if not unicodedata.combining(character)
    )


def _name_token(value: str) -> str:
    return "".join(
        part
        for part in re.sub(
            r"[^A-Za-z0-9]+",
            " ",
            _strip_diacritics(value),
        ).lower().split()
        if part
    )


def normalize_title(title: str) -> str:
    title = _strip_diacritics(
        pdf_evidence.clean_title_candidate(title)
    ).lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def title_tokens(title: str) -> set[str]:
    return {
        token
        for token in normalize_title(title).split()
        if len(token) > 2
    }


def _usable_title_hint(title: str | None) -> str | None:
    cleaned = pdf_evidence.clean_title_candidate(title or "")
    return cleaned if pdf_evidence.looks_like_title(cleaned) else None


def _title_part_numbers(title: str) -> frozenset[int]:
    return frozenset(
        int(value)
        for value in re.findall(
            r"\b(?:part|pt)\s*([1-9]\d*)\b",
            title,
            re.IGNORECASE,
        )
    )


def title_similarity(left: str | None, right: str | None) -> float:
    if not left or not right:
        return 0.0
    left_normalized = normalize_title(left)
    right_normalized = normalize_title(right)
    if not left_normalized or not right_normalized:
        return 0.0
    sequence = difflib.SequenceMatcher(
        None,
        left_normalized,
        right_normalized,
    ).ratio()
    left_tokens = title_tokens(left)
    right_tokens = title_tokens(right)
    if not left_tokens or not right_tokens:
        return sequence
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = overlap / union if union else 0.0
    containment = overlap / min(len(left_tokens), len(right_tokens))
    return max(
        sequence * 0.55 + jaccard * 0.45,
        containment * 0.92,
    )


def published_pdf_doi_evidence(
    expected_doi: str,
    observed_dois: list[str],
    extracted_texts: list[str],
) -> tuple[bool, bool]:
    """Check exact DOI identity and eLife's distinct publication versions."""
    expected = pdf_evidence.clean_doi(expected_doi)
    observed = [pdf_evidence.clean_doi(doi) for doi in observed_dois]
    if any(pdf_evidence.dois_equal(doi, expected) for doi in observed):
        return True, False
    elife = re.fullmatch(
        r"(10\.7554/elife\.\d+)(?:\.(\d+))?", expected, re.IGNORECASE,
    )
    if not elife:
        return False, False
    base, expected_version = elife.groups()
    family_versions: set[str] = set()
    has_canonical = False
    for doi in observed:
        family = re.fullmatch(rf"{re.escape(base)}(?:\.(\d+))?", doi, re.IGNORECASE)
        if family:
            if family.group(1):
                family_versions.add(family.group(1))
            else:
                has_canonical = True
    if expected_version:
        is_version_of_record = bool(re.search(
            r"\bVersion\s+of\s+Record\s+published\b",
            "\n".join(extracted_texts), re.IGNORECASE,
        ))
        if has_canonical and is_version_of_record:
            return True, False
    return False, bool(family_versions)


def replacement_pdf_rejection(
    evidence: pdf_evidence.PdfEvidence,
    expected_doi: str,
    expected_title: str,
    *,
    require_text: bool = False,
) -> str | None:
    """Accept replacement identity from PDF evidence, never its generated name."""
    if evidence.needs_rewrite or evidence.pdf_status not in {"ok", "pdf_image_only"}:
        return f"PDF condition requires normalization or review: {evidence.pdf_status}"
    if require_text and evidence.pdf_status == "pdf_image_only":
        return "An image-only PDF cannot supply the requested clean text copy"
    primary = next((item for item in evidence.dois if item.source != "filename"), None)
    texts = list(evidence.text_samples)
    doi_match, version_conflict = published_pdf_doi_evidence(
        expected_doi, [primary.doi] if primary else [], texts,
    )
    if version_conflict:
        return "PDF is a different publication version of the matched record"
    if primary and primary.score >= 80 and not doi_match:
        return "PDF primary DOI conflicts with the matched record"
    title = _usable_title_hint(evidence.title)
    similarity = title_similarity(title, expected_title)
    if title and similarity < IDENTIFIER_TITLE_CONTRADICTION_SCORE:
        return f"PDF title contradicts the matched record (similarity {similarity:.2f})"
    if doi_match and primary is not None and primary.score >= 80:
        return None
    if title and similarity >= 0.85:
        return None
    return "PDF DOI/title does not match the published record with sufficient confidence"


def _title_match_components(
    left: str | None,
    right: str | None,
) -> tuple[float, float, float]:
    if not left or not right:
        return 0.0, 0.0, 0.0
    left_normalized = normalize_title(left)
    right_normalized = normalize_title(right)
    left_tokens = title_tokens(left)
    right_tokens = title_tokens(right)
    if (
        not left_normalized
        or not right_normalized
        or not left_tokens
        or not right_tokens
    ):
        return 0.0, 0.0, 0.0
    overlap = len(left_tokens & right_tokens)
    sequence = difflib.SequenceMatcher(
        None,
        left_normalized,
        right_normalized,
    ).ratio()
    jaccard = overlap / len(left_tokens | right_tokens)
    containment = overlap / min(len(left_tokens), len(right_tokens))
    return sequence, jaccard, containment


def record_has_enough_metadata(meta: PaperMeta | None) -> bool:
    title = pdf_evidence.clean_title_candidate(meta.title if meta else "")
    family = (
        _name_token(meta.first_author.family)
        if meta and meta.first_author
        else ""
    )
    return bool(
        meta
        and len(title) >= 8
        and len(re.findall(r"[A-Za-z]", title)) >= 5
        and meta.year >= 1800
        and meta.first_author
        and family
        and family != "unknown"
    )


def _record_completeness(meta: PaperMeta) -> int:
    score = 0
    score += min(len(title_tokens(meta.title)), 10)
    score += 4 if meta.doi else 0
    score += 3 if meta.journal_abbrev else 0
    score += 2 if len(meta.authors) > 1 else 0
    score += 1 if meta.corresponding_author else 0
    score += min(
        len(meta.author_keywords) + len(meta.mesh_keywords),
        3,
    )
    return score


def _provider_trust(provider: str) -> int:
    return PROVIDER_TRUST.get(provider, 0)


def _provider_status(provider: str, *, title_lookup: bool) -> str:
    suffix = "_title_resolved" if title_lookup else "_resolved"
    return f"{provider}{suffix}"


def _material_conflicts(
    left: ProviderRecord,
    right: ProviderRecord,
) -> list[str]:
    left_meta = left.meta
    right_meta = right.meta
    conflicts: list[str] = []
    if (
        left_meta.doi
        and right_meta.doi
        and not pdf_evidence.dois_equal(left_meta.doi, right_meta.doi)
    ):
        conflicts.append(
            f"DOI {left_meta.doi} versus {right_meta.doi}"
        )
    similarity = title_similarity(left_meta.title, right_meta.title)
    if similarity < MATERIAL_TITLE_CONFLICT_SCORE:
        conflicts.append(f"title similarity {similarity:.2f}")
    if (
        left_meta.year
        and right_meta.year
        and abs(left_meta.year - right_meta.year) > 1
    ):
        conflicts.append(
            f"year {left_meta.year} versus {right_meta.year}"
        )
    left_family = _name_token(left_meta.first_author.family)
    right_family = _name_token(right_meta.first_author.family)
    if left_family and right_family and left_family != right_family:
        conflicts.append(
            f"first author {left_meta.first_author.family} versus "
            f"{right_meta.first_author.family}"
        )
    return conflicts


def _provider_conflict(
    records: list[ProviderRecord],
    *,
    same_doi_only: bool,
    title_hint: str | None = None,
) -> str | None:
    for index, left in enumerate(records):
        for right in records[index + 1:]:
            if left.provider == right.provider:
                continue
            if same_doi_only:
                if (
                    not left.meta.doi
                    or not right.meta.doi
                    or not pdf_evidence.dois_equal(
                        left.meta.doi,
                        right.meta.doi,
                    )
                ):
                    continue
                if title_hint and max(
                    title_similarity(title_hint, left.meta.title),
                    title_similarity(title_hint, right.meta.title),
                ) < TITLE_ACCEPT_SCORE:
                    continue
            conflicts = _material_conflicts(left, right)
            if conflicts:
                return (
                    f"{left.provider} and {right.provider} returned materially "
                    f"conflicting records: {', '.join(conflicts)}"
                )
    return None


def _notice_title_kind(title: str | None) -> str:
    match = re.match(
        r"^\s*(?:publisher(?:'s)?\s+|author\s+)?"
        r"(retraction(?:\s+note)?|correction|corrigendum|erratum|expression\s+of\s+concern)\b",
        title or "", re.IGNORECASE,
    )
    if not match:
        return ""
    value = match.group(1).casefold()
    if value.startswith("retraction"):
        return "retraction"
    if value in {"correction", "corrigendum", "erratum"}:
        return "correction"
    return "expression_of_concern"


def _notice_titles_conflict(left: str | None, right: str) -> bool:
    return bool(left and _notice_title_kind(left) != _notice_title_kind(right))


def _evaluate_doi_record(
    record: ProviderRecord,
    candidate: pdf_evidence.DoiCandidate,
    title_hint: str | None,
) -> tuple[bool, float, str]:
    meta = record.meta
    usable_title = _usable_title_hint(title_hint)
    if _notice_titles_conflict(usable_title, meta.title):
        return False, 0.0, "Formal-notice title does not describe the same document"
    similarity = (
        title_similarity(usable_title, meta.title)
        if usable_title
        else 0.0
    )
    if meta.doi and pdf_evidence.dois_equal(meta.doi, candidate.doi):
        if (
            usable_title
            and similarity < IDENTIFIER_TITLE_CONTRADICTION_SCORE
        ):
            return (
                False,
                similarity,
                "DOI matched, but PDF title similarity was only "
                f"{similarity:.2f}",
            )
        if candidate.score >= 80:
            confidence = (
                0.99
                if not similarity
                else min(0.99, 0.88 + similarity * 0.11)
            )
            return (
                True,
                confidence,
                f"DOI matched from {candidate.source}",
            )
        if similarity >= LOW_SOURCE_TITLE_CONFIRM_SCORE:
            return (
                True,
                min(0.98, 0.80 + similarity * 0.18),
                "Low-priority DOI source confirmed by title similarity "
                f"{similarity:.2f}",
            )
        return (
            False,
            similarity,
            f"DOI found only in {candidate.source}; title confirmation "
            "was too weak",
        )
    if similarity >= TITLE_ACCEPT_SCORE:
        sequence, jaccard, _ = _title_match_components(usable_title, meta.title)
        if sequence < 0.72 and jaccard < 0.65:
            return False, similarity, "DOI differed or was absent; title overlap was only a short fragment"
        return (
            True,
            min(0.94, similarity),
            "Record DOI differed or was absent, but title similarity was "
            f"{similarity:.2f}",
        )
    return (
        False,
        similarity,
        "Returned record did not match the candidate DOI or PDF title",
    )


def decide_doi_match(
    candidate: pdf_evidence.DoiCandidate,
    title_hint: str | None,
    records: list[ProviderRecord],
) -> MatchDecision:
    complete = [
        record
        for record in records
        if record_has_enough_metadata(record.meta)
    ]
    if not complete:
        return MatchDecision(
            "rejected",
            None,
            "DOI lookup returned no complete provider records",
        )

    claiming_records = [
        record
        for record in complete
        if record.meta.doi
        and pdf_evidence.dois_equal(record.meta.doi, candidate.doi)
    ]
    conflict = _provider_conflict(
        claiming_records,
        same_doi_only=False,
    )
    if conflict:
        return MatchDecision(
            "conflicting_provider_records",
            None,
            conflict,
        )

    scored: list[_ScoredRecord] = []
    rejections: list[str] = []
    for record in complete:
        accepted, confidence, reason = _evaluate_doi_record(
            record,
            candidate,
            title_hint,
        )
        if accepted:
            scored.append(
                _ScoredRecord(
                    record=record,
                    confidence=confidence,
                    reason=reason,
                    agreement=0,
                    completeness=_record_completeness(record.meta),
                )
            )
        else:
            rejections.append(f"{record.provider}: {reason}")

    if not scored:
        return MatchDecision(
            "rejected",
            None,
            "; ".join(rejections[:3])
            or "No provider record passed DOI confidence checks",
        )

    accepted_records = [item.record for item in scored]
    conflict = _provider_conflict(
        accepted_records,
        same_doi_only=False,
    )
    if conflict:
        return MatchDecision(
            "conflicting_provider_records",
            None,
            conflict,
        )

    agreement = len({record.provider for record in accepted_records})
    scored = [
        _ScoredRecord(
            record=item.record,
            confidence=item.confidence,
            reason=item.reason,
            agreement=agreement,
            completeness=item.completeness,
        )
        for item in scored
    ]
    scored.sort(
        key=lambda item: (
            item.confidence,
            item.agreement,
            item.completeness,
            _provider_trust(item.record.provider),
        ),
        reverse=True,
    )
    winner = scored[0]
    reason = winner.reason
    if agreement > 1:
        reason += f"; {agreement} providers agreed"
    resolved = ResolvedRecord(
        status=_provider_status(
            winner.record.provider,
            title_lookup=False,
        ),
        meta=winner.record.meta,
        pmid=winner.record.pmid,
        confidence=winner.confidence,
        source=candidate.source,
        reason=reason,
        provider=winner.record.provider,
    )
    return MatchDecision("accepted", resolved, reason)


def _same_work(
    left: ProviderRecord,
    right: ProviderRecord,
) -> bool:
    if left.meta.doi and right.meta.doi:
        return pdf_evidence.dois_equal(left.meta.doi, right.meta.doi)
    left_parts = _title_part_numbers(left.meta.title)
    right_parts = _title_part_numbers(right.meta.title)
    if (left_parts or right_parts) and left_parts != right_parts:
        return False
    return bool(
        abs(left.meta.year - right.meta.year) <= 1
        and _name_token(left.meta.first_author.family)
        == _name_token(right.meta.first_author.family)
        and title_similarity(left.meta.title, right.meta.title) >= 0.94
    )


def decide_pii_match(
    pii: str,
    title_hint: str | None,
    records: list[ProviderRecord],
) -> MatchDecision:
    """Decide whether exact publisher-item-ID records identify the PDF."""
    identifier = str(pii or "").strip()
    if not identifier:
        return MatchDecision(
            "rejected",
            None,
            "No publisher item identifier was available",
        )

    complete = [
        record
        for record in records
        if record_has_enough_metadata(record.meta)
    ]
    if not complete:
        return MatchDecision(
            "rejected",
            None,
            "Publisher identifier lookup returned no complete records",
        )

    for index, left in enumerate(complete):
        for right in complete[index + 1:]:
            if not _same_work(left, right) or _material_conflicts(left, right):
                return MatchDecision(
                    "rejected",
                    None,
                    f"Publisher identifier {identifier} matched multiple "
                    "publications or conflicting records",
                )

    usable_title = _usable_title_hint(title_hint)
    ranked: list[tuple[float, int, int, ProviderRecord]] = []
    rejection_reasons: list[str] = []
    for record in complete:
        similarity = (
            title_similarity(usable_title, record.meta.title)
            if usable_title
            else 0.0
        )
        if usable_title:
            sequence, jaccard, _containment = _title_match_components(
                usable_title,
                record.meta.title,
            )
            if similarity < IDENTIFIER_TITLE_CONTRADICTION_SCORE:
                rejection_reasons.append(
                    f"{record.provider}: exact publisher identifier conflicted "
                    f"with the PDF title (similarity {similarity:.2f})"
                )
                continue
            if sequence < 0.72 and jaccard < 0.65:
                rejection_reasons.append(
                    f"{record.provider}: title overlap was only a short "
                    f"fragment (sequence {sequence:.2f}, token overlap "
                    f"{jaccard:.2f})"
                )
                continue
        ranked.append(
            (
                similarity,
                _record_completeness(record.meta),
                _provider_trust(record.provider),
                record,
            )
        )

    if not ranked:
        return MatchDecision(
            "rejected",
            None,
            "; ".join(rejection_reasons[:3])
            or "Publisher identifier records failed title corroboration",
        )

    ranked.sort(key=lambda item: item[:3], reverse=True)
    similarity, _completeness, _trust, winner = ranked[0]
    reason = f"Publisher identifier {identifier} exactly matched"
    confidence = 0.97
    if usable_title:
        reason += f" and title similarity was {similarity:.2f}"
        confidence = min(0.98, 0.84 + similarity * 0.14)
    resolved = ResolvedRecord(
        status=_provider_status(winner.provider, title_lookup=False),
        meta=winner.meta,
        pmid=winner.pmid,
        confidence=confidence,
        source="filename-pii",
        reason=reason,
        provider=winner.provider,
    )
    return MatchDecision("accepted", resolved, reason)


def _group_title_records(
    records: list[ProviderRecord],
    title_hint: str,
) -> list[_TitleGroup]:
    groups: list[list[ProviderRecord]] = []
    for record in records:
        for group in groups:
            if any(_same_work(record, member) for member in group):
                group.append(record)
                break
        else:
            groups.append([record])

    results: list[_TitleGroup] = []
    for group in groups:
        representative = max(
            group,
            key=lambda record: (
                title_similarity(title_hint, record.meta.title),
                _record_completeness(record.meta),
                _provider_trust(record.provider),
            ),
        )
        similarity = title_similarity(
            title_hint,
            representative.meta.title,
        )
        provider_count = len({record.provider for record in group})
        decision_score = min(
            1.0,
            similarity + min(provider_count - 1, 2) * 0.01,
        )
        results.append(
            _TitleGroup(
                records=tuple(group),
                representative=representative,
                similarity=similarity,
                decision_score=decision_score,
                completeness=_record_completeness(representative.meta),
            )
        )
    return results


def _bibliographic_identity_agrees(
    left: PaperMeta,
    right: PaperMeta,
) -> bool:
    if abs(left.year - right.year) > 1:
        return False
    if title_similarity(left.title, right.title) < 0.99:
        return False
    if _title_part_numbers(left.title) != _title_part_numbers(right.title):
        return False
    if _name_token(left.first_author.family) != _name_token(
        right.first_author.family
    ):
        return False
    left_journal = _name_token(left.journal_abbrev or "")
    right_journal = _name_token(right.journal_abbrev or "")
    if left_journal and right_journal and left_journal != right_journal:
        return False
    left_authors = tuple(
        _name_token(author.family)
        for author in left.authors
        if _name_token(author.family)
    )
    right_authors = tuple(
        _name_token(author.family)
        for author in right.authors
        if _name_token(author.family)
    )
    return not (
        left_authors
        and right_authors
        and left_authors != right_authors
    )


def _consensus_rejects_duplicate_doi_outlier(
    best: _TitleGroup,
    outlier: _TitleGroup,
) -> bool:
    best_providers = {record.provider for record in best.records}
    outlier_providers = {record.provider for record in outlier.records}
    best_doi = best.representative.meta.doi
    outlier_doi = outlier.representative.meta.doi
    return bool(
        len(best_providers) >= 2
        and len(outlier_providers) == 1
        and outlier_providers <= best_providers
        and best_doi
        and outlier_doi
        and not pdf_evidence.dois_equal(best_doi, outlier_doi)
        and _bibliographic_identity_agrees(
            best.representative.meta,
            outlier.representative.meta,
        )
    )


def _filename_year_matches(meta: PaperMeta, filename_hint: str | None) -> bool:
    if not filename_hint or meta.year < 1800:
        return False
    stem = Path(filename_hint).stem
    four_digit_years = set(
        re.findall(r"(?<!\d)(?:18|19|20|21)\d{2}(?!\d)", stem)
    )
    if four_digit_years:
        return str(meta.year) in four_digit_years
    short_year = f"{meta.year % 100:02d}"
    return bool(re.search(rf"(?<!\d){short_year}(?!\d)", stem))


def _filename_title_matches(
    meta: PaperMeta,
    filename_hint: str | None,
) -> bool:
    if not filename_hint:
        return False
    record_tokens = title_tokens(meta.title)
    filename_tokens = title_tokens(Path(filename_hint).stem)
    overlap = len(record_tokens & filename_tokens)
    denominator = min(len(record_tokens), 6)
    return bool(
        denominator
        and overlap >= 2
        and overlap / denominator >= 0.40
    )


def filename_corroborates(
    meta: PaperMeta,
    filename_hint: str | None,
) -> bool:
    return _filename_year_matches(
        meta,
        filename_hint,
    ) and _filename_title_matches(meta, filename_hint)


def decide_title_match(
    title_hint: str,
    filename_hint: str | None,
    records: list[ProviderRecord],
) -> MatchDecision:
    complete = [
        record
        for record in records
        if record_has_enough_metadata(record.meta)
        and not _notice_titles_conflict(title_hint, record.meta.title)
    ]
    if not complete:
        return MatchDecision(
            "rejected",
            None,
            "Title search returned no complete provider records",
        )

    conflict = _provider_conflict(
        complete,
        same_doi_only=True,
        title_hint=title_hint,
    )
    if conflict:
        return MatchDecision(
            "conflicting_provider_records",
            None,
            conflict,
        )

    groups = _group_title_records(complete, title_hint)
    groups.sort(
        key=lambda group: (
            group.decision_score,
            len({record.provider for record in group.records}),
            group.completeness,
            _provider_trust(group.representative.provider),
        ),
        reverse=True,
    )
    if not groups:
        return MatchDecision(
            "rejected",
            None,
            "Title search returned no usable records",
        )

    best = groups[0]
    second = groups[1] if len(groups) > 1 else None
    meta = best.representative.meta
    if best.similarity < TITLE_ACCEPT_SCORE:
        return MatchDecision(
            "rejected",
            None,
            f"Best title similarity was only {best.similarity:.2f}",
        )

    hint_years = {
        int(value)
        for value in re.findall(r"\b(?:19|20)\d{2}\b", title_hint)
    }
    if (
        hint_years
        and min(abs(meta.year - year) for year in hint_years) > 2
    ):
        years = ", ".join(str(year) for year in sorted(hint_years))
        return MatchDecision(
            "rejected",
            None,
            f"Record year {meta.year} conflicted with PDF title year {years}",
        )

    sequence, jaccard, _ = _title_match_components(
        title_hint,
        meta.title,
    )
    if sequence < 0.72 and jaccard < 0.65:
        return MatchDecision(
            "rejected",
            None,
            "Title overlap was only a short fragment "
            f"(sequence {sequence:.2f}, token overlap {jaccard:.2f})",
        )

    second_score = second.decision_score if second else 0.0
    ambiguous_runners = [
        group
        for group in groups[1:]
        if (best.decision_score - group.decision_score)
        < TITLE_AMBIGUITY_MARGIN
    ]
    ambiguous = bool(ambiguous_runners)
    filename_tie_break = False
    consensus_tie_break = bool(
        len(ambiguous_runners) == 1
        and _consensus_rejects_duplicate_doi_outlier(
            best,
            ambiguous_runners[0],
        )
    )
    if ambiguous:
        if (
            best.similarity >= VERY_HIGH_TITLE_SIMILARITY
            and not consensus_tie_break
        ):
            return MatchDecision(
                "rejected",
                None,
                "Title search was ambiguous: best "
                f"{best.decision_score:.2f}, next {second_score:.2f}; "
                "near-identical records could not be distinguished safely",
            )
        if not consensus_tie_break:
            best_filename = filename_corroborates(
                meta,
                filename_hint,
            )
            second_filename = filename_corroborates(
                second.representative.meta,
                filename_hint,
            )
            if not best_filename or second_filename:
                return MatchDecision(
                    "rejected",
                    None,
                    "Title search was ambiguous: best "
                    f"{best.decision_score:.2f}, next {second_score:.2f}; "
                    "filename did not independently corroborate only one "
                    "record",
                )
            filename_tie_break = True

    providers = {
        record.provider
        for record in best.records
    }
    reason = (
        f"Title search accepted with similarity {best.similarity:.2f}"
    )
    if len(providers) > 1:
        reason += f"; {len(providers)} providers agreed"
    if consensus_tie_break:
        reason += (
            "; cross-provider consensus rejected a duplicate provider DOI "
            "outlier"
        )
    if filename_tie_break:
        reason += "; filename year and title corroboration broke the tie"

    provider = best.representative.provider
    resolved = ResolvedRecord(
        status=_provider_status(provider, title_lookup=True),
        meta=meta,
        pmid=best.representative.pmid,
        confidence=min(best.decision_score, 0.95),
        source=f"{provider}-title",
        reason=reason,
        provider=provider,
    )
    return MatchDecision("accepted", resolved, reason)


# Preliminary-to-published identity decisions. No download or placement policy.

REPLACEMENT_STRONG_TITLE_SCORE = 0.95
REPLACEMENT_TITLE_WITH_AUTHOR_SCORE = 0.84
REPLACEMENT_DIRECT_TITLE_SCORE = 0.90
REPLACEMENT_TITLE_AMBIGUITY_MARGIN = 0.035
REPLACEMENT_PLAUSIBLE_RUNNER_UP_SCORE = 0.80

ReplacementDecisionOutcome = Literal["accepted", "rejected", "ambiguous"]


@dataclass(frozen=True, slots=True)
class PreliminaryWork:
    kind: str
    title: str
    doi: str | None
    first_author: str | None
    year: int | None


@dataclass(frozen=True, slots=True)
class PublishedRecord:
    doi: str
    title: str
    year: int | None
    record_type: str
    journal: str
    authors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplacementDecision:
    outcome: ReplacementDecisionOutcome
    record: PublishedRecord | None
    confidence: float
    method: str
    reason: str
    title_similarity: float | None = None
    author_match: bool = False
    runner_up_margin: float | None = None

    @property
    def accepted(self) -> bool:
        return self.outcome == "accepted" and self.record is not None


@dataclass(frozen=True, slots=True)
class _ScoredReplacementRecord:
    record: PublishedRecord
    title_similarity: float
    author_match: bool
    score: float


def title_part_number(title: str) -> int | None:
    match = re.search(r"\bpart\s*([1-9]\d*)\b", title, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _replacement_author_matches(
    source: PreliminaryWork,
    record: PublishedRecord,
) -> bool:
    expected = (source.first_author or "").casefold()
    return bool(expected and expected in record.authors[:3])


def _replacement_record_rejection(
    source: PreliminaryWork,
    record: PublishedRecord,
    *,
    enforce_part: bool,
) -> str | None:
    doi = pdf_evidence.clean_doi(record.doi)
    if not doi:
        return "published record has no usable DOI"
    if source.doi and pdf_evidence.dois_equal(source.doi, doi):
        return "published record DOI is identical to the source DOI"
    if record.record_type != "journal-article":
        return f"record type is {record.record_type or 'missing'}, not journal-article"
    if not record.title:
        return "published record has no usable title"
    if not record.journal:
        return "published record has no journal venue"
    if record.year is None or record.year < 1900:
        return "published record has no usable publication year"
    if not record.authors:
        return "published record has no usable authors"
    if source.year and record.year < source.year - 1:
        return (
            f"published record year {record.year} predates source year "
            f"{source.year} by more than one year"
        )
    if enforce_part:
        source_part = title_part_number(source.title)
        published_part = title_part_number(record.title)
        if source_part is not None and published_part != source_part:
            return (
                f"published title does not preserve source part {source_part}"
            )
    return None


def reject_replacement(
    reason: str,
    *,
    method: str,
    outcome: ReplacementDecisionOutcome = "rejected",
) -> ReplacementDecision:
    return ReplacementDecision(
        outcome=outcome,
        record=None,
        confidence=0.0,
        method=method,
        reason=reason,
    )


def conflicting_publication_relations(dois: tuple[str, ...]) -> ReplacementDecision:
    rendered = ", ".join(sorted(dois, key=str.casefold))
    return reject_replacement(
        f"conflicting direct publication relations: {rendered}",
        method="direct_publication_relation",
        outcome="ambiguous",
    )


def decide_published_relation(
    source: PreliminaryWork,
    record: PublishedRecord,
    *,
    source_identity_verified: bool,
) -> ReplacementDecision:
    method = "direct_publication_relation"
    if not source_identity_verified:
        return reject_replacement(
            "direct relation was not tied to the source paper identity",
            method=method,
        )
    invalid = _replacement_record_rejection(source, record, enforce_part=False)
    if invalid:
        return reject_replacement(invalid, method=method)

    similarity = title_similarity(source.title, record.title)
    author_match = _replacement_author_matches(source, record)
    if not author_match and similarity < REPLACEMENT_DIRECT_TITLE_SCORE:
        return ReplacementDecision(
            outcome="rejected",
            record=None,
            confidence=0.0,
            method=method,
            reason=(
                "direct relation target lacks source-author corroboration and "
                f"title similarity is only {similarity:.3f}"
            ),
            title_similarity=similarity,
            author_match=False,
        )

    confidence = (
        0.99
        if author_match and similarity >= REPLACEMENT_DIRECT_TITLE_SCORE
        else 0.97
    )
    support = (
        "source identity, author, and title"
        if author_match and similarity >= REPLACEMENT_DIRECT_TITLE_SCORE
        else "verified source relation and author"
        if author_match
        else "verified source relation and strong title"
    )
    return ReplacementDecision(
        outcome="accepted",
        record=record,
        confidence=confidence,
        method=method,
        reason=f"accepted from {support} corroboration",
        title_similarity=similarity,
        author_match=author_match,
    )


def _deduplicate_published_records(
    records: tuple[PublishedRecord, ...],
) -> tuple[PublishedRecord, ...]:
    unique: dict[str, PublishedRecord] = {}
    for record in records:
        doi = pdf_evidence.clean_doi(record.doi).casefold()
        key = doi or f"missing:{len(unique)}"
        unique.setdefault(key, record)
    return tuple(unique.values())


def decide_published_title(
    source: PreliminaryWork,
    records: tuple[PublishedRecord, ...],
) -> ReplacementDecision:
    method = "crossref_journal_title"
    scored: list[_ScoredReplacementRecord] = []
    rejection_reasons: list[str] = []
    for record in _deduplicate_published_records(records):
        invalid = _replacement_record_rejection(source, record, enforce_part=True)
        if invalid:
            rejection_reasons.append(invalid)
            continue
        similarity = title_similarity(source.title, record.title)
        author_match = _replacement_author_matches(source, record)
        scored.append(
            _ScoredReplacementRecord(
                record=record,
                title_similarity=similarity,
                author_match=author_match,
                score=similarity + (0.03 if author_match else 0.0),
            )
        )

    if not scored:
        detail = rejection_reasons[0] if rejection_reasons else "search returned no records"
        return reject_replacement(
            f"no eligible journal record: {detail}",
            method=method,
        )

    scored.sort(
        key=lambda item: (
            item.score,
            item.title_similarity,
            item.record.doi.casefold(),
        ),
        reverse=True,
    )
    best = scored[0]
    passes_identity = (
        best.title_similarity >= REPLACEMENT_STRONG_TITLE_SCORE
        or (
            best.title_similarity >= REPLACEMENT_TITLE_WITH_AUTHOR_SCORE
            and best.author_match
        )
    )
    if not passes_identity:
        return ReplacementDecision(
            outcome="rejected",
            record=None,
            confidence=0.0,
            method=method,
            reason=(
                "best journal record did not meet identity threshold: "
                f"title={best.title_similarity:.3f}, "
                f"author_match={best.author_match}"
            ),
            title_similarity=best.title_similarity,
            author_match=best.author_match,
        )

    runner_up = next(
        (
            item
            for item in scored[1:]
            if item.title_similarity >= REPLACEMENT_PLAUSIBLE_RUNNER_UP_SCORE
        ),
        None,
    )
    margin = best.score - runner_up.score if runner_up else None
    if margin is not None and margin < REPLACEMENT_TITLE_AMBIGUITY_MARGIN:
        return ReplacementDecision(
            outcome="ambiguous",
            record=None,
            confidence=0.0,
            method=method,
            reason=(
                "top journal records are ambiguous: "
                f"score margin {margin:.3f} is below "
                f"{REPLACEMENT_TITLE_AMBIGUITY_MARGIN:.3f}"
            ),
            title_similarity=best.title_similarity,
            author_match=best.author_match,
            runner_up_margin=margin,
        )

    return ReplacementDecision(
        outcome="accepted",
        record=best.record,
        confidence=min(0.97, best.score),
        method=method,
        reason=(
            "accepted from strong title identity"
            + (" with author corroboration" if best.author_match else "")
        ),
        title_similarity=best.title_similarity,
        author_match=best.author_match,
        runner_up_margin=margin,
    )
