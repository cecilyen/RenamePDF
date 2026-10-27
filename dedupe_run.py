"""Detect duplicate scientific PDFs by embedded DOI or extracted title.

PDF inspection and DOI normalization come from the shared evidence module;
title matching comes from the shared conservative record-matching module.
Dry-run is the default. With --commit, duplicate copies are moved (never
deleted) to a Duplicated directory and the newest copy remains canonical.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import collection_mutation
import pdf_identity_cache
import filename_policy
import pdf_evidence
import pdf_process
import pdf_rewrite
import record_matching
import report_io

DEFAULT_REPORT_NAME = "0_dedupe_results.csv"
DEFAULT_TITLE_THRESHOLD = record_matching.TITLE_ACCEPT_SCORE
EXCLUDED_DIRECTORIES = {"duplicated", "failed", "superseded"}
PREPROCESS_NEEDED_PDF_STATUSES = frozenset({
    "pdf_cover_page_removal_needed",
    "pdf_decryption_needed",
    "pdf_repair_needed",
})
FAST_PREFIX_LENGTH = 7
FAST_SKIPPED_PDF_STATUS = "fast_skipped"
FAST_SKIPPED_RESULT_STATUS = "skipped_fast_prefix"
DEFAULT_INSPECTION_TIMEOUT = 120.0
IDENTITY_EXTRACTOR_VERSION = pdf_evidence.EVIDENCE_CACHE_VERSION
NON_CACHEABLE_PDF_STATUSES = pdf_rewrite.REWRITE_BLOCKING_STATUSES | {
    "pdf_processing_error",
    "pdf_processing_timeout",
}
TITLE_ONLY_MIN_TOKEN_JACCARD = 0.75
TITLE_ONLY_MIN_FILENAME_JACCARD = 0.75
DOI_TITLE_CONFLICT_SCORE = 0.60
TITLE_DUPLICATE_EVIDENCE_CACHE_SIZE = 8_192


@dataclass(frozen=True, slots=True)
class PdfIdentity:
    path: Path
    doi: str | None
    doi_source: str
    doi_score: int
    title: str | None
    pdf_status: str = "ok"
    note: str = ""


@dataclass(frozen=True, slots=True)
class _TitleDuplicateEvidence:
    normalized_title: str
    title_tokens: frozenset[str]
    filename_tokens: frozenset[str]
    first_author: str | None
    year: int | None
    part_numbers: frozenset[int]
    boilerplate: bool


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.members = [{index} for index in range(size)]

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if len(self.members[left_root]) < len(self.members[right_root]):
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.members[left_root].update(self.members[right_root])
        self.members[right_root].clear()
        return left_root


def discover_pdfs(
    root: Path,
    *,
    excluded_directories: Iterable[str] = (),
) -> list[Path]:
    """Return PDFs below root, excluding holding and caller-named directories."""
    excluded = EXCLUDED_DIRECTORIES | {
        name.casefold()
        for name in excluded_directories
        if name
    }
    return [
        path
        for path in sorted(root.rglob("*"), key=lambda item: str(item).lower())
        if path.is_file()
        and path.suffix.lower() == ".pdf"
        and not pdf_rewrite.is_rewrite_artifact(path)
        and not collection_mutation.is_replacement_artifact(path)
        and not (
            {
                part.casefold()
                for part in path.relative_to(root).parts[:-1]
            }
            & excluded
        )
    ]


def fast_prefix_plan(
    paths: Iterable[Path],
) -> tuple[list[Path], dict[Path, str]]:
    """Return same-prefix candidates and the paths skipped by ``--fast``.

    The prefix is the first seven characters of the filename stem, compared
    case-insensitively. A skipped PDF has no possible peer under this partial
    scan, so its costly DOI/title extraction can be avoided.
    """
    ordered_paths = list(paths)
    prefix_by_path = {
        path: path.stem[:FAST_PREFIX_LENGTH].casefold()
        for path in ordered_paths
    }
    counts: dict[str, int] = {}
    for prefix in prefix_by_path.values():
        counts[prefix] = counts.get(prefix, 0) + 1
    candidates = [
        path
        for path in ordered_paths
        if counts[prefix_by_path[path]] > 1
    ]
    skipped = {
        path: prefix_by_path[path]
        for path in ordered_paths
        if counts[prefix_by_path[path]] == 1
    }
    return candidates, skipped


def _fast_skipped_identity(path: Path, prefix: str) -> PdfIdentity:
    return PdfIdentity(
        path=path,
        doi=None,
        doi_source="",
        doi_score=0,
        title=None,
        pdf_status=FAST_SKIPPED_PDF_STATUS,
        note=(
            "Skipped DOI/title inspection in --fast mode: no other PDF "
            f"shares first {FAST_PREFIX_LENGTH} filename characters "
            f"({prefix!r})"
        ),
    )


def inspect_pdf(
    path: Path,
    pdf_password: str = "",
    *,
    commit: bool = False,
) -> PdfIdentity:
    """Extract the primary DOI and best title with shared evidence ranking."""
    try:
        hints = pdf_evidence.inspect_pdf_with_rewrite(
            path,
            commit=commit,
            pdf_password=pdf_password,
        )
    except Exception as exc:
        return PdfIdentity(
            path=path,
            doi=None,
            doi_source="",
            doi_score=0,
            title=None,
            pdf_status="pdf_processing_error",
            note=f"{type(exc).__name__}: {exc}",
        )

    return _identity_from_evidence(path, hints)


def _identity_from_evidence(
    path: Path,
    hints: pdf_evidence.PdfEvidence,
) -> PdfIdentity:
    """Convert shared PDF evidence into DedupePDF's compact identity."""

    primary = next(
        (
            candidate
            for candidate in hints.dois
            if candidate.source != "filename"
        ),
        None,
    )
    if primary is not None:
        doi = pdf_evidence.clean_doi(primary.doi).lower() or None
        doi_source = primary.source
        doi_score = primary.score
    else:
        doi = None
        doi_source = ""
        doi_score = 0

    return PdfIdentity(
        path=path,
        doi=doi,
        doi_source=doi_source,
        doi_score=doi_score,
        title=hints.title,
        pdf_status=hints.pdf_status,
        note=hints.pdf_note,
    )


def _processing_error(path: Path, status: str, note: str) -> PdfIdentity:
    return PdfIdentity(
        path=path,
        doi=None,
        doi_source="",
        doi_score=0,
        title=None,
        pdf_status=status,
        note=note,
    )


def _store_identity_evidence(
    cache: pdf_identity_cache.IdentityCache | None,
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
        cache.disable(
            "Disabled after cache write failed: "
            f"{type(exc).__name__}: {exc}"
        )


def inspect_pdfs(
    paths: Iterable[Path],
    *,
    pdf_password: str = "",
    workers: int = 1,
    commit: bool = False,
    inspection_timeout: float = DEFAULT_INSPECTION_TIMEOUT,
    inspector: Callable[..., PdfIdentity] = inspect_pdf,
    progress: Callable[[int, int, PdfIdentity], None] | None = None,
    cache: pdf_identity_cache.IdentityCache | None = None,
) -> list[PdfIdentity]:
    """Inspect PDFs with a reusable spawned pool and parent-owned cache."""
    ordered_paths = list(paths)
    if commit and workers > 1:
        raise ValueError(
            "commit inspection must be serial or process-isolated; "
            "PyMuPDF is not thread-safe"
        )
    if workers > 1 and inspector is not inspect_pdf:
        raise ValueError(
            "parallel inspection requires the process-isolated inspector"
        )

    if commit or inspector is not inspect_pdf:
        identities = []
        for completed, path in enumerate(ordered_paths, start=1):
            if inspector is inspect_pdf:
                identity = inspector(path, pdf_password, commit=commit)
            else:
                identity = inspector(path, pdf_password)
            identities.append(identity)
            if progress:
                progress(completed, len(ordered_paths), identity)
        return identities

    identities: dict[Path, PdfIdentity] = {}
    misses: list[Path] = []
    active_cache = cache if cache is not None and not pdf_password else None
    completed_before_workers = 0
    worker_completed = 0

    def disable_active_cache(action: str, exc: Exception) -> None:
        nonlocal active_cache
        if active_cache is not None:
            active_cache.disable(
                f"Disabled after {action} failed: {type(exc).__name__}: {exc}"
            )
            active_cache = None

    for path in ordered_paths:
        try:
            payload = (
                active_cache.get(path)
                if active_cache is not None
                else None
            )
        except Exception as exc:
            disable_active_cache("cache read", exc)
            payload = None
        if payload is None:
            misses.append(path)
            continue
        try:
            evidence = pdf_evidence.evidence_from_cache_payload(payload)
            if evidence.pdf_status in NON_CACHEABLE_PDF_STATUSES:
                raise ValueError(
                    "failed or blocking PDF evidence must not be cached"
                )
            identity = _identity_from_evidence(path, evidence)
        except (KeyError, TypeError, ValueError):
            try:
                active_cache.discard(path)
            except Exception as exc:
                disable_active_cache("invalid-entry removal", exc)
            misses.append(path)
            continue
        if _is_pdf_error(identity):
            try:
                active_cache.discard(path)
            except Exception as exc:
                disable_active_cache("error-entry removal", exc)
            misses.append(path)
            continue
        identities[path] = identity
        completed_before_workers += 1
        if progress:
            progress(completed_before_workers, len(ordered_paths), identity)

    def accept_worker_outcome(
        completed: int,
        _total: int,
        outcome: pdf_process.PdfBatchInspectionOutcome,
    ) -> None:
        nonlocal active_cache, worker_completed
        worker_completed = completed
        if outcome.evidence is not None:
            identity = _identity_from_evidence(outcome.path, outcome.evidence)
        else:
            identity = _processing_error(
                outcome.path,
                outcome.error_status or "pdf_processing_error",
                outcome.error_note or "Inspection worker returned no evidence",
            )
        identities[outcome.path] = identity
        if active_cache is not None:
            if outcome.evidence is None:
                try:
                    active_cache.discard(outcome.path)
                except Exception as exc:
                    disable_active_cache("error-entry removal", exc)
            else:
                _store_identity_evidence(
                    active_cache,
                    outcome.path,
                    outcome.evidence,
                )
                if not active_cache.usable:
                    active_cache = None
        if progress:
            progress(
                completed_before_workers + completed,
                len(ordered_paths),
                identity,
            )

    if misses:
        try:
            pdf_process.inspect_pdf_batch(
                misses,
                include_text_samples=False,
                pdf_password=pdf_password,
                max_workers=workers,
                timeout=inspection_timeout,
                progress=accept_worker_outcome,
            )
        except Exception as exc:
            remaining = [path for path in misses if path not in identities]
            for offset, path in enumerate(remaining, start=1):
                identity = _processing_error(
                    path,
                    "pdf_processing_error",
                    "Inspection pool failed: "
                    f"{type(exc).__name__}: {exc}",
                )
                identities[path] = identity
                if active_cache is not None:
                    try:
                        active_cache.discard(path)
                    except Exception as cache_exc:
                        disable_active_cache(
                            "error-entry removal",
                            cache_exc,
                        )
                if progress:
                    progress(
                        completed_before_workers + worker_completed + offset,
                        len(ordered_paths),
                        identity,
                    )
    return [identities[path] for path in ordered_paths]


def preprocess_identities(
    identities: Iterable[PdfIdentity],
    *,
    pdf_password: str = "",
    processor: Callable[..., PdfIdentity] = inspect_pdf,
    progress: Callable[[int, int, PdfIdentity], None] | None = None,
    cache: pdf_identity_cache.IdentityCache | None = None,
) -> list[PdfIdentity]:
    """Commit only required PDF rewrites, preserving identity order."""
    ordered = list(identities)
    candidates = [
        identity
        for identity in ordered
        if identity.pdf_status in PREPROCESS_NEEDED_PDF_STATUSES
    ]
    replacements: dict[Path, PdfIdentity] = {}
    for completed, identity in enumerate(candidates, start=1):
        if processor is inspect_pdf:
            try:
                evidence = pdf_evidence.inspect_pdf_with_rewrite(
                    identity.path,
                    commit=True,
                    pdf_password=pdf_password,
                )
            except Exception as exc:
                replacement = _processing_error(
                    identity.path,
                    "pdf_processing_error",
                    f"{type(exc).__name__}: {exc}",
                )
                if cache is not None and cache.usable:
                    try:
                        cache.discard(identity.path)
                    except Exception as cache_exc:
                        cache.disable(
                            "Disabled after error-entry removal failed: "
                            f"{type(cache_exc).__name__}: {cache_exc}"
                        )
            else:
                replacement = _identity_from_evidence(
                    identity.path,
                    evidence,
                )
                _store_identity_evidence(cache, identity.path, evidence)
        else:
            replacement = processor(identity.path, pdf_password)
            if cache is not None and cache.usable:
                try:
                    cache.discard(identity.path)
                except Exception as exc:
                    cache.disable(
                        "Disabled after custom-processor invalidation failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
        replacements[identity.path] = replacement
        if progress:
            progress(completed, len(candidates), replacement)
    return [replacements.get(identity.path, identity) for identity in ordered]


def _group_dois(
    identities: list[PdfIdentity],
    members: set[int],
) -> set[str]:
    return {
        identity.doi
        for index in members
        if (
            (identity := identities[index]).doi
            and identity.doi_score >= 80
        )
    }


def _doi_pair_matches(
    left: PdfIdentity,
    right: PdfIdentity,
) -> bool:
    if not left.doi or left.doi != right.doi:
        return False
    title_similarity = record_matching.title_similarity(
        left.title,
        right.title,
    )
    if min(left.doi_score, right.doi_score) >= 80:
        return bool(
            not left.title
            or not right.title
            or title_similarity >= DOI_TITLE_CONFLICT_SCORE
        )
    return title_similarity >= record_matching.LOW_SOURCE_TITLE_CONFIRM_SCORE


def _same_fast_prefix(left: PdfIdentity, right: PdfIdentity) -> bool:
    return (
        left.path.stem[:FAST_PREFIX_LENGTH].casefold()
        == right.path.stem[:FAST_PREFIX_LENGTH].casefold()
    )


def _filename_first_author(path: Path) -> str | None:
    token = path.stem.split("_", 1)[0]
    with_initials = re.fullmatch(
        r"([A-Z][A-Za-z'-]*?[a-z])([A-Z][A-Za-z]{0,2})",
        token,
    )
    if with_initials:
        return with_initials.group(1).casefold()
    if re.fullmatch(r"[A-Z][A-Za-z'-]*[a-z]", token):
        return token.casefold()
    return None


def _filename_year(path: Path) -> int | None:
    for value in re.findall(
        r"(?<!\d)((?:19|20)\d{2}|\d{2})(?!\d)",
        path.stem,
    ):
        year = int(value)
        if len(value) == 4:
            return year
        return filename_policy.resolve_two_digit_year(year)
    return None


def _filename_part_numbers(path: Path) -> frozenset[int]:
    stem = path.stem
    values = {
        int(value)
        for value in re.findall(
            r"(?:part|pt|stat|pocs|echo|crrbc)[ _-]*([1-9]\d*)",
            stem,
            re.IGNORECASE,
        )
    }
    trailing = re.search(r"(?:^|[_-])([1-9]\d*)$", stem)
    if trailing:
        values.add(int(trailing.group(1)))
    return frozenset(values)


def _title_part_numbers(title: str) -> frozenset[int]:
    return frozenset(
        int(value)
        for value in re.findall(
            r"\b(?:part|pt)\s*([1-9]\d*)\b",
            title,
            re.IGNORECASE,
        )
    )


def _is_title_duplicate_boilerplate(normalized_title: str) -> bool:
    return bool(
        normalized_title in {
            "scanned doc in pdf format",
            "scanned document from hp9100c",
        }
        or "personal use only" in normalized_title
    )


@lru_cache(maxsize=TITLE_DUPLICATE_EVIDENCE_CACHE_SIZE)
def _title_duplicate_evidence(
    path: Path,
    title: str | None,
) -> _TitleDuplicateEvidence:
    normalized_title = record_matching.normalize_title(title or "")
    return _TitleDuplicateEvidence(
        normalized_title=normalized_title,
        title_tokens=frozenset(record_matching.title_tokens(title or "")),
        filename_tokens=frozenset(record_matching.title_tokens(path.stem)),
        first_author=_filename_first_author(path),
        year=_filename_year(path),
        part_numbers=(
            _filename_part_numbers(path)
            | _title_part_numbers(title or "")
        ),
        boilerplate=_is_title_duplicate_boilerplate(normalized_title),
    )


def _token_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _title_identity_corroborates(
    left: _TitleDuplicateEvidence,
    right: _TitleDuplicateEvidence,
) -> bool:
    if left.boilerplate or right.boilerplate:
        return False
    if (
        not left.first_author
        or left.first_author != right.first_author
        or left.year is None
        or left.year != right.year
    ):
        return False
    if (
        (left.part_numbers or right.part_numbers)
        and left.part_numbers != right.part_numbers
    ):
        return False
    if left.normalized_title == right.normalized_title:
        return True
    return bool(
        _token_jaccard(left.title_tokens, right.title_tokens)
        >= TITLE_ONLY_MIN_TOKEN_JACCARD
        or _token_jaccard(left.filename_tokens, right.filename_tokens)
        >= TITLE_ONLY_MIN_FILENAME_JACCARD
    )


def title_duplicate_similarity(
    left: PdfIdentity,
    right: PdfIdentity,
) -> float:
    """Return title-only evidence strong enough to move a collection PDF."""
    left_evidence = _title_duplicate_evidence(left.path, left.title)
    right_evidence = _title_duplicate_evidence(right.path, right.title)
    if not _title_identity_corroborates(left_evidence, right_evidence):
        return 0.0
    left_tokens = left_evidence.title_tokens
    right_tokens = right_evidence.title_tokens
    if min(len(left_tokens), len(right_tokens)) < 4:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    if not union or overlap / union < 0.50:
        return 0.0
    return record_matching.title_similarity(left.title, right.title)


def _can_merge_doi_groups(
    identities: list[PdfIdentity],
    left_members: set[int],
    right_members: set[int],
) -> bool:
    return all(
        _doi_pair_matches(identities[left], identities[right])
        for left in left_members
        for right in right_members
    )


def _can_merge_title_groups(
    identities: list[PdfIdentity],
    left_members: set[int],
    right_members: set[int],
    threshold: float,
) -> bool:
    left_dois = _group_dois(identities, left_members)
    right_dois = _group_dois(identities, right_members)
    if left_dois and right_dois and left_dois.isdisjoint(right_dois):
        return False

    # Complete-linkage prevents A~B~C chains when A and C are not duplicates.
    for left in left_members:
        for right in right_members:
            left_identity = identities[left]
            right_identity = identities[right]
            if _doi_pair_matches(left_identity, right_identity):
                continue
            if (
                title_duplicate_similarity(
                    left_identity,
                    right_identity,
                )
                < threshold
            ):
                return False
    return True


def duplicate_groups(
    identities: list[PdfIdentity],
    *,
    title_threshold: float = DEFAULT_TITLE_THRESHOLD,
    fast_prefix: bool = False,
) -> list[list[int]]:
    """Cluster identities by primary DOI, then conservative title matching."""
    valid = [
        index
        for index, identity in enumerate(identities)
        if not _is_pdf_error(identity)
    ]
    sets = _DisjointSet(len(identities))

    doi_pairs: list[tuple[float, int, int]] = []
    for offset, left in enumerate(valid):
        for right in valid[offset + 1:]:
            if fast_prefix and not _same_fast_prefix(
                identities[left], identities[right]
            ):
                continue
            if not _doi_pair_matches(identities[left], identities[right]):
                continue
            similarity = record_matching.title_similarity(
                identities[left].title,
                identities[right].title,
            )
            confidence = (
                1.0
                if min(
                    identities[left].doi_score,
                    identities[right].doi_score,
                )
                >= 80
                else similarity
            )
            doi_pairs.append((confidence, left, right))

    for _confidence, left, right in sorted(doi_pairs, reverse=True):
        left_root = sets.find(left)
        right_root = sets.find(right)
        if left_root == right_root:
            continue
        if _can_merge_doi_groups(
            identities,
            sets.members[left_root],
            sets.members[right_root],
        ):
            sets.union(left_root, right_root)

    title_pairs: list[tuple[float, int, int]] = []
    for offset, left in enumerate(valid):
        for right in valid[offset + 1:]:
            if fast_prefix and not _same_fast_prefix(
                identities[left], identities[right]
            ):
                continue
            if sets.find(left) == sets.find(right):
                continue
            similarity = title_duplicate_similarity(
                identities[left],
                identities[right],
            )
            if similarity >= title_threshold:
                title_pairs.append((similarity, left, right))

    for _similarity, left, right in sorted(title_pairs, reverse=True):
        left_root = sets.find(left)
        right_root = sets.find(right)
        if left_root == right_root:
            continue
        if _can_merge_title_groups(
            identities,
            sets.members[left_root],
            sets.members[right_root],
            title_threshold,
        ):
            sets.union(left_root, right_root)

    groups = [
        sorted(members, key=lambda index: str(identities[index].path).lower())
        for members in sets.members
        if len(members) > 1
    ]
    return sorted(
        groups,
        key=lambda group: str(identities[group[0]].path).lower(),
    )


def _canonical_index(
    identities: list[PdfIdentity],
    group: list[int],
) -> int:
    def modification_time(index: int) -> int:
        try:
            return identities[index].path.stat().st_mtime_ns
        except OSError:
            return -1

    # group is path-sorted, and max() keeps the first item when mtimes tie.
    return max(group, key=modification_time)


def _is_pdf_error(identity: PdfIdentity) -> bool:
    return identity.pdf_status in pdf_rewrite.REWRITE_BLOCKING_STATUSES | {
        FAST_SKIPPED_PDF_STATUS,
        "pdf_processing_error",
        "pdf_processing_timeout",
    }


def _match_to_canonical(
    identity: PdfIdentity,
    canonical: PdfIdentity,
) -> tuple[str, float]:
    if _doi_pair_matches(identity, canonical):
        if min(identity.doi_score, canonical.doi_score) >= 80:
            return "doi", 1.0
        return (
            "doi+title",
            record_matching.title_similarity(identity.title, canonical.title),
        )
    return "title", title_duplicate_similarity(identity, canonical)


def build_results(
    identities: list[PdfIdentity],
    *,
    title_threshold: float = DEFAULT_TITLE_THRESHOLD,
    fast_prefix: bool = False,
) -> list[dict[str, str]]:
    groups = duplicate_groups(
        identities,
        title_threshold=title_threshold,
        fast_prefix=fast_prefix,
    )
    group_by_index: dict[int, tuple[int, int]] = {}
    for group_number, group in enumerate(groups, start=1):
        canonical = _canonical_index(identities, group)
        for index in group:
            group_by_index[index] = (group_number, canonical)

    results: list[dict[str, str]] = []
    for index, identity in enumerate(identities):
        group_info = group_by_index.get(index)
        if identity.pdf_status == FAST_SKIPPED_PDF_STATUS:
            status = FAST_SKIPPED_RESULT_STATUS
            group_id = ""
            canonical_path = ""
            match_method = ""
            similarity = ""
        elif identity.pdf_status == "pdf_unrepairable":
            status = "unrepairable"
            group_id = ""
            canonical_path = ""
            match_method = ""
            similarity = ""
        elif _is_pdf_error(identity):
            status = "error"
            group_id = ""
            canonical_path = ""
            match_method = ""
            similarity = ""
        elif not group_info:
            status = "unique"
            group_id = ""
            canonical_path = str(identity.path)
            match_method = ""
            similarity = ""
        else:
            group_number, canonical_index = group_info
            canonical = identities[canonical_index]
            group_id = f"D{group_number:04d}"
            canonical_path = str(canonical.path)
            if index == canonical_index:
                status = "canonical"
                match_method = ""
                similarity = ""
            else:
                status = "duplicate"
                match_method, score = _match_to_canonical(identity, canonical)
                similarity = f"{score:.3f}"

        results.append({
            "file": str(identity.path),
            "status": status,
            "group_id": group_id,
            "canonical_path": canonical_path,
            "match_method": match_method,
            "similarity": similarity,
            "doi": identity.doi or "",
            "doi_source": identity.doi_source,
            "doi_score": str(identity.doi_score) if identity.doi_score else "",
            "title": identity.title or "",
            "pdf_status": identity.pdf_status,
            "note": identity.note,
            "target_path": "",
        })
    return results


def commit_duplicates(
    root: Path,
    results: list[dict[str, str]],
    *,
    cache: pdf_identity_cache.IdentityCache | None = None,
) -> None:
    """Move duplicate rows to Duplicated, preserving every file."""
    duplicate_directory = root / "Duplicated"
    duplicates = [row for row in results if row["status"] == "duplicate"]
    if not duplicates:
        return

    for row in duplicates:
        source = Path(row["file"])
        if not source.is_file():
            row["status"] = "commit_error"
            row["note"] = "Source file no longer exists"
            continue
        payload: dict[str, object] | None = None
        if cache is not None and cache.usable:
            try:
                payload = cache.get(source)
            except Exception as exc:
                cache.disable(
                    "Disabled after pre-move cache read failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        try:
            target = collection_mutation.move_to_unique(
                source,
                duplicate_directory,
            )
        except Exception as exc:
            row["status"] = "commit_error"
            row["note"] = f"{type(exc).__name__}: {exc}"
            continue
        row["target_path"] = str(target)
        if cache is not None and cache.usable:
            try:
                if payload is not None:
                    cache.put(target, payload)
                cache.discard(source)
            except Exception as exc:
                cache.disable(
                    "Disabled after post-move cache update failed: "
                    f"{type(exc).__name__}: {exc}"
                )


def write_report(results: list[dict[str, str]], path: Path) -> None:
    fieldnames = [
        "file",
        "status",
        "group_id",
        "canonical_path",
        "match_method",
        "similarity",
        "doi",
        "doi_source",
        "doi_score",
        "title",
        "pdf_status",
        "note",
        "target_path",
    ]
    report_io.write_csv_atomic(
        path,
        results,
        fieldnames=fieldnames,
    )


@dataclass(frozen=True, slots=True)
class DedupeOptions:
    folder: Path
    commit: bool = False
    preprocess: bool = False
    exclude_dir: tuple[str, ...] = ()
    report: Path | None = None
    title_threshold: float = DEFAULT_TITLE_THRESHOLD
    workers: int = min(8, os.cpu_count() or 1)
    inspection_timeout: float = DEFAULT_INSPECTION_TIMEOUT
    fast: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.title_threshold <= 1.0:
            raise ValueError("--title-threshold must be between 0 and 1")
        if self.workers < 1:
            raise ValueError("--workers must be at least 1")
        if self.inspection_timeout <= 0:
            raise ValueError("--inspection-timeout must be greater than zero")


def run(options: DedupeOptions) -> int:
    """Inspect and optionally organize duplicates using explicit options."""
    root = options.folder.expanduser().resolve()
    if not root.is_dir():
        print(f"Folder not found or not a directory: {root}")
        return 1

    for recover, authorized in (
        (collection_mutation.recover_interrupted_replacements, options.commit),
        (pdf_rewrite.recover_interrupted_rewrites, options.preprocess or options.commit),
    ):
        recovery = recover(root, commit=authorized)
        for outcome in recovery:
            print(f"Recovery {outcome.status}: {outcome.note}", flush=True)
        if any(outcome.blocks_run for outcome in recovery):
            print("Recovery requires review or --commit; run stopped.", flush=True)
            return 2

    paths = discover_pdfs(
        root,
        excluded_directories=options.exclude_dir,
    )
    inspect_paths = paths
    skipped_prefixes: dict[Path, str] = {}
    if options.fast:
        inspect_paths, skipped_prefixes = fast_prefix_plan(paths)
        print(
            "Fast prefix mode: inspecting "
            f"{len(inspect_paths)} of {len(paths)} PDF(s); "
            f"{len(skipped_prefixes)} skipped.",
            flush=True,
        )
    else:
        print(f"Inspecting {len(paths)} PDF(s)...", flush=True)

    def print_progress(
        completed: int,
        total: int,
        identity: PdfIdentity,
    ) -> None:
        is_error = _is_pdf_error(identity)
        if completed == total or completed % 10 == 0 or is_error:
            detail = (
                f"; {identity.path.name}: {identity.pdf_status}"
                if is_error
                else ""
            )
            print(
                f"Inspection progress: {completed}/{total}{detail}",
                flush=True,
            )

    pdf_password = os.environ.get("RENAMEPDF_PDF_PASSWORD", "")
    identity_cache: pdf_identity_cache.IdentityCache | None = None
    if pdf_password:
        print(
            "Identity cache disabled because RENAMEPDF_PDF_PASSWORD is set.",
            flush=True,
        )
    else:
        candidate_cache = pdf_identity_cache.IdentityCache(
            root,
            extractor_version=IDENTITY_EXTRACTOR_VERSION,
        )
        try:
            candidate_cache.open()
        except Exception as exc:
            candidate_cache.close()
            print(
                "Identity cache unavailable; continuing without it: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        else:
            identity_cache = candidate_cache
            print(
                f"Identity cache (local): {identity_cache.local_path}",
                flush=True,
            )
            for note in identity_cache.notes:
                print(f"Identity cache: {note}", flush=True)

    try:
        inspected = inspect_pdfs(
            inspect_paths,
            pdf_password=pdf_password,
            workers=options.workers,
            inspection_timeout=options.inspection_timeout,
            progress=print_progress,
            cache=identity_cache,
        )
        preprocess_count = sum(
            identity.pdf_status in PREPROCESS_NEEDED_PDF_STATUSES
            for identity in inspected
        )
        if options.preprocess or options.commit:
            print(f"Preprocessing {preprocess_count} PDF(s)...", flush=True)

            def print_preprocess_progress(
                completed: int,
                total: int,
                identity: PdfIdentity,
            ) -> None:
                print(
                    "Preprocess progress: "
                    f"{completed}/{total}; {identity.path.name}: "
                    f"{identity.pdf_status}",
                    flush=True,
                )

            inspected = preprocess_identities(
                inspected,
                pdf_password=pdf_password,
                progress=print_preprocess_progress,
                cache=identity_cache,
            )
        inspected_by_path = {
            identity.path: identity for identity in inspected
        }
        identities = [
            inspected_by_path.get(path)
            or _fast_skipped_identity(path, skipped_prefixes[path])
            for path in paths
        ]
        results = build_results(
            identities,
            title_threshold=options.title_threshold,
            fast_prefix=options.fast,
        )
        if options.commit:
            commit_duplicates(root, results, cache=identity_cache)

        report = (
            options.report.expanduser().resolve()
            if options.report
            else root / DEFAULT_REPORT_NAME
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        write_report(results, report)

        if identity_cache is not None:
            if not identity_cache.usable:
                print(
                    "Identity cache snapshot was not published: "
                    f"{identity_cache.disabled_reason}",
                    flush=True,
                )
            else:
                try:
                    identity_cache.prune_missing()
                    snapshot = identity_cache.publish()
                except Exception as exc:
                    print(
                        "Identity cache snapshot was not published: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                else:
                    print(
                        "Identity cache: "
                        f"{identity_cache.hit_count} hit(s), "
                        f"{identity_cache.miss_count} miss(es), "
                        f"{identity_cache.write_count} write(s); "
                        f"snapshot: {snapshot}",
                        flush=True,
                    )

        duplicates = sum(row["status"] == "duplicate" for row in results)
        groups = len({
            row["group_id"] for row in results if row["group_id"]
        })
        errors = sum(
            row["status"] in {"error", "unrepairable", "commit_error"}
            for row in results
        )
        skipped = sum(
            row["status"] == FAST_SKIPPED_RESULT_STATUS
            for row in results
        )
        action = "moved" if options.commit else "detected"
        scope_parts = []
        if options.fast:
            scope_parts.append(f"{skipped} skipped by --fast")
        if options.preprocess or options.commit:
            scope_parts.append(
                f"{preprocess_count} preprocessing candidate(s)"
            )
        scope = f"; {'; '.join(scope_parts)}" if scope_parts else ""
        print(
            f"Done. {duplicates} duplicate(s) in {groups} group(s) "
            f"{action}; {errors} error(s){scope}. Report: {report}"
        )
        return 2 if errors else 0
    finally:
        if identity_cache is not None:
            try:
                identity_cache.close()
            except Exception as exc:
                print(
                    "Identity cache close warning: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
