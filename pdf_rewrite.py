"""Transactional PDF normalization and lossless compression.

This module owns every mutating PDF operation. Callers provide a path and an
optional password; the module authenticates, checks signatures, writes and
validates a replacement, and restores the original if installation fails.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pymupdf

import file_metadata


LOSSLESS_COMPRESSION_THRESHOLD_BYTES = 3_000_000


@dataclass(frozen=True, slots=True)
class InsertedCoverPage:
    page_index: int
    kind: str
    reason: str


COVER_PAGE_SCAN_LIMIT = 3


def _inserted_cover_kind(text: str) -> tuple[str, str] | None:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    lowered = normalized.casefold()
    if (
        "researchgate.net/publication" in lowered
        and (
            "see discussions, stats, and author profiles" in lowered
            or "all content following this page was uploaded" in lowered
        )
    ):
        return (
            "researchgate",
            "ResearchGate publication-profile wrapper",
        )
    if (
        "hal id:" in lowered
        and (
            "hal is a multi-disciplinary open access archive" in lowered
            or "l'archive ouverte pluridisciplinaire hal" in lowered
            or "l’archive ouverte pluridisciplinaire hal" in lowered
        )
    ):
        return (
            "hal",
            "HAL repository deposit wrapper",
        )
    if (
        "ver=pdfcov" in lowered
        and "view table of contents" in lowered
        and "published by" in lowered
        and "aip publishing" in lowered
    ):
        return (
            "aip-pdfcov",
            "AIP citation and download cover",
        )
    if (
        "this article appeared in a journal published by elsevier" in lowered
        and "attached copy is furnished to the author" in lowered
    ):
        return (
            "elsevier-author-copy",
            "Elsevier author-copy cover",
        )
    if (
        "core metadata, citation and similar papers" in lowered
        and "core.ac.uk" in lowered
    ):
        return (
            "core",
            "CORE repository metadata wrapper",
        )
    return None


def detect_inserted_cover_pages(
    document: Any,
    *,
    text_for_page: Callable[[int], str] | None = None,
    max_leading_pages: int = COVER_PAGE_SCAN_LIMIT,
) -> tuple[InsertedCoverPage, ...]:
    """Identify consecutive, high-confidence host covers at the PDF front."""
    page_count = len(document)
    if page_count < 2:
        return ()
    limit = min(max(0, max_leading_pages), page_count - 1)
    detected: list[InsertedCoverPage] = []
    for page_index in range(limit):
        try:
            text = (
                text_for_page(page_index)
                if text_for_page is not None
                else document[page_index].get_text("text")
            )
        except Exception:
            break
        match = _inserted_cover_kind(text)
        if match is None:
            break
        kind, reason = match
        detected.append(InsertedCoverPage(page_index, kind, reason))

    if not detected:
        return ()
    first_content_index = len(detected)
    try:
        content_text = (
            text_for_page(first_content_index)
            if text_for_page is not None
            else document[first_content_index].get_text("text")
        )
    except Exception:
        return ()
    if len(re.findall(r"[A-Za-z]", content_text or "")) < 40:
        return ()
    return tuple(detected)


@dataclass(frozen=True, slots=True)
class PdfRewriteOutcome:
    status: str
    note: str
    changed: bool = False
    source_token: file_metadata.FileToken | None = None


RECOVERY_BLOCKING_STATUSES = frozenset(
    {
        "pdf_recovery_needed",
        "pdf_recovery_ambiguous",
        "pdf_recovery_cleanup_blocked",
        "pdf_recovery_failed",
    }
)
RECOVERY_NONBLOCKING_STATUSES = frozenset(
    {
        "pdf_recovery_restored",
        "pdf_recovery_cleanup_needed",
        "pdf_recovery_cleanup_completed",
        "pdf_recovery_cleanup_failed",
        "pdf_recovery_completed_artifact_preserved",
        "pdf_recovery_artifact_preserved",
    }
)
RECOVERY_STATUSES = RECOVERY_BLOCKING_STATUSES | RECOVERY_NONBLOCKING_STATUSES
REWRITE_BLOCKING_STATUSES = frozenset(
    {
        "pdf_source_changed",
        "pdf_processing_error",
        "pdf_processing_timeout",
        "pdf_password_required",
        "pdf_unrepairable",
        "pdf_repair_failed",
        "pdf_signed_rewrite_blocked",
        "pdf_unreadable",
    }
)
_ROLLBACK_ARTIFACT = re.compile(
    r"^\.(?P<canonical>.+\.pdf)(?:_\d+)?\.renamepdf-rollback$",
    re.IGNORECASE,
)
_TEMPORARY_ARTIFACT = re.compile(
    r"^\.(?P<canonical>.+\.pdf)\.(?:repairing|installing)"
    r"(?:_\d+)?\.pdf$",
    re.IGNORECASE,
)
_INSTALLED_MARKER_ARTIFACT = re.compile(
    r"^\.(?P<canonical>.+\.pdf)(?:_\d+)?\.renamepdf-installed-marker$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PdfRecoveryOutcome:
    canonical_path: Path
    status: str
    note: str
    rollback_paths: tuple[Path, ...] = ()
    temporary_paths: tuple[Path, ...] = ()
    installed_marker_paths: tuple[Path, ...] = ()
    changed: bool = False

    def __post_init__(self) -> None:
        if self.status not in RECOVERY_STATUSES:
            raise ValueError(f"unknown PDF recovery status: {self.status}")

    @property
    def blocks_run(self) -> bool:
        return self.status in RECOVERY_BLOCKING_STATUSES


@dataclass(slots=True)
class _InterruptedRewrite:
    rollbacks: list[Path] = field(default_factory=list)
    temporaries: list[Path] = field(default_factory=list)
    installed_markers: list[Path] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _InstallResult:
    cleanup_note: str = ""
    source_token: file_metadata.FileToken | None = None


class _InstallFailure(RuntimeError):
    def __init__(
        self, note: str, *, source_token: file_metadata.FileToken | None = None,
    ) -> None:
        super().__init__(note)
        self.source_token = source_token


@dataclass(frozen=True, slots=True)
class _MarkerRecord:
    operation: str
    sha256: str
    expected_pages: int | None = None


class _DestinationExistsError(FileExistsError):
    pass


def _rewrite_artifact_identity(path: Path) -> tuple[Path, str] | None:
    rollback_match = _ROLLBACK_ARTIFACT.fullmatch(path.name)
    if rollback_match:
        return path.parent / rollback_match.group("canonical"), "rollback"
    temporary_match = _TEMPORARY_ARTIFACT.fullmatch(path.name)
    if temporary_match:
        return path.parent / temporary_match.group("canonical"), "temporary"
    installed_marker_match = _INSTALLED_MARKER_ARTIFACT.fullmatch(path.name)
    if installed_marker_match:
        return (
            path.parent / installed_marker_match.group("canonical"),
            "installed_marker",
        )
    return None


def is_rewrite_artifact(path: Path) -> bool:
    """Return whether a path uses RenamePDF's transaction artifact naming."""
    return _rewrite_artifact_identity(path) is not None


def _path_list(paths: tuple[Path, ...]) -> str:
    return ", ".join(str(path) for path in paths)


def _file_token(path: Path) -> tuple[int, int, int, int] | None:
    """Return a regular-file token without following symbolic links."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
    )


def _file_identity(path: Path) -> tuple[int, int] | None:
    token = _file_token(path)
    return token[:2] if token is not None else None


def _path_matches_token(
    path: Path,
    token: tuple[int, int, int, int] | None,
) -> bool:
    return token is not None and _file_token(path) == token


def _unlink_owned(
    path: Path,
    token: tuple[int, int, int, int] | None,
) -> None:
    """Unlink only the regular file observed by the current transaction."""
    if not _path_matches_token(path, token):
        raise RuntimeError(f"transaction artifact changed before removal: {path}")
    path.unlink()


def _unlink_owned_identity(
    path: Path,
    identity: tuple[int, int] | None,
) -> None:
    """Unlink an in-progress artifact only while its inode remains owned."""
    if identity is None or _file_identity(path) != identity:
        raise RuntimeError(f"transaction artifact changed before removal: {path}")
    path.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marker_payload(
    operation: str,
    pdf_path: Path,
    expected_pages: int | None = None,
    *,
    sha256: str | None = None,
) -> bytes:
    if operation not in {"install", "restore"}:
        raise ValueError(f"unsupported transaction marker operation: {operation}")
    if operation == "install" and (expected_pages is None or expected_pages < 1):
        raise ValueError("install marker requires a positive page count")
    lines = [
        "RenamePDF transaction marker v1",
        f"operation={operation}",
        f"sha256={sha256 or _sha256_file(pdf_path)}",
    ]
    if expected_pages is not None:
        lines.append(f"pages={expected_pages}")
    return ("\n".join(lines) + "\n").encode("ascii")


def _parse_marker(path: Path) -> _MarkerRecord | None:
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    if len(payload) > 4096:
        return None
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError:
        return None
    if not lines or lines[0] != "RenamePDF transaction marker v1":
        return None
    fields: dict[str, str] = {}
    for line in lines[1:]:
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            return None
        fields[key] = value
    operation = fields.get("operation")
    digest = fields.get("sha256")
    if operation not in {"install", "restore"} or not re.fullmatch(
        r"[0-9a-f]{64}", digest or ""
    ):
        return None
    pages_text = fields.get("pages")
    expected_pages: int | None = None
    if pages_text is not None:
        try:
            expected_pages = int(pages_text)
        except ValueError:
            return None
        if expected_pages < 1:
            return None
    if operation == "install" and expected_pages is None:
        return None
    if operation == "restore" and expected_pages is not None:
        return None
    if set(fields) - {"operation", "sha256", "pages"}:
        return None
    return _MarkerRecord(operation, digest, expected_pages)


def _marker_matches_target(
    path: Path,
    marker: _MarkerRecord,
    *,
    validate_pdf: bool,
) -> bool:
    token_before = _file_token(path)
    if token_before is None:
        return False
    if validate_pdf and marker.operation == "install":
        try:
            _validate_pdf(path, marker.expected_pages or 0)
        except Exception:
            return False
    try:
        digest = _sha256_file(path)
    except OSError:
        return False
    return (
        digest == marker.sha256
        and _path_matches_token(path, token_before)
    )


def _source_info(source: Path) -> os.stat_result:
    info = os.lstat(source)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"copy source is not a regular file: {source}")
    return info


def exceeds_lossless_compression_threshold(pdf_path: Path) -> bool:
    """Return whether a regular PDF is larger than the 3 MB policy limit."""
    try:
        size = int(_source_info(pdf_path).st_size)
    except OSError:
        return False
    return size > LOSSLESS_COMPRESSION_THRESHOLD_BYTES


def create_exclusive_copy(source: Path, destination: Path) -> tuple[int, int]:
    """Copy bytes to a new path without ever clobbering an existing path."""
    source_info = _source_info(source)
    source_token = (
        int(source_info.st_dev),
        int(source_info.st_ino),
        int(source_info.st_size),
        int(source_info.st_mtime_ns),
    )
    fd: int | None = None
    owned_identity: tuple[int, int] | None = None
    try:
        fd = os.open(
            destination,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_BINARY", 0),
            stat.S_IMODE(source_info.st_mode),
        )
        opened = os.fstat(fd)
        owned_identity = (int(opened.st_dev), int(opened.st_ino))
        with os.fdopen(fd, "wb") as output:
            fd = None
            with source.open("rb") as input_stream:
                opened_source = os.fstat(input_stream.fileno())
                opened_token = (
                    int(opened_source.st_dev),
                    int(opened_source.st_ino),
                    int(opened_source.st_size),
                    int(opened_source.st_mtime_ns),
                )
                if opened_token != source_token:
                    raise RuntimeError(
                        f"exclusive-copy source changed before read: {source}"
                    )
                shutil.copyfileobj(input_stream, output, length=1024 * 1024)
                output.flush()
                file_metadata.preserve_copy_metadata(
                    input_stream.fileno(),
                    output.fileno(),
                    destination,
                    owned_identity,
                    opened_source,
                )
            os.fsync(output.fileno())
        if _file_token(source) != source_token:
            raise RuntimeError(
                f"exclusive-copy source changed during read: {source}"
            )
        if _file_identity(destination) != owned_identity:
            raise RuntimeError(
                f"exclusive-copy destination changed during write: {destination}"
            )
        return owned_identity
    except BaseException:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if owned_identity is not None:
            try:
                _unlink_owned_identity(destination, owned_identity)
            except Exception:
                pass
        raise


def _copy_file_metadata(source: Path, destination: Path) -> tuple[int, int]:
    source_info = _source_info(source)
    source_token = (
        int(source_info.st_dev),
        int(source_info.st_ino),
        int(source_info.st_size),
        int(source_info.st_mtime_ns),
    )
    destination_identity = _file_identity(destination)
    if destination_identity is None:
        raise ValueError(f"metadata destination is not a regular file: {destination}")
    binary_flag = getattr(os, "O_BINARY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    source_fd: int | None = None
    destination_fd: int | None = None
    try:
        source_fd = os.open(source, os.O_RDONLY | binary_flag | nofollow_flag)
        opened_source = os.fstat(source_fd)
        opened_source_token = (
            int(opened_source.st_dev),
            int(opened_source.st_ino),
            int(opened_source.st_size),
            int(opened_source.st_mtime_ns),
        )
        if opened_source_token != source_token:
            raise RuntimeError(f"metadata source changed before read: {source}")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | binary_flag | nofollow_flag,
        )
        opened_destination = os.fstat(destination_fd)
        if (
            int(opened_destination.st_dev),
            int(opened_destination.st_ino),
        ) != destination_identity:
            raise RuntimeError(
                f"metadata destination changed before update: {destination}"
            )
        file_metadata.preserve_copy_metadata(
            source_fd,
            destination_fd,
            destination,
            destination_identity,
            opened_source,
        )
        os.fsync(destination_fd)
        if _file_token(source) != source_token:
            raise RuntimeError(f"metadata source changed during read: {source}")
        if _file_identity(destination) != destination_identity:
            raise RuntimeError(
                f"metadata destination changed during update: {destination}"
            )
        return destination_identity
    finally:
        for descriptor in (source_fd, destination_fd):
            if descriptor is not None:
                os.close(descriptor)


def _create_exclusive_copy_unique(
    source: Path,
    directory: Path,
    base_name: str,
) -> Path:
    candidate = directory / base_name
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    index = 2
    while True:
        try:
            create_exclusive_copy(source, candidate)
            return candidate
        except FileExistsError:
            candidate = directory / f"{stem}_{index}{suffix}"
            index += 1


def _write_exclusive_bytes(destination: Path, payload: bytes) -> Path:
    fd: int | None = None
    owned_identity: tuple[int, int] | None = None
    try:
        fd = os.open(
            destination,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_BINARY", 0),
            0o600,
        )
        opened = os.fstat(fd)
        owned_identity = (int(opened.st_dev), int(opened.st_ino))
        with os.fdopen(fd, "wb") as output:
            fd = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if _file_identity(destination) != owned_identity:
            raise RuntimeError(
                f"exclusive marker changed during write: {destination}"
            )
        return destination
    except BaseException:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if owned_identity is not None:
            try:
                _unlink_owned_identity(destination, owned_identity)
            except Exception:
                pass
        raise


def _write_marker_unique(
    directory: Path,
    base_name: str,
    payload: bytes,
) -> Path:
    candidate = directory / base_name
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    index = 2
    while True:
        try:
            return _write_exclusive_bytes(candidate, payload)
        except FileExistsError:
            candidate = directory / f"{stem}_{index}{suffix}"
            index += 1


def recover_interrupted_rewrites(
    root: Path,
    *,
    commit: bool = False,
) -> tuple[PdfRecoveryOutcome, ...]:
    """Detect interrupted transactions and restore only unambiguous rollbacks.

    A missing canonical PDF with exactly one rollback is safe to restore when
    ``commit`` is true. Every candidate temporary copy is preserved. States
    with competing canonical or rollback paths are reported for manual review.
    """
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Recovery root is not a directory: {root}")

    groups: dict[Path, _InterruptedRewrite] = {}
    for artifact in sorted(root.rglob("*"), key=lambda path: str(path).casefold()):
        identity = _rewrite_artifact_identity(artifact)
        if identity is None:
            continue
        if not artifact.is_file() and not artifact.is_symlink():
            continue
        canonical, kind = identity
        group = groups.setdefault(canonical, _InterruptedRewrite())
        if kind == "rollback":
            group.rollbacks.append(artifact)
        elif kind == "temporary":
            group.temporaries.append(artifact)
        else:
            group.installed_markers.append(artifact)

    outcomes: list[PdfRecoveryOutcome] = []
    for canonical, group in sorted(
        groups.items(),
        key=lambda item: str(item[0]).casefold(),
    ):
        rollbacks = tuple(group.rollbacks)
        temporaries = tuple(group.temporaries)
        installed_markers = tuple(group.installed_markers)
        all_artifacts = rollbacks + temporaries + installed_markers
        if canonical.is_symlink() or any(
            path.is_symlink() for path in all_artifacts
        ):
            outcomes.append(
                PdfRecoveryOutcome(
                    canonical,
                    "pdf_recovery_ambiguous",
                    "Recovery state includes a symbolic link; manual review "
                    f"required. Canonical: {canonical}; artifacts: "
                    f"{_path_list(all_artifacts)}",
                    rollbacks,
                    temporaries,
                    installed_markers,
                )
            )
            continue

        if installed_markers:
            marker_records = tuple(_parse_marker(marker) for marker in installed_markers)
            markers_agree = bool(marker_records) and all(
                record is not None and record == marker_records[0]
                for record in marker_records[1:]
            )
            marker_record = marker_records[0] if markers_agree else None
            if marker_record is None:
                outcomes.append(
                    PdfRecoveryOutcome(
                        canonical,
                        "pdf_recovery_ambiguous",
                        "Installed marker state is invalid or inconsistent; no "
                        f"path was changed. Canonical: {canonical}; rollback(s): "
                        f"{_path_list(rollbacks) or 'none'}; installed marker(s): "
                        f"{_path_list(installed_markers)}; temporary copy/copies: "
                        f"{_path_list(temporaries) or 'none'}",
                        rollbacks,
                        temporaries,
                        installed_markers,
                    )
                )
                continue
            if not canonical.exists() and len(rollbacks) <= 1:
                outcomes.append(
                    PdfRecoveryOutcome(
                        canonical,
                        "pdf_recovery_completed_artifact_preserved",
                        "Install completion marker is intact, but the canonical "
                        "path moved or disappeared after validation. Completed "
                        "transaction artifacts were preserved for review and "
                        "remain excluded from scanning: "
                        f"{_path_list(rollbacks + installed_markers)}",
                        rollbacks,
                        temporaries,
                        installed_markers,
                    )
                )
                continue
            completed_state = bool(
                canonical.exists()
                and len(rollbacks) <= 1
                and _marker_matches_target(
                    canonical,
                    marker_record,
                    validate_pdf=True,
                )
            )
            if not completed_state:
                outcomes.append(
                    PdfRecoveryOutcome(
                        canonical,
                        "pdf_recovery_ambiguous",
                        "Installed marker does not match the canonical PDF; no "
                        f"path was changed. Canonical: {canonical}; rollback(s): "
                        f"{_path_list(rollbacks) or 'none'}; installed marker(s): "
                        f"{_path_list(installed_markers)}; temporary copy/copies: "
                        f"{_path_list(temporaries) or 'none'}",
                        rollbacks,
                        temporaries,
                        installed_markers,
                    )
                )
                continue

            cleanup_paths = rollbacks + installed_markers
            if not commit:
                outcomes.append(
                    PdfRecoveryOutcome(
                        canonical,
                        "pdf_recovery_cleanup_needed",
                        "Validated replacement is installed; commit can remove "
                        "completed transaction backup(s): "
                        f"{_path_list(cleanup_paths)}. Temporary copy/copies "
                        f"preserved: {_path_list(temporaries) or 'none'}",
                        rollbacks,
                        temporaries,
                        installed_markers,
                    )
                )
                continue

            cleanup_entries = tuple(
                (artifact, _file_token(artifact)) for artifact in cleanup_paths
            )
            removed: list[Path] = []
            cleanup_failure: tuple[Path, Exception] | None = None
            for artifact, artifact_token in cleanup_entries:
                try:
                    _unlink_owned(artifact, artifact_token)
                except Exception as exc:
                    cleanup_failure = artifact, exc
                    break
                removed.append(artifact)
            if cleanup_failure is not None:
                artifact, exc = cleanup_failure
                outcomes.append(
                    PdfRecoveryOutcome(
                        canonical,
                        "pdf_recovery_cleanup_failed",
                        "Validated replacement remains installed; completed "
                        f"transaction cleanup failed at {artifact}: "
                        f"{type(exc).__name__}: {exc}. Remaining artifacts "
                        "stay excluded from scanning.",
                        tuple(path for path in rollbacks if path.exists()),
                        temporaries,
                        tuple(
                            path for path in installed_markers if path.exists()
                        ),
                        changed=bool(removed),
                    )
                )
                continue
            outcomes.append(
                PdfRecoveryOutcome(
                    canonical,
                    "pdf_recovery_cleanup_completed",
                    "Removed completed transaction backup(s); validated "
                    f"replacement remains installed at {canonical}. Temporary "
                    f"copy/copies preserved: {_path_list(temporaries) or 'none'}",
                    temporary_paths=temporaries,
                    changed=True,
                )
            )
            continue

        if rollbacks:
            if len(rollbacks) != 1:
                outcomes.append(
                    PdfRecoveryOutcome(
                        canonical,
                        "pdf_recovery_ambiguous",
                        "Canonical and rollback state is ambiguous; no path "
                        "was changed. Canonical: "
                        f"{canonical}; rollback(s): {_path_list(rollbacks)}; "
                        f"temporary copy/copies: {_path_list(temporaries) or 'none'}",
                        rollbacks,
                        temporaries,
                    )
                )
                continue
            rollback = rollbacks[0]
            if canonical.exists():
                outcomes.append(
                    PdfRecoveryOutcome(
                        canonical,
                        "pdf_recovery_ambiguous",
                        "Canonical and rollback files both exist; no path was "
                        f"changed. Canonical: {canonical}; rollback: {rollback}; "
                        "manual review is required because their relationship "
                        "cannot be established without a transaction marker. "
                        f"Temporary copy/copies: {_path_list(temporaries) or 'none'}",
                        rollbacks,
                        temporaries,
                    )
                )
                continue
            if not commit:
                outcomes.append(
                    PdfRecoveryOutcome(
                        canonical,
                        "pdf_recovery_needed",
                        "Canonical PDF is missing; commit can restore rollback "
                        f"{rollback} to {canonical}",
                        rollbacks,
                        temporaries,
                    )
                )
                continue
            restore_marker: Path | None = None
            restore_marker_token: tuple[int, int, int, int] | None = None
            rollback_token: tuple[int, int, int, int] | None = None
            rollback_digest: str | None = None
            try:
                rollback_token = _file_token(rollback)
                if rollback_token is None:
                    raise FileNotFoundError(f"rollback is not a regular file: {rollback}")
                rollback_digest = _sha256_file(rollback)
                create_exclusive_copy(rollback, canonical)
                restore_marker = _write_marker_unique(
                    canonical.parent,
                    f".{canonical.name}.renamepdf-installed-marker",
                    _marker_payload(
                        "restore",
                        canonical,
                        sha256=rollback_digest,
                    ),
                )
                restore_marker_token = _file_token(restore_marker)
                marker_record = _parse_marker(restore_marker)
                if (
                    marker_record is None
                    or not _marker_matches_target(
                        canonical,
                        marker_record,
                        validate_pdf=False,
                    )
                ):
                    raise RuntimeError(
                        "canonical path changed during rollback restoration"
                    )
                if not _path_matches_token(rollback, rollback_token):
                    raise RuntimeError(
                        f"rollback changed before cleanup: {rollback}"
                    )
            except Exception as exc:
                outcomes.append(
                    PdfRecoveryOutcome(
                        canonical,
                        "pdf_recovery_failed",
                        "Rollback restoration or recovery journaling failed; no "
                        "temporary copy was deleted: "
                        f"{type(exc).__name__}: {exc}",
                        rollbacks,
                        temporaries,
                        (restore_marker,) if restore_marker is not None else (),
                    )
                )
                continue
            try:
                _unlink_owned(rollback, rollback_token)
            except Exception as exc:
                outcomes.append(
                    PdfRecoveryOutcome(
                        canonical,
                        "pdf_recovery_cleanup_blocked",
                        "Rollback was restored with exclusive no-clobber semantics, "
                        "but the run must stop before the canonical path can "
                        "move; stale rollback cleanup failed: "
                        f"{type(exc).__name__}: {exc}",
                        rollbacks,
                        temporaries,
                        (restore_marker,) if restore_marker is not None else (),
                        changed=True,
                    )
                )
                continue
            try:
                if restore_marker is not None:
                    _unlink_owned(restore_marker, restore_marker_token)
            except Exception as exc:
                outcomes.append(
                    PdfRecoveryOutcome(
                        canonical,
                        "pdf_recovery_cleanup_blocked",
                        "Rollback was restored and removed, but recovery marker "
                        f"cleanup failed: {type(exc).__name__}: {exc}",
                        temporary_paths=temporaries,
                        installed_marker_paths=(restore_marker,)
                        if restore_marker is not None
                        else (),
                        changed=True,
                    )
                )
                continue
            outcomes.append(
                PdfRecoveryOutcome(
                    canonical,
                    "pdf_recovery_restored",
                    f"Restored rollback {rollback} to {canonical}; temporary "
                    f"copy/copies preserved: {_path_list(temporaries) or 'none'}",
                    (),
                    temporaries,
                    changed=True,
                )
            )
            continue

        if canonical.exists():
            outcomes.append(
                PdfRecoveryOutcome(
                    canonical,
                    "pdf_recovery_artifact_preserved",
                    "Canonical PDF exists; interrupted temporary copy/copies "
                    f"were preserved and excluded from scanning: {_path_list(temporaries)}",
                    temporary_paths=temporaries,
                )
            )
        else:
            outcomes.append(
                PdfRecoveryOutcome(
                    canonical,
                    "pdf_recovery_ambiguous",
                    "Canonical PDF and rollback are missing; temporary "
                    "copy/copies require manual review: "
                    f"{_path_list(temporaries)}",
                    temporary_paths=temporaries,
                )
            )
    return tuple(outcomes)


def _encryption_label(document: Any) -> str:
    try:
        metadata = document.metadata or {}
        return str(metadata.get("encryption") or "").strip()
    except Exception:
        return ""


def pdf_has_signatures(document: Any) -> bool:
    try:
        return int(document.get_sigflags()) > 0
    except Exception:
        return False


def _rewrite_status(
    needs_repair: bool,
    needs_decryption: bool,
    *,
    removed_cover_pages: int = 0,
    compressed: bool = False,
    committed: bool,
) -> str:
    if removed_cover_pages:
        return (
            "pdf_cover_page_removed"
            if committed
            else "pdf_cover_page_removal_needed"
        )
    if needs_repair and needs_decryption:
        return (
            "pdf_repaired_and_decrypted"
            if committed
            else "pdf_repair_and_decryption_needed"
        )
    if needs_repair:
        return "pdf_repaired" if committed else "pdf_repair_needed"
    if needs_decryption:
        return "pdf_decrypted" if committed else "pdf_decryption_needed"
    if compressed:
        return "pdf_compressed" if committed else "pdf_compression_needed"
    return "ok"


def _validated_page_indices(
    document: Any,
    page_indices: tuple[int, ...],
) -> tuple[int, ...]:
    normalized = tuple(sorted(set(page_indices)))
    page_count = len(document)
    for page_index in normalized:
        if page_index < 0 or page_index >= page_count:
            raise ValueError(
                f"cover page index {page_index} is outside 0..{page_count - 1}"
            )
    if normalized and len(normalized) >= page_count:
        raise ValueError("refusing to remove every page from a PDF")
    return normalized


def _cleanup_failed_temporary(
    temporary: Path | None,
    token: tuple[int, int, int, int] | None,
) -> str:
    if temporary is None:
        return ""
    if (
        _file_token(temporary) is None
        and not temporary.exists()
        and not temporary.is_symlink()
    ):
        return ""
    try:
        _unlink_owned(temporary, token)
    except Exception as exc:
        return (
            f"; temporary artifact was preserved because owned cleanup failed at "
            f"{temporary}: "
            f"{type(exc).__name__}: {exc}"
        )
    return ""


def _open_authenticated_pdf(
    pdf_path: Path,
    password: str,
) -> tuple[Any | None, bool, bool, PdfRewriteOutcome | None]:
    try:
        document = pymupdf.open(str(pdf_path))
    except Exception as exc:
        return (
            None,
            False,
            False,
            PdfRewriteOutcome(
                "pdf_unreadable",
                f"PyMuPDF could not open the PDF: {type(exc).__name__}: {exc}",
            ),
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
            return (
                None,
                needs_repair,
                True,
                PdfRewriteOutcome(
                    "pdf_password_required",
                    "PDF authentication failed; set RENAMEPDF_PDF_PASSWORD "
                    "to a valid user or owner password and retry",
                ),
            )

    needs_decryption = needs_decryption or bool(_encryption_label(document))
    return document, needs_repair, needs_decryption, None


def _validate_pdf(pdf_path: Path, expected_pages: int) -> None:
    check = None
    try:
        check = pymupdf.open(str(pdf_path))
        if (
            getattr(check, "needs_pass", False)
            or getattr(check, "is_encrypted", False)
            or _encryption_label(check)
        ):
            raise ValueError("rewritten PDF is still encrypted")
        if getattr(check, "is_repaired", False):
            raise ValueError("rewritten PDF still requires structural repair")
        if len(check) != expected_pages:
            raise ValueError(
                f"rewritten PDF page count changed from {expected_pages} to {len(check)}"
            )
        for page_number in range(len(check)):
            check.load_page(page_number)
    finally:
        if check is not None:
            check.close()


def _save_validated_document(
    document: Any,
    destination: Path,
    *,
    use_large_compression: bool = False,
    allow_save_repair: bool = False,
) -> tuple[int, tuple[int, int]]:
    if destination.exists() or destination.is_symlink():
        raise _DestinationExistsError(
            f"refusing to overwrite existing PDF: {destination}"
        )
    expected_pages = len(document)
    if expected_pages < 1:
        raise ValueError("PDF contains no pages")
    for page_number in range(expected_pages):
        document.load_page(page_number)
    with tempfile.TemporaryDirectory(prefix="renamepdf-render-") as directory:
        staged = Path(directory) / "document.pdf"
        document.save(
            str(staged),
            garbage=4,
            clean=False,
            deflate=1,
            deflate_images=use_large_compression,
            deflate_fonts=use_large_compression,
            incremental=False,
            expand=0,
            linear=False,
            encryption=pymupdf.PDF_ENCRYPT_NONE,
            preserve_metadata=True,
            use_objstms=use_large_compression,
            compression_effort=0,
            raise_on_repair=not allow_save_repair,
        )
        _validate_pdf(staged, expected_pages)
        staged_digest = _sha256_file(staged)
        try:
            owned_identity = create_exclusive_copy(staged, destination)
        except FileExistsError as exc:
            raise _DestinationExistsError(
                f"refusing to overwrite existing PDF: {destination}"
            ) from exc
    try:
        _validate_pdf(destination, expected_pages)
        if _file_identity(destination) != owned_identity:
            raise RuntimeError(
                f"validated destination changed after exclusive copy: {destination}"
            )
        if _sha256_file(destination) != staged_digest:
            raise RuntimeError(
                f"validated destination bytes changed after exclusive copy: {destination}"
            )
        if _file_identity(destination) != owned_identity:
            raise RuntimeError(
                f"validated destination changed during digest check: {destination}"
            )
    except Exception as validation_error:
        try:
            _unlink_owned_identity(destination, owned_identity)
        except Exception as cleanup_error:
            raise RuntimeError(
                "validated destination failed post-copy verification; foreign "
                f"path preserved at {destination}: {type(cleanup_error).__name__}: "
                f"{cleanup_error}"
            ) from validation_error
        raise
    return expected_pages, owned_identity


def write_validated_copy(
    source: Path,
    destination: Path,
    *,
    pdf_password: str = "",
    remove_page_indices: tuple[int, ...] = (),
    allow_signed_source: bool = False,
) -> Path:
    """Write a complete, unencrypted PyMuPDF copy and validate every page."""
    document, needs_repair, _needs_decryption, failure = (
        _open_authenticated_pdf(source, pdf_password)
    )
    if failure is not None:
        raise ValueError(failure.note)
    if document is None:
        raise ValueError("PyMuPDF did not return a document")
    try:
        if pdf_has_signatures(document) and not allow_signed_source:
            raise ValueError(
                "refusing to normalize a digitally signed PDF because "
                "rewriting would invalidate its signatures"
            )
        page_indices = _validated_page_indices(
            document,
            remove_page_indices,
        )
        if page_indices:
            document.delete_pages(page_indices)
        _save_validated_document(
            document,
            destination,
            use_large_compression=exceeds_lossless_compression_threshold(
                source
            ),
            allow_save_repair=needs_repair,
        )
        return destination
    finally:
        document.close()


def _write_validated_copy(
    document: Any,
    pdf_path: Path,
    *,
    use_large_compression: bool = False,
    allow_save_repair: bool = False,
) -> tuple[Path, int, tuple[int, int, int, int]]:
    base_name = f".{pdf_path.name}.repairing.pdf"
    candidate = pdf_path.parent / base_name
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    index = 2
    while True:
        try:
            expected_pages, owned_identity = _save_validated_document(
                document,
                candidate,
                use_large_compression=use_large_compression,
                allow_save_repair=allow_save_repair,
            )
            break
        except _DestinationExistsError:
            candidate = pdf_path.parent / f"{stem}_{index}{suffix}"
            index += 1
    try:
        _copy_file_metadata(pdf_path, candidate)
        temporary_token = _file_token(candidate)
        if temporary_token is None:
            raise RuntimeError(
                f"validated temporary changed before transaction: {candidate}"
            )
        return candidate, expected_pages, temporary_token
    except Exception:
        _unlink_owned_identity(candidate, owned_identity)
        raise


def _install_replacement(
    pdf_path: Path,
    temporary: Path,
    expected_pages: int,
    *,
    temporary_token: tuple[int, int, int, int] | None = None,
    expected_source_token: file_metadata.FileToken | None = None,
) -> _InstallResult:
    rollback: Path | None = None
    rollback_token: tuple[int, int, int, int] | None = None
    installed_marker: Path | None = None
    installed_marker_token: tuple[int, int, int, int] | None = None
    original_token = _file_token(pdf_path)
    installed_token: tuple[int, int, int, int] | None = None
    original_digest: str | None = None
    replacement_digest: str | None = None
    try:
        if expected_source_token is not None and original_token != expected_source_token:
            raise RuntimeError("source changed since analysis")
        observed_temporary_token = _file_token(temporary)
        if observed_temporary_token is None or (
            temporary_token is not None
            and observed_temporary_token != temporary_token
        ):
            raise RuntimeError(
                f"validated temporary changed before installation: {temporary}"
            )
        temporary_token = observed_temporary_token
        if original_token is None:
            raise ValueError(
                f"canonical PDF is not a regular file: {pdf_path}"
            )
        original_digest = _sha256_file(pdf_path)
        rollback = _create_exclusive_copy_unique(
            pdf_path,
            pdf_path.parent,
            f".{pdf_path.name}.renamepdf-rollback",
        )
        rollback_token = _file_token(rollback)
        if _sha256_file(rollback) != original_digest:
            raise RuntimeError(
                "rollback copy does not match the original canonical PDF; "
                f"original copy preserved at {rollback}"
            )
        if not _path_matches_token(pdf_path, original_token):
            raise RuntimeError(
                "canonical path changed while preserving the original; "
                f"original copy preserved at {rollback}"
        )
        _unlink_owned(pdf_path, original_token)
        if not _path_matches_token(temporary, temporary_token):
            raise RuntimeError(
                f"validated temporary changed before digest: {temporary}"
            )
        replacement_digest = _sha256_file(temporary)
        if not _path_matches_token(temporary, temporary_token):
            raise RuntimeError(
                f"validated temporary changed during digest: {temporary}"
            )
        create_exclusive_copy(temporary, pdf_path)
        if not _path_matches_token(temporary, temporary_token):
            raise RuntimeError(
                f"validated temporary changed during installation: {temporary}"
            )
        installed_token = _file_token(pdf_path)
        if installed_token is None:
            raise RuntimeError("installed canonical PDF is not a regular file")
        if _sha256_file(pdf_path) != replacement_digest:
            raise RuntimeError(
                "canonical path changed during replacement installation"
            )
        _validate_pdf(pdf_path, expected_pages)
        if not _path_matches_token(pdf_path, installed_token):
            raise RuntimeError(
                "canonical path changed before install completion was marked"
            )
        installed_marker = _write_marker_unique(
            pdf_path.parent,
            f".{pdf_path.name}.renamepdf-installed-marker",
            _marker_payload(
                "install",
                pdf_path,
                expected_pages,
                sha256=replacement_digest,
            ),
        )
        installed_marker_token = _file_token(installed_marker)
        marker_record = _parse_marker(installed_marker)
        if marker_record is None:
            raise RuntimeError("install completion marker could not be read")
        if not _marker_matches_target(
            pdf_path,
            marker_record,
            validate_pdf=False,
        ):
            raise RuntimeError(
                "canonical path changed after install completion was marked"
            )
    except Exception as install_error:
        recovery_notes: list[str] = []
        restored_token: file_metadata.FileToken | None = None
        if _path_matches_token(pdf_path, installed_token):
            try:
                _unlink_owned(pdf_path, installed_token)
            except Exception as remove_error:
                recovery_notes.append(
                    "installed canonical cleanup failed; original preserved "
                    f"at {rollback}: {type(remove_error).__name__}: "
                    f"{remove_error}"
                )
        if rollback is None:
            recovery_notes.append(
                "original rollback was not created; canonical path was left "
                "unchanged"
            )
        elif not pdf_path.exists() and not pdf_path.is_symlink():
            restored = False
            restore_digest: str | None = None
            try:
                restore_digest = _sha256_file(rollback)
                if restore_digest != original_digest:
                    raise RuntimeError("rollback no longer matches the original PDF")
                create_exclusive_copy(rollback, pdf_path)
                restored_token = _file_token(pdf_path)
                restored = True
            except Exception as restore_error:
                recovery_notes.append(
                    "original restore failed; rollback preserved at "
                    f"{rollback}: {type(restore_error).__name__}: "
                    f"{restore_error}"
                )
            if restored:
                restore_marker: Path | None = None
                restore_marker_token: tuple[int, int, int, int] | None = None
                try:
                    restore_marker = _write_marker_unique(
                        pdf_path.parent,
                        f".{pdf_path.name}.renamepdf-installed-marker",
                        _marker_payload(
                            "restore",
                            pdf_path,
                            sha256=restore_digest,
                        ),
                    )
                    restore_marker_token = _file_token(restore_marker)
                    restore_record = _parse_marker(restore_marker)
                    if (
                        restore_record is None
                        or not _marker_matches_target(
                            pdf_path,
                            restore_record,
                            validate_pdf=False,
                        )
                    ):
                        raise RuntimeError(
                            "canonical path changed during original restoration"
                        )
                except Exception as marker_error:
                    recovery_notes.append(
                        "original restore journaling failed; rollback preserved at "
                        f"{rollback}: {type(marker_error).__name__}: {marker_error}"
                    )
                else:
                    try:
                        if not _path_matches_token(rollback, rollback_token):
                            raise RuntimeError(
                                f"rollback changed before cleanup: {rollback}"
                            )
                        _unlink_owned(rollback, rollback_token)
                    except Exception as cleanup_error:
                        recovery_notes.append(
                            "original was restored; stale rollback cleanup failed at "
                            f"{rollback}: {type(cleanup_error).__name__}: "
                            f"{cleanup_error}"
                        )
                    else:
                        if restore_marker is not None:
                            try:
                                _unlink_owned(restore_marker, restore_marker_token)
                            except Exception as marker_cleanup_error:
                                recovery_notes.append(
                                    "original was restored; recovery marker cleanup "
                                    "failed at "
                                    f"{restore_marker}: "
                                    f"{type(marker_cleanup_error).__name__}: "
                                    f"{marker_cleanup_error}"
                                )
        elif (
            installed_token is None
            and _path_matches_token(pdf_path, original_token)
        ):
            try:
                if not _path_matches_token(rollback, rollback_token):
                    raise RuntimeError(
                        f"rollback changed before cleanup: {rollback}"
                    )
                rollback.unlink()
            except Exception as cleanup_error:
                recovery_notes.append(
                    "canonical original remains installed, but stale rollback "
                    f"cleanup failed at {rollback}: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        else:
            recovery_notes.append(
                "canonical path exists after install failure; original "
                f"preserved at {rollback} for recovery"
            )
        if installed_marker is not None:
            try:
                _unlink_owned(installed_marker, installed_marker_token)
            except Exception as marker_cleanup_error:
                recovery_notes.append(
                    "install completion marker was preserved at "
                    f"{installed_marker}: {type(marker_cleanup_error).__name__}: "
                    f"{marker_cleanup_error}"
                )
        note = f"{type(install_error).__name__}: {install_error}"
        if recovery_notes:
            note += "; " + "; ".join(recovery_notes)
        # Only a fully verified, journal-free restoration may authorize later
        # placement. An observed current token alone is not identity evidence.
        verified_restore = (
            restored_token
            if not recovery_notes and _path_matches_token(pdf_path, restored_token)
            else None
        )
        raise _InstallFailure(note, source_token=verified_restore) from install_error

    cleanup_errors: list[str] = []
    try:
        _unlink_owned(temporary, temporary_token)
    except Exception as exc:
        cleanup_errors.append(
            f"{temporary}: {type(exc).__name__}: {exc}"
        )
    marker_record = _parse_marker(installed_marker)
    if (
        marker_record is None
        or not _path_matches_token(pdf_path, installed_token)
        or not _marker_matches_target(
            pdf_path,
            marker_record,
            validate_pdf=False,
        )
    ):
        raise RuntimeError(
            "canonical path changed before original rollback cleanup; "
            f"original preserved at {rollback}"
        )
    try:
        _unlink_owned(rollback, rollback_token)
    except Exception as exc:
        cleanup_errors.append(
            f"{rollback}: {type(exc).__name__}: {exc}"
        )
    else:
        try:
            _unlink_owned(installed_marker, installed_marker_token)
        except Exception as exc:
            cleanup_errors.append(
                f"{installed_marker}: {type(exc).__name__}: {exc}"
            )
    if cleanup_errors:
        return _InstallResult(
            "post-install cleanup failed; preserved transaction artifact(s): "
            + "; ".join(cleanup_errors),
            source_token=installed_token,
        )
    return _InstallResult(source_token=installed_token)


def install_validated_copy(
    pdf_path: Path,
    validated_copy: Path,
) -> PdfRewriteOutcome:
    """Install an already validated PDF with rollback protection."""
    temporary: Path | None = None
    temporary_token: tuple[int, int, int, int] | None = None
    try:
        check = pymupdf.open(str(validated_copy))
        try:
            expected_pages = len(check)
        finally:
            check.close()
        _validate_pdf(validated_copy, expected_pages)
        temporary = _create_exclusive_copy_unique(
            validated_copy,
            pdf_path.parent,
            f".{pdf_path.name}.installing.pdf",
        )
        temporary_token = _file_token(temporary)
        if temporary_token is None:
            raise RuntimeError(
                f"validated temporary changed before metadata copy: {temporary}"
            )
        _copy_file_metadata(pdf_path, temporary)
        temporary_token = _file_token(temporary)
        if temporary_token is None:
            raise RuntimeError(
                f"validated temporary changed before installation: {temporary}"
            )
        _validate_pdf(temporary, expected_pages)
        install_result = _install_replacement(
            pdf_path,
            temporary,
            expected_pages,
            temporary_token=temporary_token,
        )
        temporary = None
        temporary_token = None
    except Exception as exc:
        cleanup_note = _cleanup_failed_temporary(temporary, temporary_token)
        return PdfRewriteOutcome(
            "pdf_repair_failed",
            "PyMuPDF could not install a validated replacement PDF: "
            f"{type(exc).__name__}: {exc}{cleanup_note}",
            source_token=exc.source_token if isinstance(exc, _InstallFailure) else None,
        )
    note = "Installed a validated replacement PDF"
    if install_result.cleanup_note:
        note += f"; {install_result.cleanup_note}"
    return PdfRewriteOutcome(
        "pdf_validated_copy_installed",
        note,
        changed=True,
        source_token=install_result.source_token,
    )


def rewrite_pdf(
    pdf_path: Path,
    *,
    pdf_password: str = "",
    remove_page_indices: tuple[int, ...] = (),
    compress_if_large: bool = False,
    expected_source_token: file_metadata.FileToken | None = None,
) -> PdfRewriteOutcome:
    """Normalize or compress a PDF and install a validated replacement."""
    original_token = _file_token(pdf_path)
    if expected_source_token is not None and original_token != expected_source_token:
        return PdfRewriteOutcome("pdf_source_changed", "Source changed since analysis; rerun analysis")
    document, needs_repair, needs_decryption, failure = _open_authenticated_pdf(
        pdf_path,
        pdf_password,
    )
    if failure is not None:
        return failure
    if document is None:
        return PdfRewriteOutcome(
            "pdf_repair_failed",
            "PyMuPDF did not return a document for rewriting",
        )

    temporary: Path | None = None
    temporary_token: tuple[int, int, int, int] | None = None
    source_size = 0
    replacement_size = 0
    try:
        source_size = int(_source_info(pdf_path).st_size)
        if _file_token(pdf_path) != original_token:
            return PdfRewriteOutcome("pdf_source_changed", "Source changed while opening for rewrite")
        use_large_compression = (
            source_size > LOSSLESS_COMPRESSION_THRESHOLD_BYTES
        )
        compression_requested = compress_if_large and use_large_compression
        try:
            page_indices = _validated_page_indices(
                document,
                remove_page_indices,
            )
        except Exception as exc:
            return PdfRewriteOutcome(
                "pdf_repair_failed",
                "PDF rewrite request was invalid: "
                f"{type(exc).__name__}: {exc}",
            )
        normalization_required = bool(
            needs_repair or needs_decryption or page_indices
        )
        if not normalization_required and not compression_requested:
            return PdfRewriteOutcome("ok", "PDF does not require rewriting")
        if pdf_has_signatures(document):
            if not normalization_required:
                return PdfRewriteOutcome(
                    "pdf_signed_compression_skipped",
                    "Lossless compression was skipped because rewriting "
                    "would invalidate a digital signature",
                )
            return PdfRewriteOutcome(
                "pdf_signed_rewrite_blocked",
                "PDF rewrite was skipped because rewriting "
                "would invalidate a digital signature",
            )
        if page_indices:
            detected = tuple(page.page_index for page in detect_inserted_cover_pages(document))
            if detected != page_indices:
                return PdfRewriteOutcome("pdf_source_changed", "Inserted-cover evidence changed since analysis")
            document.delete_pages(page_indices)
        temporary, expected_pages, temporary_token = _write_validated_copy(
            document,
            pdf_path,
            use_large_compression=use_large_compression,
            allow_save_repair=needs_repair,
        )
        replacement_size = int(_source_info(temporary).st_size)
        if (
            compression_requested
            and not normalization_required
            and replacement_size >= source_size
        ):
            cleanup_note = _cleanup_failed_temporary(
                temporary,
                temporary_token,
            )
            temporary = None
            temporary_token = None
            if cleanup_note:
                return PdfRewriteOutcome(
                    "pdf_compression_failed",
                    "Lossless compression produced no smaller PDF"
                    f"{cleanup_note}",
                )
            return PdfRewriteOutcome(
                "pdf_compression_not_beneficial",
                "Lossless compression was skipped because the validated "
                f"output was not smaller ({source_size} -> "
                f"{replacement_size} bytes)",
            )
    except Exception as exc:
        cleanup_note = _cleanup_failed_temporary(temporary, temporary_token)
        normalization_required = bool(
            needs_repair or needs_decryption or remove_page_indices
        )
        return PdfRewriteOutcome(
            (
                "pdf_unrepairable"
                if normalization_required
                else "pdf_compression_failed"
            ),
            "PyMuPDF could not create a validated "
            f"{'normalized' if normalization_required else 'compressed'} PDF: "
            f"{type(exc).__name__}: {exc}{cleanup_note}",
        )
    finally:
        document.close()

    try:
        install_result = _install_replacement(
            pdf_path,
            temporary,
            expected_pages,
            temporary_token=temporary_token,
            expected_source_token=original_token,
        )
        temporary = None
        temporary_token = None
    except Exception as exc:
        cleanup_note = _cleanup_failed_temporary(temporary, temporary_token)
        normalization_required = bool(
            needs_repair or needs_decryption or remove_page_indices
        )
        return PdfRewriteOutcome(
            (
                "pdf_repair_failed"
                if normalization_required
                else "pdf_compression_failed"
            ),
            "PyMuPDF could not install a validated "
            f"{'normalized' if normalization_required else 'compressed'} PDF: "
            f"{type(exc).__name__}: {exc}{cleanup_note}",
            source_token=exc.source_token if isinstance(exc, _InstallFailure) else None,
        )

    actions: list[str] = []
    if needs_repair:
        actions.append("repaired PDF structure")
    if needs_decryption:
        actions.append("removed PDF encryption")
    if page_indices:
        page_numbers = ", ".join(str(index + 1) for index in page_indices)
        actions.append(f"removed inserted cover page(s) {page_numbers}")
    if compression_requested:
        saved_bytes = max(0, source_size - replacement_size)
        actions.append(
            "applied lossless compression "
            f"({source_size} -> {replacement_size} bytes; "
            f"saved {saved_bytes} bytes)"
        )
    note = (
        f"PyMuPDF {' and '.join(actions)} and installed a validated replacement"
    )
    if install_result.cleanup_note:
        note += f"; {install_result.cleanup_note}"
    return PdfRewriteOutcome(
        _rewrite_status(
            needs_repair,
            needs_decryption,
            removed_cover_pages=len(page_indices),
            compressed=compression_requested,
            committed=True,
        ),
        note,
        changed=True,
        source_token=install_result.source_token,
    )
