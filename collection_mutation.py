"""Conservative, rollback-aware mutations of a PDF collection."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, Sequence

import file_lock
import file_metadata


PlacementAction = Literal[
    "unchanged",
    "canonical",
    "duplicate",
    "canonical_replaced",
]


class CollectionMutationError(RuntimeError):
    """A collection mutation could not be completed without data loss."""


class CollectionRollbackError(CollectionMutationError):
    """A mutation failed and its rollback also failed."""


@dataclass(frozen=True, slots=True)
class PlacementOutcome:
    action: PlacementAction
    target: Path
    displaced: Path | None = None


@dataclass(frozen=True, slots=True)
class ReplacementOutcome:
    archive: Path
    target: Path


@dataclass(frozen=True, slots=True)
class ReplacementRecoveryOutcome:
    journal: Path
    status: str
    note: str
    blocks_run: bool = True


@dataclass(frozen=True, slots=True)
class _ReplacementRecord:
    source: Path
    archive: Path
    target: Path
    original: dict[str, object]
    replacement: dict[str, object]


_REPLACEMENT_JOURNAL = ".renamepdf-replacement-*.json"


def is_replacement_artifact(path: Path) -> bool:
    """Never rediscover downloaded staging copies after an interrupted run."""
    return any(part.startswith(".published-") for part in path.parts[:-1])


_FileToken = tuple[int, int, int, int]
_FileIdentity = tuple[int, int]
_AT_FDCWD = -100
_DARWIN_RENAME_EXCL = 0x00000004
_LINUX_RENAME_NOREPLACE = 0x00000001
_NATIVE_FALLBACK_ERRNOS = {
    errno.EXDEV,
    errno.ENOSYS,
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}


class _NoReplaceUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=None)
def _libc_function(name: str):
    try:
        return getattr(ctypes.CDLL(None, use_errno=True), name)
    except AttributeError as exc:
        raise _NoReplaceUnavailable(name) from exc


def _raise_native_rename_error(source: Path, target: Path) -> None:
    error_number = ctypes.get_errno()
    raise OSError(
        error_number,
        os.strerror(error_number),
        source,
        target,
    )


def _native_move_noreplace(source: Path, target: Path) -> None:
    """Use the host's atomic no-replace rename when it exposes one."""
    if sys.platform == "darwin":
        renamex = _libc_function("renamex_np")
        renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex.restype = ctypes.c_int
        if (
            renamex(
                os.fsencode(source),
                os.fsencode(target),
                _DARWIN_RENAME_EXCL,
            )
            != 0
        ):
            _raise_native_rename_error(source, target)
        return
    if sys.platform.startswith("linux"):
        renameat2 = _libc_function("renameat2")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(target),
            _LINUX_RENAME_NOREPLACE,
        ) != 0:
            _raise_native_rename_error(source, target)
        return
    if os.name == "nt":
        os.rename(source, target)
        return
    raise _NoReplaceUnavailable(sys.platform)


def _file_token(path: Path) -> _FileToken | None:
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


def _file_identity(path: Path) -> _FileIdentity | None:
    token = _file_token(path)
    return token[:2] if token is not None else None


def _remove_owned_path(path: Path, identity: _FileIdentity | None) -> None:
    if identity is None or _file_identity(path) != identity:
        raise RuntimeError(f"destination changed before cleanup: {path}")
    path.unlink()


def _copy_exclusive_and_remove(source: Path, target: Path) -> None:
    """Cross-volume no-clobber fallback: copy, sync, verify, then remove."""
    source_info = os.lstat(source)
    if not stat.S_ISREG(source_info.st_mode):
        raise ValueError(f"move source is not a regular file: {source}")
    source_token: _FileToken = (
        int(source_info.st_dev),
        int(source_info.st_ino),
        int(source_info.st_size),
        int(source_info.st_mtime_ns),
    )
    binary_flag = getattr(os, "O_BINARY", 0)
    input_fd: int | None = None
    output_fd: int | None = None
    destination_identity: _FileIdentity | None = None
    try:
        input_fd = os.open(source, os.O_RDONLY | binary_flag)
        opened_source = os.fstat(input_fd)
        opened_token: _FileToken = (
            int(opened_source.st_dev),
            int(opened_source.st_ino),
            int(opened_source.st_size),
            int(opened_source.st_mtime_ns),
        )
        if opened_token != source_token:
            raise RuntimeError(f"source changed before move: {source}")

        output_fd = os.open(
            target,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | binary_flag,
            stat.S_IMODE(source_info.st_mode),
        )
        opened_target = os.fstat(output_fd)
        if not stat.S_ISREG(opened_target.st_mode):
            raise RuntimeError(f"reserved destination is not a regular file: {target}")
        destination_identity = (
            int(opened_target.st_dev),
            int(opened_target.st_ino),
        )

        with (
            os.fdopen(input_fd, "rb") as input_stream,
            os.fdopen(output_fd, "wb") as output_stream,
        ):
            input_fd = None
            output_fd = None
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            file_metadata.preserve_copy_metadata(
                input_stream.fileno(),
                output_stream.fileno(),
                target,
                destination_identity,
                opened_source,
            )
            os.fsync(output_stream.fileno())

        if _file_token(source) != source_token:
            raise RuntimeError(f"source changed during move: {source}")
        if _file_identity(target) != destination_identity:
            raise RuntimeError(f"destination changed during move: {target}")
        if target.stat().st_size != source_token[2]:
            raise RuntimeError(f"destination size changed during move: {target}")

        if _file_token(source) != source_token:
            raise RuntimeError(f"source changed before removal: {source}")
        if _file_identity(target) != destination_identity:
            raise RuntimeError(f"destination changed before completion: {target}")
        source.unlink()
    except BaseException as error:
        for descriptor in (input_fd, output_fd):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if destination_identity is not None:
            try:
                _remove_owned_path(target, destination_identity)
            except Exception as cleanup_error:
                raise CollectionMutationError(
                    f"Move failed ({type(error).__name__}: {error}); partial "
                    f"destination cleanup also failed ({type(cleanup_error).__name__}: "
                    f"{cleanup_error})"
                ) from cleanup_error
        raise


def _move_exclusive(source: Path, target: Path) -> None:
    """Move a file atomically when possible, without replacing a destination."""
    if _file_token(source) is None:
        raise ValueError(f"move source is not a regular file: {source}")
    try:
        _native_move_noreplace(source, target)
        return
    except _NoReplaceUnavailable:
        pass
    except OSError as exc:
        if exc.errno not in _NATIVE_FALLBACK_ERRNOS:
            raise
    _copy_exclusive_and_remove(source, target)


def _rename_path(source: Path, target: Path) -> None:
    """Compatibility seam for the collection's no-clobber move primitive."""
    if source == target:
        return
    try:
        same_file = source.samefile(target)
    except OSError:
        same_file = False
    if same_file:
        same_parent = source.parent.samefile(target.parent)
        identity = _file_identity(source)
        if identity is None or _file_identity(target) != identity:
            raise RuntimeError("same-file destination is not a regular file")
        if not same_parent:
            target_entries = {
                entry.name for entry in target.parent.iterdir()
                if _file_identity(entry) == identity
            }
            if source.stat().st_nlink < 2 or target.name not in target_entries:
                raise RuntimeError("same-file destination is not a verified hard link")
            source.unlink()
            return
        entry_names = {
            entry.name
            for entry in source.parent.iterdir()
            if _file_identity(entry) == identity
        }
        if source.name == target.name and source.name in entry_names:
            return
        if source.name in entry_names and target.name not in entry_names:
            source.rename(target)
        elif source.name in entry_names and target.name in entry_names:
            if source.stat().st_nlink < 2:
                raise RuntimeError("same-file destination is not a verified hard link")
            source.unlink()
        elif target.name not in entry_names:
            raise RuntimeError(
                f"same-file directory entry changed before normalization: {source}"
            )
        return
    _move_exclusive(source, target)


def _leaf_name(name: str) -> str:
    candidate = Path(name)
    if not name or candidate.name != name or name in {".", ".."}:
        raise ValueError(f"Destination must be one filename, got {name!r}")
    return name


def _require_file(path: Path, role: str) -> None:
    if not path.is_file():
        raise CollectionMutationError(f"{role} file does not exist: {path}")


def ensure_directories(root: Path, names: Sequence[str]) -> tuple[Path, ...]:
    directories: list[Path] = []
    for name in names:
        directory = root / _leaf_name(name)
        directory.mkdir(parents=True, exist_ok=True)
        directories.append(directory)
    return tuple(directories)


def _numbered_destination(directory: Path, name: str, index: int) -> Path:
    name = _leaf_name(name)
    if index == 0:
        return directory / name
    candidate = Path(name)
    return directory / f"{candidate.stem}_{index}{candidate.suffix}"


def move_to_unique(source: Path, directory: Path, name: str | None = None) -> Path:
    """Move one file into a directory under a collision-safe filename."""
    _require_file(source, "Source")
    directory.mkdir(parents=True, exist_ok=True)
    destination_name = name or source.name
    index = 0
    while True:
        target = _numbered_destination(directory, destination_name, index)
        try:
            _rename_path(source, target)
        except FileExistsError:
            index += 1
            continue
        except Exception as exc:
            raise CollectionMutationError(
                f"Could not move {source} to {target}: {type(exc).__name__}: {exc}"
            ) from exc
        return target


def _restore_or_raise(
    rollback_source: Path,
    rollback_target: Path,
    *,
    operation_error: Exception,
    operation: str,
) -> None:
    try:
        _rename_path(rollback_source, rollback_target)
    except Exception as rollback_error:
        raise CollectionRollbackError(
            f"{operation} failed ({type(operation_error).__name__}: "
            f"{operation_error}); rollback {rollback_source} -> "
            f"{rollback_target} also failed ({type(rollback_error).__name__}: "
            f"{rollback_error})"
        ) from rollback_error


def place_canonical(
    source: Path,
    canonical: Path,
    duplicate_directory: Path,
) -> PlacementOutcome:
    """Place the newest copy at the canonical path and preserve the other copy."""
    _require_file(source, "Source")
    canonical.parent.mkdir(parents=True, exist_ok=True)

    if source == canonical:
        return PlacementOutcome("unchanged", canonical)

    if canonical.exists():
        try:
            same_file = source.samefile(canonical)
        except OSError:
            same_file = False
        if same_file:
            try:
                _rename_path(source, canonical)
            except Exception as exc:
                raise CollectionMutationError(
                    f"Could not normalize canonical filename {source} -> "
                    f"{canonical}: {type(exc).__name__}: {exc}"
                ) from exc
            return PlacementOutcome("canonical", canonical)

    if not canonical.exists():
        try:
            _rename_path(source, canonical)
        except Exception as exc:
            raise CollectionMutationError(
                f"Could not place canonical file {source} -> {canonical}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return PlacementOutcome("canonical", canonical)

    _require_file(canonical, "Canonical")
    try:
        source_is_newer = (
            source.stat().st_mtime_ns > canonical.stat().st_mtime_ns
        )
    except OSError as exc:
        raise CollectionMutationError(
            f"Could not compare collection copies: {type(exc).__name__}: {exc}"
        ) from exc

    if not source_is_newer:
        duplicate = move_to_unique(source, duplicate_directory, canonical.name)
        return PlacementOutcome("duplicate", duplicate)

    try:
        displaced = move_to_unique(
            canonical,
            duplicate_directory,
            canonical.name,
        )
    except Exception as exc:
        raise CollectionMutationError(
            f"Could not preserve existing canonical file {canonical} -> "
            f"{duplicate_directory}: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        _rename_path(source, canonical)
    except Exception as exc:
        _restore_or_raise(
            displaced,
            canonical,
            operation_error=exc,
            operation=f"Canonical placement {source} -> {canonical}",
        )
        raise CollectionMutationError(
            f"Could not place newer canonical file {source} -> {canonical}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return PlacementOutcome("canonical_replaced", canonical, displaced)


def prefer_newest_as_canonical(canonical: Path, duplicate: Path) -> bool:
    """Swap preserved copies when the duplicate has the newer modification time."""
    if not canonical.is_file() or not duplicate.is_file():
        return False
    if duplicate.stat().st_mtime_ns <= canonical.stat().st_mtime_ns:
        return False

    try:
        temporary = move_to_unique(
            canonical,
            duplicate.parent,
            f".{duplicate.name}.canonical-swap",
        )
    except Exception as exc:
        raise CollectionMutationError(
            f"Could not begin canonical swap {canonical} -> {duplicate.parent}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        _rename_path(duplicate, canonical)
    except Exception as exc:
        _restore_or_raise(
            temporary,
            canonical,
            operation_error=exc,
            operation=f"Canonical swap {duplicate} -> {canonical}",
        )
        raise CollectionMutationError(
            f"Could not promote duplicate {duplicate} -> {canonical}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        _rename_path(temporary, duplicate)
    except Exception as exc:
        try:
            _rename_path(canonical, duplicate)
            _rename_path(temporary, canonical)
        except Exception as rollback_error:
            raise CollectionRollbackError(
                f"Canonical swap failed ({type(exc).__name__}: {exc}); "
                f"rollback also failed ({type(rollback_error).__name__}: "
                f"{rollback_error})"
            ) from rollback_error
        raise CollectionMutationError(
            f"Could not preserve prior canonical at {duplicate}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return True


def archive_source(
    root: Path,
    source: Path,
    *,
    directory_name: str = "Superseded",
    verify_preserved: Callable[[], None] | None = None,
) -> Path:
    """Move an original into a collision-safe collection archive."""
    archive_directory = ensure_directories(root, (directory_name,))[0]
    if verify_preserved:
        verify_preserved()
    archive = move_to_unique(source, archive_directory)
    if verify_preserved:
        try:
            verify_preserved()
        except Exception as exc:
            _restore_or_raise(
                archive, source, operation_error=exc,
                operation="Existing replacement changed during archive",
            )
            raise CollectionMutationError(str(exc)) from exc
    return archive


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, getattr(errno, "ENOTSUP", errno.EINVAL)}:
                raise
    finally:
        os.close(descriptor)


def _transaction_path(root: Path, path: Path) -> Path:
    path = path.expanduser().absolute()
    relative = path.relative_to(root)
    if not relative.parts or ".." in relative.parts:
        raise ValueError("Replacement path must stay within the collection")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Replacement path contains a symbolic link")
    return path


def _replacement_fingerprint(path: Path) -> dict[str, object]:
    before = _file_token(path)
    if before is None:
        raise CollectionMutationError(f"Replacement transaction needs a regular file: {path}")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != before:
            raise CollectionMutationError(f"File changed before transaction inspection: {path}")
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if _file_token(path) != before:
        raise CollectionMutationError(f"File changed during transaction inspection: {path}")
    return {"size": before[2], "mtime_ns": before[3], "sha256": digest}


def _matches_fingerprint(path: Path, expected: dict[str, object]) -> bool:
    try:
        return _replacement_fingerprint(path) == expected
    except (OSError, CollectionMutationError):
        return False


def _write_replacement_journal(root: Path, record: dict[str, object]) -> Path:
    path = root / f".renamepdf-replacement-{uuid.uuid4().hex}.json"
    payload = json.dumps(record, sort_keys=True).encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(root)
    return path


def _remove_replacement_journal(path: Path) -> None:
    path.unlink()
    _sync_directory(path.parent)


def _read_replacement_journal(root: Path, path: Path) -> _ReplacementRecord:
    _transaction_path(root, path)
    token = _file_token(path)
    if token is None or token[2] > 16_384:
        raise ValueError("Invalid replacement journal file")
    with path.open("rb") as handle:
        record = json.loads(handle.read(16_385))
    if _file_token(path) != token:
        raise ValueError("Replacement journal changed while reading")
    fields = {"version", "source", "archive", "target", "original", "replacement"}
    if (
        not isinstance(record, dict) or set(record) != fields
        or type(record["version"]) is not int or record["version"] != 1
    ):
        raise ValueError("Unsupported replacement journal format")
    for field in ("source", "archive", "target"):
        value = record[field]
        if not isinstance(value, str) or Path(value).is_absolute():
            raise ValueError("Journal paths must be collection-relative")
        record[field] = _transaction_path(root, root / value)
        if record[field].suffix.casefold() != ".pdf":
            raise ValueError("Replacement journal must name PDF files")
    if record["archive"] in {record["source"], record["target"]}:
        raise ValueError("Journal archive overlaps an active path")
    for field in ("original", "replacement"):
        fingerprint = record[field]
        if (
            not isinstance(fingerprint, dict)
            or set(fingerprint) != {"size", "mtime_ns", "sha256"}
            or type(fingerprint["size"]) is not int
            or fingerprint["size"] < 0
            or type(fingerprint["mtime_ns"]) is not int
            or not isinstance(fingerprint["sha256"], str)
            or len(fingerprint["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint["sha256"])
        ):
            raise ValueError("Invalid replacement journal fingerprint")
    return _ReplacementRecord(
        source=record["source"], archive=record["archive"], target=record["target"],
        original=record["original"], replacement=record["replacement"],
    )


def _recover_replacement(root: Path, journal: Path, commit: bool) -> ReplacementRecoveryOutcome:
    try:
        record = _read_replacement_journal(root, journal)
        source, archive, target = record.source, record.archive, record.target
        original = record.original
        archived = _matches_fingerprint(archive, original)
        installed = _matches_fingerprint(target, record.replacement)
        if archived and installed and (source == target or not os.path.lexists(source)):
            action = "completed"
        elif _matches_fingerprint(source, original) and not os.path.lexists(archive):
            action = "cancelled"
        elif archived and not os.path.lexists(source) and not os.path.lexists(target):
            action = "restored"
        else:
            return ReplacementRecoveryOutcome(
                journal, "replacement_recovery_ambiguous",
                f"Preserved competing or changed files; review {journal}",
            )
        if not commit:
            return ReplacementRecoveryOutcome(
                journal, "replacement_recovery_needed",
                f"Replacement can be {action} with --commit; journal: {journal}",
            )
        if action == "restored":
            _rename_path(archive, source)
            if not _matches_fingerprint(source, original):
                raise CollectionMutationError("Restored original failed its fingerprint check")
            _sync_directory(source.parent)
            _sync_directory(archive.parent)
        _remove_replacement_journal(journal)
        return ReplacementRecoveryOutcome(
            journal, f"replacement_recovery_{action}",
            f"Replacement {action}; original: {source}; archive: {archive}; target: {target}",
            blocks_run=False,
        )
    except Exception as exc:
        return ReplacementRecoveryOutcome(
            journal, "replacement_recovery_failed",
            f"Preserved replacement state at {journal}: {type(exc).__name__}: {exc}",
        )


def recover_interrupted_replacements(
    root: Path, *, commit: bool = False,
) -> tuple[ReplacementRecoveryOutcome, ...]:
    """Recover only fingerprint-verified replacement transactions, without clobber."""
    root = root.expanduser().resolve()
    groups: dict[Path, list[Path]] = {}
    for journal in sorted(root.rglob(_REPLACEMENT_JOURNAL)):
        groups.setdefault(journal.parent, []).append(journal)
    outcomes: list[ReplacementRecoveryOutcome] = []
    for collection_root, journals in groups.items():
        try:
            with file_lock.local_lock(collection_root / ".renamepdf-replacement-lock", timeout=0):
                outcomes.extend(
                    _recover_replacement(collection_root, path, commit)
                    for path in sorted(collection_root.glob(_REPLACEMENT_JOURNAL))
                )
        except TimeoutError as exc:
            outcomes.append(ReplacementRecoveryOutcome(
                journals[0], "replacement_recovery_busy", str(exc),
            ))
        except Exception as exc:
            outcomes.append(ReplacementRecoveryOutcome(
                journals[0], "replacement_recovery_failed", str(exc),
            ))
    return tuple(outcomes)


def install_replacement(
    root: Path,
    source: Path,
    replacement: Path,
    target: Path,
    *,
    archive_directory_name: str = "Superseded",
) -> ReplacementOutcome:
    """Journal archive/install before either move, retaining crash recovery state."""
    root = root.expanduser().absolute()
    source, replacement, target = (
        _transaction_path(root, path) for path in (source, replacement, target)
    )
    if any(path.suffix.casefold() != ".pdf" for path in (source, replacement, target)):
        raise CollectionMutationError("Replacement transaction requires PDF paths")
    with file_lock.local_lock(root / ".renamepdf-replacement-lock"):
        if any(root.glob(_REPLACEMENT_JOURNAL)):
            raise CollectionMutationError("Pending replacement journal requires recovery first")
        if os.path.lexists(target) and target != source:
            raise CollectionMutationError(f"Replacement target already exists: {target}")
        if source == replacement or replacement == target:
            raise CollectionMutationError("Replacement staging path must be distinct")
        original = _replacement_fingerprint(source)
        updated = _replacement_fingerprint(replacement)
        archive_directory = root / _leaf_name(archive_directory_name)
        _transaction_path(root, archive_directory)
        archive_directory.mkdir(exist_ok=True)
        index = 0
        while True:
            archive = _numbered_destination(archive_directory, source.name, index)
            index += 1
            if os.path.lexists(archive):
                continue
            if archive in {source, replacement, target}:
                raise CollectionMutationError("Archive overlaps a replacement transaction path")
            journal = _write_replacement_journal(root, {
                "version": 1,
                "source": source.relative_to(root).as_posix(),
                "archive": archive.relative_to(root).as_posix(),
                "target": target.relative_to(root).as_posix(),
                "original": original,
                "replacement": updated,
            })
            try:
                if not _matches_fingerprint(source, original) or not _matches_fingerprint(replacement, updated):
                    raise CollectionMutationError("Replacement inputs changed before archiving")
                try:
                    _rename_path(source, archive)
                except FileExistsError:
                    _remove_replacement_journal(journal)
                    continue
                _sync_directory(source.parent)
                _sync_directory(archive.parent)
                _rename_path(replacement, target)
                if not _matches_fingerprint(archive, original) or not _matches_fingerprint(target, updated):
                    raise CollectionMutationError("Replacement outputs failed their fingerprint checks")
                _sync_directory(target.parent)
            except Exception as exc:
                if _matches_fingerprint(archive, original) and not os.path.lexists(source):
                    _restore_or_raise(
                        archive, source, operation_error=exc,
                        operation=f"Replacement installation; journal: {journal}",
                    )
                if _matches_fingerprint(source, original) and not os.path.lexists(archive):
                    _remove_replacement_journal(journal)
                    detail = f"original preserved at {source}"
                else:
                    detail = f"recovery journal retained at {journal}"
                raise CollectionMutationError(
                    f"Replacement installation failed; {detail}: {exc}"
                ) from exc
            _remove_replacement_journal(journal)
            return ReplacementOutcome(archive=archive, target=target)
