"""Shared two-tier cache for reusable PDF evidence."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from types import TracebackType
from urllib.parse import quote

import file_lock
import file_metadata

CACHE_FILENAME = ".renamepdf_identity_cache.sqlite"
CACHE_FORMAT_VERSION = "1"
DEFAULT_COMMIT_INTERVAL = 100
DEFAULT_LOCAL_CACHE_NAMESPACE = "renamepdf-identity-cache"
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


class IdentityCache:
    """Keep SQLite activity local and publish one atomic collection snapshot."""

    def __init__(
        self,
        collection_root: Path,
        *,
        extractor_version: str,
        temp_root: Path | None = None,
        commit_interval: int = DEFAULT_COMMIT_INTERVAL,
        snapshot_filename: str = CACHE_FILENAME,
        local_cache_namespace: str = DEFAULT_LOCAL_CACHE_NAMESPACE,
    ) -> None:
        if not extractor_version:
            raise ValueError("extractor_version must not be empty")
        if commit_interval < 1:
            raise ValueError("commit_interval must be at least 1")
        if (
            not snapshot_filename
            or snapshot_filename in {".", ".."}
            or Path(snapshot_filename).name != snapshot_filename
        ):
            raise ValueError("snapshot_filename must be one filename")
        if (
            not local_cache_namespace
            or local_cache_namespace in {".", ".."}
            or Path(local_cache_namespace).name != local_cache_namespace
        ):
            raise ValueError("local_cache_namespace must be one name")
        self.collection_root = collection_root.expanduser().resolve()
        self.extractor_version = extractor_version
        self.snapshot_path = self.collection_root / snapshot_filename
        user_id = (
            os.getuid() if hasattr(os, "getuid")
            else hashlib.sha256(getpass.getuser().encode()).hexdigest()[:16]
        )
        base = (
            temp_root.expanduser().resolve()
            if temp_root is not None
            else Path(tempfile.gettempdir())
            / f"{local_cache_namespace}-{user_id}"
        )
        root_digest = hashlib.sha256(
            os.fsencode(str(self.collection_root))
        ).hexdigest()[:24]
        self.local_directory = base / root_digest
        self.local_path = self.local_directory / snapshot_filename
        self.commit_interval = commit_interval
        self.hit_count = 0
        self.miss_count = 0
        self.write_count = 0
        self.notes: list[str] = []
        self.disabled_reason: str | None = None
        self._writes_since_commit = 0
        self._connection: sqlite3.Connection | None = None

    @property
    def usable(self) -> bool:
        return self.disabled_reason is None

    def disable(self, reason: str) -> None:
        if self.disabled_reason is None:
            self.disabled_reason = reason
            self.notes.append(reason)

    def __enter__(self) -> IdentityCache:
        self.open()
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _active_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("identity cache is not open")
        return self._connection

    def _metadata_matches(self, path: Path) -> bool:
        if path.is_symlink() or not path.is_file():
            return False
        uri = f"file:{quote(str(path), safe='/')}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=5)
            try:
                rows = dict(connection.execute(
                    "SELECT key, value FROM cache_meta"
                ))
                integrity = connection.execute(
                    "PRAGMA quick_check"
                ).fetchone()
            finally:
                connection.close()
        except sqlite3.OperationalError as exc:
            if getattr(exc, "sqlite_errorcode", 0) & 0xff in {
                sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED,
            }:
                raise
            return False
        except (OSError, sqlite3.Error, ValueError):
            return False
        return bool(
            rows.get("format_version") == CACHE_FORMAT_VERSION
            and rows.get("extractor_version") == self.extractor_version
            and integrity
            and integrity[0] == "ok"
        )

    @staticmethod
    def _remove_local_database(path: Path) -> None:
        for candidate in (
            path,
            path.with_name(f"{path.name}-journal"),
            path.with_name(f"{path.name}-shm"),
            path.with_name(f"{path.name}-wal"),
        ):
            candidate.unlink(missing_ok=True)

    @staticmethod
    def _copy_file_atomic(source: Path, destination: Path) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with source.open("rb") as input_handle:
                with os.fdopen(descriptor, "wb") as output_handle:
                    descriptor = -1
                    shutil.copyfileobj(input_handle, output_handle)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
            os.chmod(temporary, PRIVATE_FILE_MODE)
            os.replace(temporary, destination)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise

    def _prepare_local_database(self) -> None:
        self.local_directory.mkdir(
            mode=PRIVATE_DIRECTORY_MODE,
            parents=True,
            exist_ok=True,
        )
        os.chmod(self.local_directory, PRIVATE_DIRECTORY_MODE)
        if self._metadata_matches(self.local_path):
            return
        if self.local_path.exists() or self.local_path.is_symlink():
            self.notes.append("Discarded an invalid local identity cache")
            self._remove_local_database(self.local_path)
        if self._metadata_matches(self.snapshot_path):
            self._copy_file_atomic(self.snapshot_path, self.local_path)
            self.notes.append(
                "Seeded the local identity cache from the collection snapshot"
            )
        elif self.snapshot_path.exists() or self.snapshot_path.is_symlink():
            self.notes.append(
                "Ignored an invalid collection identity cache snapshot"
            )

    def open(self) -> None:
        if self._connection is not None:
            return
        with file_lock.local_lock(self.local_path):
            self._open_local_database()

    def _open_local_database(self) -> None:
        self._prepare_local_database()
        connection = sqlite3.connect(self.local_path, timeout=30)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS identity_cache (
                    relative_path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    extractor_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    cached_at_ns INTEGER NOT NULL
                )
                """
            )
            connection.executemany(
                """
                INSERT OR REPLACE INTO cache_meta (key, value)
                VALUES (?, ?)
                """,
                (
                    ("format_version", CACHE_FORMAT_VERSION),
                    ("extractor_version", self.extractor_version),
                ),
            )
            connection.commit()
            os.chmod(self.local_path, PRIVATE_FILE_MODE)
        except BaseException:
            connection.close()
            raise
        self._connection = connection

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.commit()
            finally:
                connection.close()

    def _relative_path(self, path: Path) -> str:
        normalized = path.expanduser().resolve()
        return normalized.relative_to(self.collection_root).as_posix()

    def get(self, path: Path) -> dict[str, object] | None:
        connection = self._active_connection()
        try:
            relative_path = self._relative_path(path)
            metadata = path.stat()
            token = file_metadata.file_token(path)
        except (OSError, ValueError):
            self.miss_count += 1
            return None
        row = connection.execute(
            """
            SELECT size, mtime_ns, extractor_version, payload_json
            FROM identity_cache
            WHERE relative_path = ?
            """,
            (relative_path,),
        ).fetchone()
        if (
            token is None
            or token != (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
            or row is None
            or row[0] != metadata.st_size
            or row[1] != metadata.st_mtime_ns
            or row[2] != self.extractor_version
        ):
            self.miss_count += 1
            return None
        try:
            payload = json.loads(row[3])
        except (TypeError, ValueError):
            connection.execute(
                "DELETE FROM identity_cache WHERE relative_path = ?",
                (relative_path,),
            )
            self.miss_count += 1
            return None
        if not isinstance(payload, dict):
            connection.execute(
                "DELETE FROM identity_cache WHERE relative_path = ?",
                (relative_path,),
            )
            self.miss_count += 1
            return None
        if file_metadata.file_token(path) != token:
            self.miss_count += 1
            return None
        payload["path"] = str(path)
        payload["_source_token"] = list(token)
        self.hit_count += 1
        return payload

    def put(self, path: Path, payload: dict[str, object]) -> None:
        connection = self._active_connection()
        try:
            relative_path = self._relative_path(path)
            token = file_metadata.file_token(path)
        except (OSError, ValueError):
            return
        stored_payload = dict(payload)
        expected_token = stored_payload.pop("_source_token", None)
        if token is None or (
            expected_token is not None and list(token) != expected_token
        ):
            self.discard(path)
            return
        stored_payload["path"] = relative_path
        encoded = json.dumps(
            stored_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        )
        if file_metadata.file_token(path) != token:
            self.discard(path)
            return
        connection.execute(
            """
            INSERT INTO identity_cache (
                relative_path,
                size,
                mtime_ns,
                extractor_version,
                payload_json,
                cached_at_ns
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(relative_path) DO UPDATE SET
                size = excluded.size,
                mtime_ns = excluded.mtime_ns,
                extractor_version = excluded.extractor_version,
                payload_json = excluded.payload_json,
                cached_at_ns = excluded.cached_at_ns
            """,
            (
                relative_path,
                token[2],
                token[3],
                self.extractor_version,
                encoded,
                time.time_ns(),
            ),
        )
        self.write_count += 1
        self._writes_since_commit += 1
        if self._writes_since_commit >= self.commit_interval:
            connection.commit()
            self._writes_since_commit = 0

    def discard(self, path: Path) -> None:
        try:
            relative_path = self._relative_path(path)
        except ValueError:
            return
        self._active_connection().execute(
            "DELETE FROM identity_cache WHERE relative_path = ?",
            (relative_path,),
        )

    def prune_missing(self) -> None:
        """Remove entries whose collection-relative files no longer exist."""
        connection = self._active_connection()
        stale: list[str] = []
        for row in connection.execute(
            "SELECT relative_path FROM identity_cache"
        ):
            relative_path = str(row[0])
            candidate = Path(relative_path)
            if (
                candidate.is_absolute()
                or ".." in candidate.parts
                or not (self.collection_root / candidate).is_file()
            ):
                stale.append(relative_path)
        connection.executemany(
            "DELETE FROM identity_cache WHERE relative_path = ?",
            ((relative_path,) for relative_path in stale),
        )

    def publish(self) -> Path:
        """Atomically copy a consistent local SQLite snapshot to the root."""
        if not self.usable:
            raise RuntimeError(self.disabled_reason or "identity cache disabled")
        connection = self._active_connection()
        connection.commit()
        self._writes_since_commit = 0
        # Commit before waiting: another publisher may need to read this DB.
        with file_lock.local_lock(self.snapshot_path):
            return self._publish_locked(connection)

    def _publish_locked(self, connection: sqlite3.Connection) -> Path:
        descriptor, snapshot_name = tempfile.mkstemp(
            dir=self.local_directory,
            prefix=".identity-snapshot.",
            suffix=".sqlite",
        )
        os.close(descriptor)
        local_snapshot = Path(snapshot_name)
        try:
            destination = sqlite3.connect(local_snapshot)
            try:
                connection.backup(destination)
                integrity = destination.execute(
                    "PRAGMA integrity_check"
                ).fetchone()
                if not integrity or integrity[0] != "ok":
                    raise sqlite3.DatabaseError(
                        "identity cache snapshot failed integrity_check"
                    )
            finally:
                destination.close()
            os.chmod(local_snapshot, PRIVATE_FILE_MODE)
            self._copy_file_atomic(local_snapshot, self.snapshot_path)
        finally:
            local_snapshot.unlink(missing_ok=True)
        return self.snapshot_path
