"""Bounded, crash-released interprocess locks kept outside synced collections."""

from __future__ import annotations

import getpass
import hashlib
import os
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def local_lock(resource: Path, *, timeout: float = 30.0) -> Iterator[None]:
    """Serialize local processes for one resource; never lock across machines.

    A separate SQLite file supplies portable OS-backed exclusion. It contains no
    collection data and must not be unlinked while another process may use it.
    """
    if timeout < 0:
        raise ValueError("lock timeout must not be negative")
    user = str(os.getuid()) if hasattr(os, "getuid") else getpass.getuser()
    user_key = hashlib.sha256(user.encode()).hexdigest()[:16]
    directory = Path(tempfile.gettempdir()) / f"renamepdf-locks-{user_key}"
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise OSError("Local lock directory is not a regular directory")
    os.chmod(directory, 0o700)
    key = hashlib.sha256(os.fsencode(str(resource.resolve()))).hexdigest()
    path = directory / f"{key}.sqlite"
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if path.is_symlink() or not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("Local lock is not a regular file")
    finally:
        os.close(descriptor)
    connection = sqlite3.connect(path, timeout=timeout)
    try:
        try:
            connection.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            if getattr(exc, "sqlite_errorcode", 0) & 0xff in {
                sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED,
            }:
                raise TimeoutError("Another local process holds the operation lock") from exc
            raise
        yield
    finally:
        connection.rollback()
        connection.close()
