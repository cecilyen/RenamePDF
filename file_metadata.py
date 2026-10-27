"""Descriptor-based metadata preservation for copy-backed file transactions."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import sys
from functools import lru_cache
from pathlib import Path


FileIdentity = tuple[int, int]
FileToken = tuple[int, int, int, int]


def file_token(path: Path) -> FileToken | None:
    """Identify a regular file's observed state without following symlinks."""
    try:
        info = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


_COPYFILE_METADATA = (1 << 0) | (1 << 1) | (1 << 2)
_IGNORABLE_XATTR_ERRNOS = {
    errno.EACCES,
    errno.EINVAL,
    getattr(errno, "ENODATA", errno.EINVAL),
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    errno.EPERM,
}


def _file_identity(path: Path) -> FileIdentity | None:
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return int(info.st_dev), int(info.st_ino)


def _require_destination_identity(
    destination: Path,
    expected: FileIdentity,
    stage: str,
) -> None:
    if _file_identity(destination) != expected:
        raise RuntimeError(
            f"copy destination changed {stage}: {destination}"
        )


@lru_cache(maxsize=1)
def _darwin_fcopyfile():
    function = ctypes.CDLL(None, use_errno=True).fcopyfile
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    return function


def _copy_darwin_metadata(
    source_descriptor: int,
    destination_descriptor: int,
) -> None:
    if _darwin_fcopyfile()(
        source_descriptor,
        destination_descriptor,
        None,
        _COPYFILE_METADATA,
    ) == 0:
        return
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number))


def _copy_extended_attributes(
    source_descriptor: int,
    destination_descriptor: int,
) -> None:
    operations = (
        getattr(os, "listxattr", None),
        getattr(os, "getxattr", None),
        getattr(os, "setxattr", None),
    )
    if any(operation is None for operation in operations):
        return
    listxattr, getxattr, setxattr = operations
    try:
        names = listxattr(source_descriptor)
    except OSError as exc:
        if exc.errno in _IGNORABLE_XATTR_ERRNOS:
            return
        raise
    for name in names:
        try:
            value = getxattr(source_descriptor, name)
            setxattr(destination_descriptor, name, value)
        except OSError as exc:
            if exc.errno not in _IGNORABLE_XATTR_ERRNOS:
                raise


def preserve_copy_metadata(
    source_descriptor: int,
    destination_descriptor: int,
    destination: Path,
    expected_identity: FileIdentity,
    source_info: os.stat_result,
) -> None:
    """Copy supported metadata while retaining destination-path ownership."""
    _require_destination_identity(
        destination,
        expected_identity,
        "before metadata copy",
    )
    if sys.platform == "darwin":
        _copy_darwin_metadata(source_descriptor, destination_descriptor)
        _require_destination_identity(
            destination,
            expected_identity,
            "during metadata copy",
        )
        return

    mode = stat.S_IMODE(source_info.st_mode)
    if hasattr(os, "fchmod"):
        os.fchmod(destination_descriptor, mode)
    else:
        _require_destination_identity(
            destination,
            expected_identity,
            "before mode copy",
        )
        os.chmod(destination, mode)

    timestamps = (source_info.st_atime_ns, source_info.st_mtime_ns)
    if os.utime in os.supports_fd:
        os.utime(destination_descriptor, ns=timestamps)
    else:
        _require_destination_identity(
            destination,
            expected_identity,
            "before timestamp copy",
        )
        os.utime(destination, ns=timestamps)

    _copy_extended_attributes(source_descriptor, destination_descriptor)
    _require_destination_identity(
        destination,
        expected_identity,
        "during metadata copy",
    )
