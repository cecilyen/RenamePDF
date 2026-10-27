"""Atomic serialization for collection CSV reports."""

from __future__ import annotations

import csv
import os
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


NEW_REPORT_MODE = 0o600


def read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    """Read a report while retaining its exact column order."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV report has no header: {path}")
        return tuple(reader.fieldnames), [dict(row) for row in reader]


def write_csv_atomic(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
    extrasaction: str = "raise",
) -> None:
    """Replace a CSV only after its complete same-directory write succeeds."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=False,
    )
    temporary = Path(temporary_name)
    try:
        handle = os.fdopen(
            descriptor,
            "w",
            newline="",
            encoding="utf-8",
        )
        descriptor = -1
        with handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction=extrasaction,
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        mode = (
            stat.S_IMODE(path.stat().st_mode)
            if path.exists()
            else NEW_REPORT_MODE
        )
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
