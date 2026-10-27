#!/usr/bin/env python3

"""Command-line adapter for the reusable duplicate workflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import dedupe_run


def parse_args(argv: list[str] | None = None) -> dedupe_run.DedupeOptions:
    parser = argparse.ArgumentParser(
        description=(
            "Detect duplicate scientific PDFs using RenamePDF's embedded DOI "
            "and title extraction. Dry-run is the default."
        )
    )
    parser.add_argument("folder", type=Path, help="Root folder to scan recursively")
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Move duplicates to Duplicated/; never delete files",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help=(
            "Transactionally repair, decrypt, and remove supported cover "
            "pages; do not move duplicates unless --commit is also used"
        ),
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Exclude every directory component named NAME "
            "(repeatable; always excludes Duplicated, Failed, Superseded)"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=f"CSV output path (default: FOLDER/{dedupe_run.DEFAULT_REPORT_NAME})",
    )
    parser.add_argument(
        "--title-threshold",
        type=float,
        default=dedupe_run.DEFAULT_TITLE_THRESHOLD,
        metavar="0..1",
        help=(
            "Minimum RenamePDF title similarity "
            f"(default: {dedupe_run.DEFAULT_TITLE_THRESHOLD:.2f})"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Concurrent isolated PDF inspection workers (default: up to 8)",
    )
    parser.add_argument(
        "--inspection-timeout",
        type=float,
        default=dedupe_run.DEFAULT_INSPECTION_TIMEOUT,
        metavar="SECONDS",
        help=(
            "Maximum dry-run inspection time per PDF "
            f"(default: {dedupe_run.DEFAULT_INSPECTION_TIMEOUT:g})"
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Partial scan: inspect only PDFs sharing their first seven "
            "filename characters with another PDF; other rows are reported "
            "as skipped_fast_prefix"
        ),
    )
    args = parser.parse_args(argv)
    args.exclude_dir = tuple(args.exclude_dir)
    try:
        return dedupe_run.DedupeOptions(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> int:
    return dedupe_run.run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
