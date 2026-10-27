#!/usr/bin/env python3

"""Command-line adapter for the reusable reviewed-replacement workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import audit_report
import replacement_run


def parse_args(argv: list[str] | None = None) -> replacement_run.ReplacementOptions:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve, validate, and optionally install replacements selected "
            "by a current AuditPDF report."
        )
    )
    parser.add_argument("folder", type=Path, help="Collection root")
    parser.add_argument(
        "--email",
        required=True,
        help="Contact email required by Crossref and Unpaywall",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Archive originals and install validated replacements",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        help=f"AuditPDF CSV (default: {audit_report.DEFAULT_REPORT_NAME})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Replacement CSV (default: 0_replace_results.csv)",
    )
    return replacement_run.ReplacementOptions(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    return replacement_run.run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
