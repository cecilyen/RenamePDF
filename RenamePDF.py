#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Command-line entry point for scientific PDF organization."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

import pdf_identity_cache as identity_cache_storage
import organization_run
import collection_mutation
import pdf_evidence
import pdf_process
import pdf_rewrite
import publication_audit
import report_io
from pdf_evidence import PdfEvidence

__all__ = ["driver", "sniff_pdf_hints"]


IDENTITY_CACHE_FILENAME = identity_cache_storage.CACHE_FILENAME
IDENTITY_CACHE_NAMESPACE = (
    identity_cache_storage.DEFAULT_LOCAL_CACHE_NAMESPACE
)


def sniff_pdf_hints(
    pdf_path: Path,
    *,
    commit: bool = False,
    pdf_password: str = "",
) -> PdfEvidence:
    """Inspect one PDF and optionally perform its isolated rewrite transaction."""
    return pdf_evidence.inspect_pdf_with_rewrite(
        pdf_path,
        commit=commit,
        pdf_password=pdf_password,
    )


def _print_progress(
    completed: int,
    total: int,
    outcome: organization_run.OrganizationOutcome,
) -> None:
    pdf_detail = (
        outcome.pdf_status
        if outcome.pdf_status and outcome.pdf_status != "ok"
        else ""
    )
    detail = "; ".join(
        value
        for value in (
            pdf_detail,
            outcome.failure_category,
            outcome.note or outcome.match_source,
        )
        if value
    )
    suffix = f" ({detail})" if detail else ""
    print(
        f"[{completed}/{total}] {outcome.status} - "
        f"{outcome.source.name}{suffix}"
    )


async def driver(
    root: Path,
    commit: bool = False,
    api_key: str = "",
    email: str = "",
    newer_than: float | None = None,
    retry_failed: bool = False,
    pdf_password: str = "",
    scite_api_key: str = "",
    identity_cache: identity_cache_storage.IdentityCache | None = None,
    inspection_timeout: float = pdf_process.DEFAULT_INSPECTION_TIMEOUT,
) -> list[dict[str, Any]]:
    """Compatibility entry point returning the historical result dictionaries."""
    outcomes = await organization_run.run(
        organization_run.RunOptions(
            root=root,
            commit=commit,
            api_key=api_key,
            email=email,
            newer_than=newer_than,
            retry_failed=retry_failed,
            pdf_password=pdf_password,
            scite_api_key=scite_api_key,
            inspection_timeout=inspection_timeout,
        ),
        identity_cache=identity_cache,
        progress=_print_progress,
    )
    return [
        outcome.to_legacy_dict()
        for outcome in outcomes
    ]


def _write_logs(results: list[dict[str, Any]], root: Path) -> None:
    if not results:
        return
    report_io.write_csv_atomic(
        root / "0_results.csv",
        results,
        fieldnames=organization_run.CSV_FIELDS,
        extrasaction="ignore",
    )


def _open_identity_cache(
    root: Path,
    pdf_password: str,
) -> identity_cache_storage.IdentityCache | None:
    if pdf_password:
        print(
            "PDF identity cache disabled because "
            "RENAMEPDF_PDF_PASSWORD is set."
        )
        return None
    candidate = identity_cache_storage.IdentityCache(
        root,
        extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION,
        snapshot_filename=IDENTITY_CACHE_FILENAME,
        local_cache_namespace=IDENTITY_CACHE_NAMESPACE,
    )
    try:
        candidate.open()
    except Exception as exc:
        try:
            candidate.close()
        except Exception:
            pass
        print(
            "PDF identity cache unavailable; continuing without it: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    print(f"PDF identity cache (local): {candidate.local_path}")
    for note in candidate.notes:
        print(f"PDF identity cache: {note}")
    return candidate


def _publish_identity_cache(
    cache: identity_cache_storage.IdentityCache,
) -> None:
    if not cache.usable:
        print(
            "PDF identity cache snapshot was not published: "
            f"{cache.disabled_reason}"
        )
        return
    try:
        cache.prune_missing()
        snapshot = cache.publish()
    except Exception as exc:
        print(
            "PDF identity cache snapshot was not published: "
            f"{type(exc).__name__}: {exc}"
        )
        return
    print(
        "PDF identity cache: "
        f"{cache.hit_count} hit(s), "
        f"{cache.miss_count} miss(es), "
        f"{cache.write_count} write(s); "
        f"snapshot: {snapshot}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scientific PDF renamer with DOI/title metadata lookup and "
            "strict confidence plus publication-status checks."
        )
    )
    parser.add_argument("folder", type=str, help="Root folder to scan")
    parser.add_argument(
        "--inspection-timeout", type=pdf_process.inspection_timeout_seconds,
        default=pdf_process.DEFAULT_INSPECTION_TIMEOUT, metavar="SECONDS",
        help="Maximum PDF inspection time per file; finite and positive (default: 120)",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help=(
            "Actually normalize, compress, rename, and move files "
            "(default: dry-run)"
        ),
    )
    parser.add_argument(
        "--ncbi-api-key",
        type=str,
        default="",
        help="NCBI API key for 10 req/s PubMed limit (3 req/s without)",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="",
        help="Contact email for Crossref polite pool and NCBI E-utilities",
    )
    parser.add_argument(
        "--newer-than-results",
        action="store_true",
        help="Process only PDFs newer than the existing 0_results.csv",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Process only PDFs currently in the Failed directory",
    )
    parser.add_argument(
        "--retry-audit-incomplete",
        action="store_true",
        help=(
            "Refresh only incomplete audit fields in the existing "
            "0_results.csv; do not inspect or change PDFs"
        ),
    )
    args = parser.parse_args(argv)
    if args.newer_than_results and args.retry_failed:
        parser.error(
            "--newer-than-results and --retry-failed cannot be used together"
        )
    if args.retry_audit_incomplete and (
        args.commit or args.newer_than_results or args.retry_failed
    ):
        parser.error(
            "--retry-audit-incomplete cannot be combined with --commit, "
            "--newer-than-results, or --retry-failed"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        print(f"Folder not found: {root}")
        return 1

    if args.retry_audit_incomplete:
        results_path = root / "0_results.csv"
        if not results_path.exists():
            print(
                "Cannot retry incomplete audits because this file does not "
                f"exist: {results_path}"
            )
            return 1
        try:
            fieldnames, rows = report_io.read_csv(results_path)
            missing = tuple(
                name
                for name in publication_audit.compact_report_field_names()
                if name not in fieldnames
            )
            if missing:
                print(
                    "Cannot retry incomplete audits because the report lacks "
                    f"columns: {', '.join(missing)}"
                )
                return 1
            updated, selected = asyncio.run(
                organization_run.retry_incomplete_audits(
                    organization_run.RunOptions(
                        root=root,
                        api_key=args.ncbi_api_key,
                        email=args.email,
                        scite_api_key=os.environ.get("SCITE_API_KEY", ""),
                    ),
                    rows,
                )
            )
        except KeyboardInterrupt:
            print(
                "\nAudit retry interrupted; the existing results CSV was "
                "not replaced."
            )
            return 130
        except Exception as exc:
            print(f"Could not retry incomplete audits: {exc}")
            return 1
        if selected:
            report_io.write_csv_atomic(
                results_path,
                updated,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            print(
                f"Retried {selected} incomplete audit row(s). "
                f"CSV updated: {results_path}"
            )
        else:
            print("No retryable incomplete audit rows; CSV was not replaced.")
        return 0

    for recover in (
        collection_mutation.recover_interrupted_replacements,
        pdf_rewrite.recover_interrupted_rewrites,
    ):
        recovery = recover(root, commit=args.commit)
        for outcome in recovery:
            print(f"Recovery {outcome.status}: {outcome.note}")
        if any(outcome.blocks_run for outcome in recovery):
            print("Recovery requires review or --commit; run stopped.")
            return 2

    newer_than = None
    if args.newer_than_results:
        results_path = root / "0_results.csv"
        if not results_path.exists():
            print(
                "Cannot use --newer-than-results because this file does not "
                f"exist: {results_path}"
            )
            return 1
        newer_than = results_path.stat().st_mtime

    pdf_password = os.environ.get("RENAMEPDF_PDF_PASSWORD", "")
    identity_cache = _open_identity_cache(root, pdf_password)
    try:
        try:
            results = asyncio.run(
                driver(
                    root,
                    commit=args.commit,
                    api_key=args.ncbi_api_key,
                    email=args.email,
                    newer_than=newer_than,
                    retry_failed=args.retry_failed,
                    pdf_password=pdf_password,
                    scite_api_key=os.environ.get("SCITE_API_KEY", ""),
                    identity_cache=identity_cache,
                    inspection_timeout=args.inspection_timeout,
                )
            )
        except KeyboardInterrupt:
            print(
                "\nExecution interrupted; the existing results CSV was not "
                "replaced."
            )
            return 130

        _write_logs(results, root)
        if results and identity_cache is not None:
            _publish_identity_cache(identity_cache)
        if not results:
            print(
                f"Done. Dry-run={not args.commit}. No PDFs selected; "
                "the results CSV was not replaced."
            )
            return 0
        print(
            f"Done. Dry-run={not args.commit}. CSV saved beside: "
            f"{root / '0_results.csv'}"
        )
        return 0
    finally:
        if identity_cache is not None:
            try:
                identity_cache.close()
            except Exception as exc:
                print(
                    "PDF identity cache close warning: "
                    f"{type(exc).__name__}: {exc}"
                )


if __name__ == "__main__":
    raise SystemExit(main())
