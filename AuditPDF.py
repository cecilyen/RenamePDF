#!/usr/bin/env python3

"""Create the reviewed audit report consumed by ReplacePDF."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import audit_report
import collection_mutation
import organization_run
import pdf_evidence
import pdf_process
import pdf_identity_cache as identity_cache_storage
import pdf_rewrite
import report_io


IDENTITY_CACHE_FILENAME = identity_cache_storage.CACHE_FILENAME
IDENTITY_CACHE_NAMESPACE = (
    identity_cache_storage.DEFAULT_LOCAL_CACHE_NAMESPACE
)


def _print_progress(
    completed: int,
    total: int,
    outcome: organization_run.OrganizationOutcome,
) -> None:
    detail = "; ".join(
        value
        for value in (
            outcome.pdf_status if outcome.pdf_status != "ok" else "",
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
            "Read-only audit of PDF identity, formal publication updates, "
            "commentary, document condition, and replacement eligibility."
        )
    )
    parser.add_argument("folder", type=Path, help="Root folder to audit")
    parser.add_argument(
        "--inspection-timeout", type=pdf_process.inspection_timeout_seconds,
        default=pdf_process.DEFAULT_INSPECTION_TIMEOUT, metavar="SECONDS",
        help="Maximum PDF inspection time per file; finite and positive (default: 120)",
    )
    parser.add_argument(
        "--email",
        default="",
        help="Contact email for Crossref and NCBI requests",
    )
    parser.add_argument(
        "--ncbi-api-key",
        default="",
        help="NCBI API key for the higher PubMed request limit",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help=f"CSV report path (default: {audit_report.DEFAULT_REPORT_NAME})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.folder.expanduser().resolve()
    if not root.is_dir():
        print(f"Folder not found or not a directory: {root}")
        return 1

    for recover in (
        collection_mutation.recover_interrupted_replacements,
        pdf_rewrite.recover_interrupted_rewrites,
    ):
        recovery = recover(root, commit=False)
        for outcome in recovery:
            print(f"Recovery {outcome.status}: {outcome.note}")
        if any(outcome.blocks_run for outcome in recovery):
            print("Recovery requires review; audit stopped.")
            return 2

    report = (
        args.report or root / audit_report.DEFAULT_REPORT_NAME
    ).expanduser().resolve()
    pdf_password = os.environ.get("RENAMEPDF_PDF_PASSWORD", "")
    identity_cache = _open_identity_cache(root, pdf_password)
    try:
        try:
            rows = asyncio.run(
                organization_run.run_audit(
                    organization_run.AuditOptions(
                        root=root,
                        api_key=args.ncbi_api_key,
                        email=args.email,
                        pdf_password=pdf_password,
                        scite_api_key=os.environ.get("SCITE_API_KEY", ""),
                        inspection_timeout=args.inspection_timeout,
                    ),
                    identity_cache=identity_cache,
                    progress=_print_progress,
                )
            )
        except KeyboardInterrupt:
            print(
                "\nAudit interrupted; the existing report and identity "
                "cache snapshot were not replaced."
            )
            return 130
        except Exception as exc:
            print(f"Audit failed; the existing report was not replaced: {exc}")
            return 1

        try:
            report_io.write_csv_atomic(
                report,
                rows,
                fieldnames=audit_report.AUDIT_REPORT_FIELDS,
                extrasaction="ignore",
            )
        except Exception as exc:
            print(
                "Audit report was not written; the identity-cache snapshot "
                f"was not published: {type(exc).__name__}: {exc}"
            )
            return 1
        print(f"Audited PDFs: {len(rows)}")
        print(f"Report: {report}")
        if identity_cache is not None:
            _publish_identity_cache(identity_cache)
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
