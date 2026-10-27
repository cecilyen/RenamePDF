"""Focused tests consolidated by validation surface."""

import ast
import stat
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest import mock

import dedupe_run
import AuditPDF
import collection_mutation
import RenamePDF
import replacement_run
import filename_policy
import metadata_providers
import organization_run
import pdf_rewrite
import record_matching
import report_io


class ReplacementPreflightTests(unittest.TestCase):
    def test_all_commands_check_replacement_recovery_before_other_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocked = collection_mutation.ReplacementRecoveryOutcome(
                root / ".renamepdf-replacement-test.json", "replacement_recovery_needed", "fixture",
            )
            calls = (
                (lambda: RenamePDF.main([str(root), "--commit"]), True),
                (lambda: AuditPDF.main([str(root)]), False),
                (lambda: dedupe_run.run(dedupe_run.DedupeOptions(root, preprocess=True)), False),
                (lambda: dedupe_run.run(dedupe_run.DedupeOptions(root, commit=True)), True),
                (lambda: replacement_run.run(replacement_run.ReplacementOptions(root, "test@example.test", commit=True)), True),
            )
            for index, (run, commit) in enumerate(calls):
                with (
                    self.subTest(command=index),
                    mock.patch.object(collection_mutation, "recover_interrupted_replacements", return_value=(blocked,)) as recovery,
                    mock.patch.object(pdf_rewrite, "recover_interrupted_rewrites") as rewrite,
                ):
                    self.assertEqual(run(), 2)
                    self.assertEqual(recovery.call_args.kwargs["commit"], commit)
                    rewrite.assert_not_called()

    def test_download_staging_is_not_discovered_after_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            active = root / "paper.pdf"
            active.write_bytes(b"fixture")
            staging = root / ".published-interrupted"
            staging.mkdir()
            (staging / "download.pdf").write_bytes(b"download fixture")
            (staging / "normalized.pdf").write_bytes(b"normalized fixture")
            self.assertEqual(dedupe_run.discover_pdfs(root), [active])
            self.assertEqual(organization_run._discover_pdfs(organization_run.RunOptions(root)), [active])


# Cross-cutting safety contracts


class ModuleBoundaryTests(unittest.TestCase):
    def test_library_imports_are_acyclic_and_do_not_depend_on_clis(self) -> None:
        root = Path(__file__).resolve().parent.parent
        modules = {path.stem: path for path in root.glob("*.py")}
        cli_names = {"RenamePDF", "DedupePDF", "AuditPDF", "ReplacePDF"}

        def imports(node: ast.AST):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
                if node.test.id == "TYPE_CHECKING":
                    return
            if isinstance(node, ast.Import):
                yield from (alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                yield node.module.split(".")[0]
            for child in ast.iter_child_nodes(node):
                yield from imports(child)

        graph = {}
        for name, path in modules.items():
            dependencies = set(imports(ast.parse(path.read_text())))
            if name not in cli_names:
                self.assertFalse(dependencies & cli_names, name)
                self.assertNotIn("argparse", dependencies, name)
            graph[name] = dependencies & modules.keys()
        self.assertFalse(
            graph["audit_report"]
            & {"organization_run", "dedupe_run", "replacement_run"}
        )

        visited = set()

        def visit(name, active):
            self.assertNotIn(name, active, f"Circular imports: {active} -> {name}")
            if name in visited:
                return
            for dependency in graph[name]:
                visit(dependency, (*active, name))
            visited.add(name)

        for name in graph:
            visit(name, ())

    def test_duplicate_library_options_reject_invalid_input(self) -> None:
        for invalid in (
            {"workers": 0},
            {"inspection_timeout": 0},
            {"title_threshold": 1.1},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                dedupe_run.DedupeOptions(Path("papers"), **invalid)


def _paper_meta(*, doi: str, family: str) -> record_matching.PaperMeta:
    author = record_matching.Author(given="A", family=family)
    return record_matching.PaperMeta(
        title="Reliable Quantitative Imaging with Magnetic Resonance",
        year=2024,
        journal_abbrev="Magn Reson Med",
        first_author=author,
        authors=[author],
        corresponding_author=author,
        doi=doi,
        pmid=None,
        author_keywords=[],
        mesh_keywords=[],
    )


class SameFileMoveTests(unittest.TestCase):
    def test_parent_case_alias_preserves_the_only_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "others"
            parent.mkdir()
            source = parent / "scan.pdf"
            source.write_bytes(b"only copy")
            target = root / "Others" / source.name
            if not target.exists():
                self.skipTest("requires a case-insensitive temporary volume")
            collection_mutation._rename_path(source, target)
            self.assertEqual(target.read_bytes(), b"only copy")

    def test_distinct_hard_link_entry_can_be_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            destination = root / "other"
            destination.mkdir()
            target = destination / "target.pdf"
            source.write_bytes(b"shared bytes")
            try:
                target.hardlink_to(source)
            except OSError as exc:
                self.skipTest(f"hard links unavailable: {exc}")
            collection_mutation._rename_path(source, target)
            self.assertFalse(source.exists())
            self.assertEqual(target.read_bytes(), b"shared bytes")


class PdfThreadSafetyTests(unittest.TestCase):
    def test_unisolated_pdf_inspectors_are_rejected(self) -> None:
        with self.subTest(case="commit_inspection_rejects_threaded_default_inspector"):
            with self.assertRaisesRegex(ValueError, "PyMuPDF is not thread-safe"):
                dedupe_run.inspect_pdfs(
                    [Path("/tmp/ShaoX_commit.pdf")],
                    workers=4,
                    commit=True,
                    inspector=dedupe_run.inspect_pdf,
                )

        with self.subTest(case="commit_inspection_rejects_threaded_custom_inspector"):
            path = Path("/tmp/ShaoX_custom_commit.pdf")
            inspector = mock.Mock(
                return_value=dedupe_run.PdfIdentity(
                    path,
                    None,
                    "",
                    0,
                    None,
                )
            )

            with self.assertRaisesRegex(ValueError, "PyMuPDF is not thread-safe"):
                dedupe_run.inspect_pdfs(
                    [path],
                    workers=2,
                    commit=True,
                    inspector=inspector,
                )

            inspector.assert_not_called()

        with self.subTest(case="parallel_dry_run_rejects_unisolated_custom_inspector"):
            path = Path("/tmp/ShaoX_custom_dry_run.pdf")
            inspector = mock.Mock()

            with self.assertRaisesRegex(
                ValueError,
                "parallel inspection requires the process-isolated inspector",
            ):
                dedupe_run.inspect_pdfs(
                    [path],
                    workers=2,
                    commit=False,
                    inspector=inspector,
                )

            inspector.assert_not_called()


class TitleMatchSafetyTests(unittest.TestCase):
    def test_distinct_high_similarity_records_are_ambiguous(self) -> None:
        title = "Reliable Quantitative Imaging with Magnetic Resonance"
        records = [
            record_matching.ProviderRecord(
                "crossref",
                _paper_meta(doi="10.1000/one", family="Example"),
            ),
            record_matching.ProviderRecord(
                "datacite",
                _paper_meta(doi="10.1000/two", family="Different"),
            ),
        ]

        decision = record_matching.decide_title_match(title, None, records)

        self.assertEqual(decision.outcome, "rejected")
        self.assertIn("ambiguous", decision.reason.lower())


class DuplicateCacheSafetyTests(unittest.TestCase):
    def test_title_duplicate_evidence_cache_is_bounded(self) -> None:
        evidence_cache = dedupe_run._title_duplicate_evidence
        evidence_cache.cache_clear()
        self.addCleanup(evidence_cache.cache_clear)

        for index in range(8_193):
            evidence_cache(
                Path(f"/tmp/Example_Journal24_Paper_{index}.pdf"),
                f"Reliable Quantitative Imaging Study {index}",
            )

        cache_info = evidence_cache.cache_info()
        self.assertIsNotNone(cache_info.maxsize)
        self.assertLess(cache_info.currsize, 8_193)


class HttpFailureBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_configured_client_redirects_and_decoding_retries(self) -> None:
        with self.subTest(case="fetch_http_follows_redirect_with_configured_client"):
            def handler(request):
                if request.url.path == "/redirect":
                    return metadata_providers.httpx.Response(
                        301,
                        headers={"Location": "/record"},
                    )
                return metadata_providers.httpx.Response(
                    200,
                    json={"record": "resolved"},
                )

            transport = metadata_providers.httpx.MockTransport(handler)
            async with metadata_providers.httpx.AsyncClient(
                transport=transport,
                follow_redirects=True,
            ) as client:
                result = await metadata_providers.fetch_http(
                    client,
                    "https://example.test/redirect",
                    None,
                    False,
                    1,
                    None,
                    None,
                )

            self.assertEqual(result, {"record": "resolved"})

        with self.subTest(case="fetch_http_retries_response_decoding_error"):
            url = "https://example.test/record"
            decoding_error = metadata_providers.httpx.DecodingError(
                "invalid compressed response",
                request=metadata_providers.httpx.Request("GET", url),
            )
            response = mock.Mock(status_code=200)
            response.json.return_value = {"record": "recovered"}
            client = mock.Mock()
            client.get = mock.AsyncMock(
                side_effect=[decoding_error, response]
            )

            with mock.patch(
                "metadata_providers.asyncio.sleep",
                mock.AsyncMock(),
            ):
                result = await metadata_providers.fetch_http(
                    client,
                    url,
                    None,
                    False,
                    2,
                    None,
                    None,
                )

            self.assertEqual(result, {"record": "recovered"})
            self.assertEqual(client.get.await_count, 2)


    async def test_unexpected_transport_errors_propagate(self) -> None:
        with self.subTest(case="fetch_http_propagates_unexpected_json_error"):
            response = mock.Mock(status_code=200)
            response.json.side_effect = TypeError("parser contract changed")
            client = mock.Mock()
            client.get = mock.AsyncMock(return_value=response)

            with self.assertRaisesRegex(TypeError, "parser contract changed"):
                await metadata_providers.fetch_http(
                    client,
                    "https://example.test/record",
                    None,
                    False,
                    1,
                    None,
                    None,
                )

        with self.subTest(case="fetch_http_propagates_unexpected_retry_header_type"):
            response = mock.Mock(status_code=429)
            response.headers.get.return_value = object()
            client = mock.Mock()
            client.get = mock.AsyncMock(return_value=response)

            with self.assertRaises(TypeError):
                await metadata_providers.fetch_http(
                    client,
                    "https://example.test/record",
                    None,
                    False,
                    2,
                    None,
                    None,
                )


    def test_retry_after_parses_http_date_and_applies_cap(self) -> None:
        now = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)

        self.assertEqual(
            metadata_providers.retry_after_seconds(
                format_datetime(now + timedelta(seconds=5), usegmt=True),
                0.5,
                now=now,
            ),
            5.0,
        )
        self.assertEqual(
            metadata_providers.retry_after_seconds(
                format_datetime(now + timedelta(seconds=30), usegmt=True),
                0.5,
                now=now,
            ),
            10.0,
        )


class TransactionSeamTests(unittest.TestCase):
    def test_shared_transaction_registries(self) -> None:
        with self.subTest(case="all_discovery_paths_exclude_every_rewrite_artifact"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts = (
                    root / ".paper.pdf.renamepdf-rollback",
                    root / ".paper.pdf.repairing.pdf",
                    root / ".paper.pdf.installing_2.pdf",
                    root / ".paper.pdf.renamepdf-installed-marker",
                )
                for artifact in artifacts:
                    artifact.write_bytes(b"transaction artifact")
                    self.assertTrue(pdf_rewrite.is_rewrite_artifact(artifact))

                organization_paths = organization_run._discover_pdfs(
                    organization_run.RunOptions(root)
                )
                dedupe_paths = dedupe_run.discover_pdfs(root)
                self.assertFalse(set(artifacts) & set(organization_paths))
                self.assertFalse(set(artifacts) & set(dedupe_paths))

        with self.subTest(case="recovery_status_registry_drives_blocks_run"):
            self.assertFalse(
                pdf_rewrite.RECOVERY_BLOCKING_STATUSES
                & pdf_rewrite.RECOVERY_NONBLOCKING_STATUSES
            )
            for status in pdf_rewrite.RECOVERY_STATUSES:
                with self.subTest(status=status):
                    outcome = pdf_rewrite.PdfRecoveryOutcome(
                        Path("paper.pdf"),
                        status,
                        "test",
                    )
                    self.assertEqual(
                        outcome.blocks_run,
                        status in pdf_rewrite.RECOVERY_BLOCKING_STATUSES,
                    )
            with self.assertRaisesRegex(ValueError, "unknown PDF recovery status"):
                pdf_rewrite.PdfRecoveryOutcome(
                    Path("paper.pdf"),
                    "pdf_recovery_new_unregistered_status",
                    "test",
                )

        with self.subTest(case="pdf_blocking_status_registry_is_shared"):
            self.assertIs(
                organization_run.BLOCKING_PDF_STATUSES,
                pdf_rewrite.REWRITE_BLOCKING_STATUSES,
            )


class ReportDurabilityTests(unittest.TestCase):
    @unittest.skipUnless(
        sys.platform != "win32",
        "POSIX file modes are not enforced on Windows",
    )
    def test_first_csv_uses_intentional_owner_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.csv"

            report_io.write_csv_atomic(
                report,
                [{"name": "paper.pdf"}],
                fieldnames=("name",),
            )

            self.assertEqual(
                stat.S_IMODE(report.stat().st_mode),
                report_io.NEW_REPORT_MODE,
            )

    def test_atomic_report_write_preservation(self) -> None:
        with self.subTest(case="csv_temp_descriptor_uses_binary_mode"):
            with tempfile.TemporaryDirectory() as directory:
                report = Path(directory) / "report.csv"
                real_mkstemp = report_io.tempfile.mkstemp

                with mock.patch(
                    "report_io.tempfile.mkstemp",
                    wraps=real_mkstemp,
                ) as create_temp:
                    report_io.write_csv_atomic(
                        report,
                        [{"name": "paper.pdf"}],
                        fieldnames=("name",),
                    )

                self.assertIs(create_temp.call_args.kwargs["text"], False)
                self.assertEqual(report.read_bytes(), b"name\r\npaper.pdf\r\n")

        with self.subTest(case="interrupted_writes_preserve_all_existing_reports"):
            writers = (
                (
                    "rename",
                    "0_results.csv",
                    lambda path: RenamePDF._write_logs(
                        [{"source": "paper.pdf"}],
                        path.parent,
                    ),
                ),
                (
                    "dedupe",
                    "dedupe.csv",
                    lambda path: dedupe_run.write_report([], path),
                ),
                (
                    "replace",
                    "replace.csv",
                    lambda path: replacement_run.write_report(path, []),
                ),
            )
            for label, filename, write in writers:
                with (
                    self.subTest(label=label),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    report = root / filename
                    report.write_text("previous report\n", encoding="utf-8")
                    csv_writer = mock.Mock()
                    csv_writer.writerows.side_effect = RuntimeError("interrupted")

                    with (
                        mock.patch("report_io.csv.DictWriter", return_value=csv_writer),
                        self.assertRaisesRegex(RuntimeError, "interrupted"),
                    ):
                        write(report)

                    self.assertEqual(
                        report.read_text(encoding="utf-8"),
                        "previous report\n",
                    )
                    self.assertEqual(list(root.iterdir()), [report])


class YearPolicyTests(unittest.TestCase):
    def test_shared_year_interpretation_and_tie_breaking(self) -> None:
        with self.subTest(case="two_digit_year_policy_is_shared_by_all_tools"):
            cases = {
                "00": 2000,
                "29": 2029,
                "30": 1930,
                "69": 1969,
                "70": 1970,
                "99": 1999,
            }
            for token, expected in cases.items():
                with self.subTest(token=token):
                    self.assertEqual(
                        filename_policy.resolve_two_digit_year(int(token)),
                        expected,
                    )
                    self.assertEqual(
                        dedupe_run._filename_year(
                            Path(f"Example_Journal{token}_Paper.pdf")
                        ),
                        expected,
                    )
                    self.assertEqual(
                        replacement_run.source_year(
                            Path(f"Example_ConferencePaper{token}_Paper.pdf")
                        ),
                        expected,
                    )

        with self.subTest(case="filename_tie_break_prefers_present_four_digit_year"):
            meta = _paper_meta(doi="10.1000/year", family="Example")

            self.assertTrue(
                record_matching._filename_year_matches(
                    meta,
                    "Example_2024_Volume23_Imaging.pdf",
                )
            )
            self.assertFalse(
                record_matching._filename_year_matches(
                    meta,
                    "Example_2023_Volume24_Imaging.pdf",
                )
            )
            self.assertTrue(
                record_matching._filename_year_matches(
                    meta,
                    "Example_Journal24_Imaging.pdf",
                )
            )


if __name__ == "__main__":
    unittest.main()
