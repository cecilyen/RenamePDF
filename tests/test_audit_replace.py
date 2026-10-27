"""Focused tests consolidated by validation surface."""

import asyncio
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pymupdf

import AuditPDF
import replacement_run
import ReplacePDF as replace_cli
import document_classification
import audit_report
import collection_mutation
import filename_policy
import metadata_providers
import pdf_evidence
import pdf_rewrite
import publication_status
import organization_run
import record_matching as decision
import record_matching
import report_io


# Audit labels and report boundaries

class AuditLabelTests(unittest.TestCase):
    def test_preprint_review_and_version_labels(self) -> None:
        with self.subTest(case="version_link_does_not_block_a_preprint_replacement"):
            formal = publication_status.assess_retractions(
                pubmed_xml='<PubmedArticleSet><CommentsCorrectionsList><CommentsCorrections RefType="UpdateIn"><PMID>200</PMID></CommentsCorrections></CommentsCorrectionsList></PubmedArticleSet>',
                pubmed_checked=True,
            )
            result = document_classification.classify_audit(
                path=Path("Author_bioRxiv24_Study.pdf"), status="crossref_resolved",
                doi="10.1101/example", title="A long scientific preprint study title",
                retraction_status=formal.status,
            )
            self.assertEqual(result.replacement_action, "find_published_version")
            self.assertTrue(result.replacement_eligible)

        with self.subTest(case="preprint_labels_do_not_infer_absent_peer_review"):
            ordinary = document_classification.classify_audit(
                path=Path("AuthorA_bioRxiv24_Long_Study_Title.pdf"),
                status="crossref_title_resolved",
                title="A Long Study Title for Reliable Classification",
            )
            reviewed = document_classification.classify_audit(
                path=Path("AuthorA_bioRxiv24_Long_Study_Title.pdf"),
                status="crossref_title_resolved",
                title="Reviewed preprint: a long study title",
            )

            self.assertEqual(ordinary.review_status, "explicitly_preliminary")
            self.assertEqual(reviewed.review_status, "reviewed_preprint")
            self.assertNotIn("not_peer_reviewed", ordinary.labels)


    def test_condition_and_formal_update_eligibility(self) -> None:
        with self.subTest(case="formal_update_requires_review_and_is_not_eligible"):
            result = document_classification.classify_audit(
                path=Path("AuthorA_Journal24_Long_Study_Title.pdf"),
                status="pubmed_resolved",
                doi="10.1000/example",
                title="A Long Study Title Requiring Formal Review",
                journal="J Test",
                retraction_status="correction_or_partial_retraction",
            )

            self.assertEqual(result.replacement_action, "review_formal_update")
            self.assertFalse(result.replacement_eligible)

        with self.subTest(case="image_only_accepted_doi_can_request_the_same_work"):
            result = document_classification.classify_audit(
                path=Path("AuthorA_Journal24_Long_Study_Title.pdf"),
                status="pubmed_resolved",
                doi="10.1000/example",
                title="A Long Study Title with an Accepted Identity",
                journal="J Test",
                pdf_status="pdf_image_only",
                retraction_status="no_retraction_signal",
            )

            self.assertEqual(result.replacement_action, "find_exact_clean_copy")
            self.assertTrue(result.replacement_eligible)

        with self.subTest(case="normalization_precedes_preprint_replacement"):
            result = document_classification.classify_audit(
                path=Path("AuthorA_bioRxiv24_Long_Study_Title.pdf"),
                status="crossref_resolved",
                doi="10.1101/example",
                title="A Long Study Title with Repairable Evidence",
                pdf_status="pdf_repair_needed",
                retraction_status="no_retraction_signal",
            )

            self.assertEqual(result.replacement_action, "normalize_pdf")
            self.assertFalse(result.replacement_eligible)


class AuditReportBoundaryTests(unittest.TestCase):
    def test_inspection_timeout_cli_contract(self) -> None:
        with self.subTest(case="audit_cli_timeout_is_positive_finite_and_has_a_default"):
            self.assertEqual(AuditPDF._parse_args(["papers"]).inspection_timeout, 120.0)
            self.assertEqual(AuditPDF._parse_args(["papers", "--inspection-timeout", "900"]).inspection_timeout, 900.0)
            for value in ("0", "-1", "nan", "inf", "-inf", "not-a-number"):
                with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        AuditPDF._parse_args(["papers", f"--inspection-timeout={value}"])
                    self.assertEqual(error.exception.code, 2)

        with self.subTest(case="audit_cli_passes_timeout_to_workflow"):
            with tempfile.TemporaryDirectory() as directory:
                with (
                    mock.patch.object(AuditPDF, "_open_identity_cache", return_value=None),
                    mock.patch.object(organization_run, "run_audit", new=mock.AsyncMock(return_value=[])) as run,
                ):
                    self.assertEqual(AuditPDF.main([directory, "--inspection-timeout", "900"]), 0)
                self.assertEqual(run.call_args.args[0].inspection_timeout, 900.0)


    def test_fresh_source_controls_eligibility(self) -> None:
        with self.subTest(case="fresh_condition_and_stage_control_replacement_eligibility"):
            cases = (
                ("journal.pdf", "find_exact_clean_copy", "journal_article", "ok", False),
                ("journal.pdf", "find_exact_clean_copy", "journal_article", "pdf_image_only", True),
                ("journal.pdf", "find_published_version", "preprint", "ok", False),
                ("Author_bioRxiv24_Study.pdf", "find_published_version", "preprint", "ok", True),
            )
            for name, action, stage, pdf_status, eligible in cases:
                with self.subTest(action=action, pdf_status=pdf_status, name=name), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory).resolve()
                    source = root / name
                    source.write_bytes(b"private synthetic fixture")
                    self._write_report(root / audit_report.DEFAULT_REPORT_NAME, source, action=action, stage=stage, doi="10.1000/exact")
                    evidence = pdf_evidence.PdfEvidence([], "A Long Audited Scientific Study Title", None, None, pdf_status=pdf_status)
                    with (
                        mock.patch.object(replacement_run, "inspect_candidate_sources", return_value={source: evidence}),
                        mock.patch.object(replacement_run, "ApiClient"),
                        mock.patch.object(replacement_run, "process_candidate", return_value={"status": "fixture_ok", "path": str(source), "archive_path": ""}) as process,
                    ):
                        self.assertEqual(replacement_run.run(replacement_run.ReplacementOptions(root, "user@example.test", commit=True)), 0)
                    self.assertEqual(process.call_count, int(eligible))
                    self.assertEqual(source.read_bytes(), b"private synthetic fixture")
                    if not eligible:
                        _, rows = report_io.read_csv(root / "0_replace_results.csv")
                        self.assertEqual(rows[0]["status"], "replacement_not_eligible_fresh_evidence")

        with self.subTest(case="stale_row_blocks_before_network_client_creation"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pdf = root / "AuthorA_bioRxiv24_Study.pdf"
                pdf.write_bytes(b"%PDF-audited")
                report = root / audit_report.DEFAULT_REPORT_NAME
                self._write_report(report, pdf)
                pdf.write_bytes(b"%PDF-changed-and-longer")

                with mock.patch.object(replacement_run, "ApiClient") as api_client:
                    result = replace_cli.main(
                        [str(root), "--email", "user@example.test"]
                    )

                self.assertEqual(result, 1)
                api_client.assert_not_called()


    @staticmethod
    def _write_report(
        report: Path,
        pdf: Path,
        *,
        action: str = "find_published_version",
        stage: str = "preprint",
        doi: str = "10.1101/example",
    ) -> None:
        metadata = pdf.stat()
        row = dict.fromkeys(audit_report.AUDIT_REPORT_FIELDS, "")
        row.update(
            {
                "audit_schema_version": audit_report.AUDIT_SCHEMA_VERSION,
                "path": str(pdf.resolve()),
                "relative_path": pdf.name,
                "file_size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
                "identity_status": "accepted",
                "doi": doi,
                "title": "A Long Audited Scientific Study Title",
                "publication_stage": stage,
                "replacement_action": action,
                "replacement_eligible": "yes",
            }
        )
        report_io.write_csv_atomic(
            report,
            [row],
            fieldnames=audit_report.AUDIT_REPORT_FIELDS,
            extrasaction="ignore",
        )

    def test_current_report_schema_and_selection(self) -> None:
        with self.subTest(case="loader_selects_current_eligible_row"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pdf = root / "AuthorA_bioRxiv24_Study.pdf"
                pdf.write_bytes(b"%PDF-audited")
                report = root / audit_report.DEFAULT_REPORT_NAME
                self._write_report(report, pdf)

                selected = replacement_run.load_audit_candidates(root, report)

            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].path, pdf.resolve())
            self.assertEqual(
                selected[0].replacement_action,
                "find_published_version",
            )

        with self.subTest(case="report_row_records_versioned_labels_and_file_state"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                pdf = root / "AuthorA_bioRxiv24_Study.pdf"
                pdf.write_bytes(b"%PDF-audited")
                outcome = organization_run.OrganizationOutcome(
                    source=pdf,
                    status="crossref_resolved",
                    doi="10.1101/example",
                    title="A Long Audited Scientific Study Title",
                    retraction_status="no_retraction_signal",
                    commentary_status="no_commentary_signal",
                )

                row = audit_report.report_row(root, outcome)

            self.assertEqual(
                row["audit_schema_version"],
                audit_report.AUDIT_SCHEMA_VERSION,
            )
            self.assertEqual(row["replacement_action"], "find_published_version")
            self.assertEqual(row["replacement_eligible"], "yes")
            self.assertEqual(row["file_size"], len(b"%PDF-audited"))


    def test_exact_copy_resolution_keeps_the_audited_doi(self) -> None:
        candidate = replacement_run.Candidate(
            path=Path("AuthorA_Journal24_Study.pdf"),
            kind="journal_article",
            title="A Long Audited Scientific Study Title",
            source_doi="10.1000/exact",
            replacement_action="find_exact_clean_copy",
        )
        message = {
            "DOI": "10.1000/exact",
            "title": ["A Long Audited Scientific Study Title"],
            "type": "journal-article",
            "container-title": ["Journal of Tests"],
            "issued": {"date-parts": [[2025]]},
            "author": [{"family": "Author"}],
        }

        with mock.patch.object(
            replacement_run,
            "crossref_message",
            return_value=message,
        ):
            resolution = replacement_run.resolve_exact_copy(object(), candidate)

        self.assertTrue(resolution.decision.accepted)
        self.assertEqual(resolution.decision.record.doi, "10.1000/exact")
        self.assertEqual(resolution.decision.method, "exact_doi_copy")

    def test_failed_source_audit_is_read_only(self) -> None:
        with self.subTest(case="audit_discovery_includes_failed_but_not_holding_copies"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                active = root / "active.pdf"
                failed = root / "Failed" / "Preprints" / "failed.pdf"
                duplicated = root / "Duplicated" / "duplicate.pdf"
                failed.parent.mkdir(parents=True)
                duplicated.parent.mkdir()
                active.touch()
                failed.touch()
                duplicated.touch()

                found = organization_run._discover_pdfs(
                    organization_run.RunOptions(root, include_failed=True)
                )

            self.assertEqual(set(found), {active, failed})

        with self.subTest(case="include_failed_cannot_authorize_collection_mutation"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaisesRegex(
                    organization_run.OrganizationRunError,
                    "include_failed is read-only",
                ):
                    asyncio.run(
                        organization_run.run(
                            organization_run.RunOptions(
                                root,
                                commit=True,
                                include_failed=True,
                            )
                        )
                    )


    def test_audit_cli_uses_the_shared_identity_cache_name(self) -> None:
        self.assertEqual(
            AuditPDF.IDENTITY_CACHE_FILENAME,
            replacement_run.IDENTITY_CACHE_FILENAME,
        )

    def test_cache_publication_follows_successful_report(self) -> None:
        with self.subTest(case="audit_publishes_cache_only_after_atomic_report"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cache = mock.Mock()
                events: list[str] = []
                with (
                    mock.patch.object(
                        AuditPDF,
                        "_open_identity_cache",
                        return_value=cache,
                    ),
                    mock.patch.object(
                        AuditPDF.organization_run,
                        "run_audit",
                        new=mock.AsyncMock(return_value=[]),
                    ),
                    mock.patch.object(
                        AuditPDF.report_io,
                        "write_csv_atomic",
                        side_effect=lambda *_args, **_kwargs: events.append(
                            "report"
                        ),
                    ),
                    mock.patch.object(
                        AuditPDF,
                        "_publish_identity_cache",
                        side_effect=lambda *_args: events.append("snapshot"),
                    ),
                ):
                    result = AuditPDF.main([str(root)])

            self.assertEqual(result, 0)
            self.assertEqual(events, ["report", "snapshot"])
            cache.close.assert_called_once_with()

        with self.subTest(case="audit_report_failure_does_not_publish_cache"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cache = mock.Mock()
                with (
                    mock.patch.object(
                        AuditPDF,
                        "_open_identity_cache",
                        return_value=cache,
                    ),
                    mock.patch.object(
                        AuditPDF.organization_run,
                        "run_audit",
                        new=mock.AsyncMock(return_value=[]),
                    ),
                    mock.patch.object(
                        AuditPDF.report_io,
                        "write_csv_atomic",
                        side_effect=OSError("report unavailable"),
                    ),
                    mock.patch.object(
                        AuditPDF,
                        "_publish_identity_cache",
                    ) as publish,
                ):
                    result = AuditPDF.main([str(root)])

            self.assertEqual(result, 1)
            publish.assert_not_called()


# Published-replacement identity decisions

TITLE = "Quantitative magnetic resonance imaging of brain tissue"


def source(
    *,
    title: str = TITLE,
    doi: str | None = "10.1101/2024.01.01.123456",
    author: str | None = "smith",
    year: int | None = 2024,
) -> decision.PreliminaryWork:
    return decision.PreliminaryWork(
        kind="preprint",
        title=title,
        doi=doi,
        first_author=author,
        year=year,
    )


def record(
    doi: str = "10.1000/published",
    *,
    title: str = TITLE,
    year: int | None = 2025,
    record_type: str = "journal-article",
    journal: str = "Magnetic Resonance in Medicine",
    authors: tuple[str, ...] = ("smith", "jones"),
) -> decision.PublishedRecord:
    return decision.PublishedRecord(
        doi=doi,
        title=title,
        year=year,
        record_type=record_type,
        journal=journal,
        authors=authors,
    )


class DirectRelationDecisionTests(unittest.TestCase):
    def test_direct_relation_identity_boundary(self) -> None:
        with self.subTest(case="verified_relation_accepts_a_retitled_record_with_same_author"):
            result = decision.decide_published_relation(
                source(),
                record(title="A substantially revised title"),
                source_identity_verified=True,
            )

            self.assertTrue(result.accepted)
            self.assertTrue(result.author_match)
            self.assertEqual(result.method, "direct_publication_relation")

        with self.subTest(case="relation_rejects_unrelated_title_and_authors"):
            result = decision.decide_published_relation(
                source(),
                record(
                    title="Unrelated spectroscopy calibration experiment",
                    authors=("lee", "patel"),
                ),
                source_identity_verified=True,
            )

            self.assertEqual(result.outcome, "rejected")
            self.assertIn("lacks source-author corroboration", result.reason)

        with self.subTest(case="relation_rejects_source_doi_as_replacement"):
            result = decision.decide_published_relation(
                source(),
                record(doi="10.1101/2024.01.01.123456"),
                source_identity_verified=True,
            )

            self.assertEqual(result.outcome, "rejected")
            self.assertIn("identical to the source DOI", result.reason)

        with self.subTest(case="relation_requires_a_complete_journal_record"):
            result = decision.decide_published_relation(
                source(),
                record(journal=""),
                source_identity_verified=True,
            )

            self.assertEqual(result.outcome, "rejected")
            self.assertIn("no journal venue", result.reason)


class TitleSearchDecisionTests(unittest.TestCase):
    def test_corroborated_title_search(self) -> None:
        with self.subTest(case="author_corroboration_allows_close_title_match"):
            with mock.patch.object(
                decision,
                "title_similarity",
                return_value=0.86,
            ):
                result = decision.decide_published_title(source(), (record(),))

            self.assertTrue(result.accepted)
            self.assertTrue(result.author_match)

        with self.subTest(case="strong_title_can_match_without_filename_author"):
            with mock.patch.object(
                decision,
                "title_similarity",
                return_value=0.96,
            ):
                result = decision.decide_published_title(
                    source(author=None),
                    (record(authors=("lee",)),),
                )

            self.assertTrue(result.accepted)
            self.assertFalse(result.author_match)

        with self.subTest(case="duplicate_search_rows_for_one_doi_are_not_ambiguous"):
            result = decision.decide_published_title(
                source(),
                (
                    record("10.1000/same"),
                    record("10.1000/SAME"),
                ),
            )

            self.assertTrue(result.accepted)


    def test_title_search_rejection_boundary(self) -> None:
        with self.subTest(case="close_distinct_records_are_ambiguous"):
            scores = {
                "Candidate A": 0.97,
                "Candidate B": 0.96,
            }
            with mock.patch.object(
                decision,
                "title_similarity",
                side_effect=lambda _source, title: scores[title],
            ):
                result = decision.decide_published_title(
                    source(),
                    (
                        record("10.1000/a", title="Candidate A"),
                        record("10.1000/b", title="Candidate B"),
                    ),
                )

            self.assertEqual(result.outcome, "ambiguous")
            self.assertIn("score margin", result.reason)

        with self.subTest(case="part_number_must_be_preserved"):
            result = decision.decide_published_title(
                source(title="Consensus recommendations Part 2"),
                (
                    record(
                        title="Consensus recommendations Part 1",
                    ),
                ),
            )

            self.assertEqual(result.outcome, "rejected")
            self.assertIn("does not preserve source part 2", result.reason)

        with self.subTest(case="implausibly_old_journal_record_is_rejected"):
            result = decision.decide_published_title(
                source(year=2024),
                (record(year=2020),),
            )

            self.assertEqual(result.outcome, "rejected")
            self.assertIn("predates source year", result.reason)

        with self.subTest(case="records_without_authors_are_rejected"):
            result = decision.decide_published_title(
                source(),
                (record(authors=()),),
            )

            self.assertEqual(result.outcome, "rejected")
            self.assertIn("no usable authors", result.reason)


# Audit-selected replacement workflow

class StreamingResponse:
    status_code = 200

    def __init__(
        self,
        payload: bytes,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    @staticmethod
    def raise_for_status() -> None:
        return None

    def iter_bytes(self, *, chunk_size: int):
        for offset in range(0, len(self.payload), chunk_size):
            yield self.payload[offset : offset + chunk_size]


class StreamingHttpClient:
    def __init__(
        self,
        payload: bytes,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.headers = headers

    def stream(self, *_args, **_kwargs) -> StreamingResponse:
        return StreamingResponse(self.payload, headers=self.headers)


class ArchiveLayoutTests(unittest.TestCase):
    def test_original_is_archived_directly_under_superseded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "AuthorA_bioRxiv24_Result.pdf"
            source.write_bytes(b"preprint")
            candidate = replacement_run.Candidate(
                path=source,
                kind="preprint",
                title="Result",
                source_doi="10.1101/example",
            )

            archived = replacement_run.archive_original(root, candidate)

            self.assertEqual(archived, root / "Superseded" / source.name)
            self.assertEqual(archived.read_bytes(), b"preprint")
            self.assertFalse(source.exists())


class CandidateTests(unittest.TestCase):
    def test_report_row_contains_the_complete_ordered_schema(self) -> None:
        row = replacement_run.make_report_row(
            replacement_run.Candidate(
                Path("paper.pdf"),
                "preprint",
                "Example paper",
                "10.1101/example",
            )
        )

        self.assertEqual(tuple(row), replacement_run.REPORT_FIELDS)

    def test_preflight_blocks_network(self) -> None:
        with self.subTest(case="main_halts_before_network_when_recovery_is_needed"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rollback = root / ".paper.pdf.renamepdf-rollback"
                rollback.write_bytes(b"original PDF")

                with mock.patch.object(replacement_run, "ApiClient") as client:
                    exit_code = replace_cli.main(
                        [str(root), "--email", "test@example.test"]
                    )

                self.assertEqual(exit_code, 2)
                client.assert_not_called()

        with self.subTest(case="missing_audit_report_blocks_before_network"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)

                with mock.patch.object(replacement_run, "ApiClient") as client:
                    exit_code = replace_cli.main(
                        [str(root), "--email", "test@example.test"]
                    )

                self.assertEqual(exit_code, 1)
                client.assert_not_called()


    def test_source_filename_and_preprint_evidence(self) -> None:
        with self.subTest(case="filename_author_and_year_are_parsed"):
            path = Path("NiuY_Humphreys_bioRxiv24_Longitudinal_Pregnancy.pdf")
            self.assertEqual(replacement_run.expected_first_author(path), "niu")
            self.assertEqual(replacement_run.source_year(path), 2024)
            self.assertEqual(
                replacement_run.expected_first_author(Path("SchillingKG_Jelescu_arXiv22_ExVivo.pdf")),
                "schilling",
            )
            self.assertEqual(
                record_matching.title_part_number(
                    "Guidelines: Part 2 - Ex vivo"
                ),
                2,
            )

        with self.subTest(case="citation_preprint_is_not_treated_as_source_record"):
            candidate = replacement_run.Candidate(
                Path("SchillingK_Jelescu_arXiv22_ExVivo.pdf"),
                "preprint",
                "Recommendations from the ISMRM diffusion study group for ex vivo imaging",
                "10.1101/example",
            )
            self.assertFalse(
                replacement_run._source_title_matches(
                    candidate,
                    "Tissue optimization strategies for high-quality ex vivo diffusion imaging",
                    "Barrett; DellAcqua",
                )
            )


class OpenAccessLocationTests(unittest.TestCase):
    def test_version_of_record_locations(self) -> None:
        with self.subTest(case="crossref_cc_by_version_of_record_pdf_is_accepted"):
            version, url = replacement_run.crossref_licensed_pdf(
                {
                    "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
                    "link": [
                        {
                            "URL": "https://publisher.example/article.pdf",
                            "content-type": "application/pdf",
                            "content-version": "vor",
                        }
                    ],
                }
            )
            self.assertEqual(version, "publishedVersion")
            self.assertEqual(url, "https://publisher.example/article.pdf")

        with self.subTest(case="published_version_is_preferred"):
            class Client:
                def get_json(self, _url, _params):
                    return {
                        "oa_locations": [
                            {
                                "version": "acceptedVersion",
                                "host_type": "repository",
                                "url_for_pdf": "https://example.test/accepted.pdf",
                            },
                            {
                                "version": "publishedVersion",
                                "host_type": "publisher",
                                "url_for_pdf": "https://example.test/published.pdf",
                            },
                            {
                                "version": "submittedVersion",
                                "host_type": "publisher",
                                "url_for_pdf": "https://example.test/preprint.pdf",
                            },
                        ]
                    }

                email = "test@example.test"

            version, url = replacement_run.oa_location(Client(), "10.1/example")
            self.assertEqual(version, "publishedVersion")
            self.assertEqual(url, "https://example.test/published.pdf")

        with self.subTest(case="crossref_elife_pdf_uses_deposited_vor_link"):
            version, url = replacement_run.crossref_licensed_pdf(
                {
                    "DOI": "10.7554/eLife.97125.3",
                    "license": [
                        {"URL": "https://creativecommons.org/publicdomain/zero/1.0/"}
                    ],
                    "link": [
                        {
                            "URL": "https://cdn.elifesciences.org/articles/97125/elife-97125-v1.pdf",
                            "content-type": "application/pdf",
                            "content-version": "vor",
                        }
                    ],
                }
            )
            self.assertEqual(version, "publishedVersion")
            self.assertEqual(
                url,
                "https://cdn.elifesciences.org/articles/97125/elife-97125-v1.pdf",
            )


class PdfIdentityEvidenceTests(unittest.TestCase):
    def test_elife_canonical_version_boundary(self) -> None:
        with self.subTest(case="elife_vor_accepts_canonical_doi_with_vor_timeline"):
            matched, conflicting = record_matching.published_pdf_doi_evidence(
                "10.7554/eLife.97125.3",
                ["10.7554/eLife.97125", "10.7554/eLife.97125.2"],
                ["Reviewed preprint revised\nVersion of Record published"],
            )
            self.assertTrue(matched)
            self.assertFalse(conflicting)

        with self.subTest(case="elife_reviewed_preprint_cannot_pass_as_vor"):
            matched, conflicting = record_matching.published_pdf_doi_evidence(
                "10.7554/eLife.97125.3",
                ["10.7554/eLife.97125.2"],
                ["Reviewed Preprint"],
            )
            self.assertFalse(matched)
            self.assertTrue(conflicting)

        with self.subTest(case="elife_version_cannot_pass_as_canonical_by_title"):
            matched, conflicting = record_matching.published_pdf_doi_evidence(
                "10.7554/eLife.97125",
                ["10.7554/eLife.97125.2"],
                ["Reviewed Preprint"],
            )
            self.assertFalse(matched)
            self.assertTrue(conflicting)


class ExistingReplacementTests(unittest.TestCase):
    TITLE = "Validated Published Magnetic Resonance Study"
    DOI = "10.1000/existing-published"

    def _fixture(self, root: Path):
        source = root / "preprint.pdf"
        source.write_bytes(b"preserved synthetic preliminary PDF")
        candidate = replacement_run.Candidate(source, "preprint", self.TITLE, "10.1101/example")
        message = {
            "DOI": self.DOI, "title": [self.TITLE], "type": "journal-article",
            "issued": {"date-parts": [[2025]]}, "container-title": ["Example Journal"],
            "author": [{"given": "A", "family": "Example"}],
        }
        record = replacement_run._crossref_record(message)
        resolution = replacement_run.PublishedResolution(
            record_matching.ReplacementDecision("accepted", record, 0.99, "test", "accepted fixture"),
            message,
        )
        target = root / filename_policy.make_filename(metadata_providers.parse_crossref_message(message))
        return candidate, resolution, target

    def _pdf(self, path, doi=None, title=None):
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), f"{title or self.TITLE}\nhttps://doi.org/{doi or self.DOI}")
        document.set_metadata({"title": title or self.TITLE, "subject": f"doi:{doi or self.DOI}"})
        document.save(path)
        document.close()

    def _run(self, root, candidate, resolution, commit, cache=None):
        with (
            mock.patch.object(replacement_run, "resolve_published", return_value=resolution),
            mock.patch.object(replacement_run, "oa_locations", return_value=[]),
        ):
            return replacement_run.process_candidate(root, object(), candidate, commit, cache)

    def test_matching_target_is_checked_in_dry_run_and_commit(self) -> None:
        for commit in (False, True):
            with self.subTest(commit=commit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                candidate, resolution, target = self._fixture(root)
                self._pdf(target)
                original = candidate.path.read_bytes()
                target_bytes = target.read_bytes()
                cache = mock.Mock(usable=True)
                row = self._run(root, candidate, resolution, commit, cache)
                self.assertEqual(row["status"], "published_copy_already_present", row["note"])
                self.assertEqual(target.read_bytes(), target_bytes)
                self.assertEqual(candidate.path.exists(), not commit)
                cache.get.assert_not_called()
                if commit:
                    self.assertEqual(Path(row["archive_path"]).read_bytes(), original)
                    self.assertEqual(cache.put.call_args.args[0], target)
                    self.assertEqual(cache.put.call_args.args[1]["dois"][0]["doi"], self.DOI)
                else:
                    self.assertFalse((root / "Superseded").exists())
                    cache.put.assert_not_called()

    def test_unsafe_targets_do_not_authorize_archiving(self) -> None:
        with self.subTest(case="unsafe_collision_preserves_both_paths"):
            for kind in ("wrong_doi", "wrong_title", "damaged", "symlink", "directory"):
                with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    candidate, resolution, target = self._fixture(root)
                    if kind == "wrong_doi":
                        self._pdf(target, doi="10.1000/unrelated")
                    elif kind == "wrong_title":
                        self._pdf(target, title="Agricultural policies regulating irrigation across regional farming districts")
                    elif kind == "damaged":
                        target.write_bytes(b"not a PDF")
                    elif kind == "symlink":
                        target.symlink_to(candidate.path)
                    else:
                        target.mkdir()
                    original = candidate.path.read_bytes()
                    cache = mock.Mock(usable=True)
                    row = self._run(root, candidate, resolution, True, cache)
                    self.assertEqual(row["status"], "existing_target_requires_review", row["note"])
                    self.assertEqual(candidate.path.read_bytes(), original)
                    self.assertTrue(target.exists())
                    self.assertFalse((root / "Superseded").exists())
                    cache.put.assert_not_called()

        with self.subTest(case="citations_stage_and_required_normalization_cannot_authorize_archiving"):
            cases = (
                pdf_evidence.PdfEvidence(
                    [pdf_evidence.DoiCandidate(self.DOI, "raw-bytes", 30)], None, None, None,
                    text_samples=(self.TITLE,),
                ),
                pdf_evidence.PdfEvidence([], self.TITLE, None, None, needs_decryption=True),
                pdf_evidence.PdfEvidence([], self.TITLE, None, None, pdf_status="pdf_unrepairable"),
            )
            for evidence in cases:
                with self.subTest(evidence=evidence):
                    self.assertIsNotNone(record_matching.replacement_pdf_rejection(evidence, self.DOI, self.TITLE))
            evidence = pdf_evidence.PdfEvidence(
                [pdf_evidence.DoiCandidate("10.7554/eLife.99999.2", "metadata:doi", 120)],
                self.TITLE, None, None,
            )
            self.assertIn("version", record_matching.replacement_pdf_rejection(evidence, "10.7554/eLife.99999.3", self.TITLE))
            evidence = pdf_evidence.PdfEvidence(
                [pdf_evidence.DoiCandidate(self.DOI, "metadata:doi", 120)],
                None, None, None, pdf_status="pdf_image_only",
            )
            self.assertIsNotNone(record_matching.replacement_pdf_rejection(evidence, self.DOI, self.TITLE, require_text=True))


    def test_target_change_during_archiving_restores_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, resolution, target = self._fixture(root)
            self._pdf(target)
            original = candidate.path.read_bytes()
            move = collection_mutation.move_to_unique
            def changed_target(*args, **kwargs):
                archive = move(*args, **kwargs)
                target.write_bytes(b"concurrent target")
                return archive
            with mock.patch.object(collection_mutation, "move_to_unique", side_effect=changed_target):
                row = self._run(root, candidate, resolution, True)
            self.assertEqual(row["status"], "existing_target_requires_review")
            self.assertEqual(candidate.path.read_bytes(), original)
            self.assertEqual(target.read_bytes(), b"concurrent target")
            self.assertFalse(any((root / "Superseded").iterdir()))

class DownloadedPdfValidationTests(unittest.TestCase):
    def test_download_normalization_and_blocking(self) -> None:
        with self.subTest(case="download_is_normalized_and_identity_checked_with_pymupdf"):
            title = "Validated Published Magnetic Resonance Study"
            doi = "10.1000/validated-download"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text(
                (72, 72),
                f"{title}\nhttps://doi.org/{doi}",
            )
            document.set_metadata({"title": title})
            content = document.tobytes()
            cover = document.new_page(pno=0)
            cover.insert_text(
                (72, 72),
                "Citation: Journal of Reliable Imaging 12, 34 (2025)\n"
                "View online: https://doi.org/10.1000/wrapper-only\n"
                "View Table of Contents: https://example.test/?ver=pdfcov\n"
                "Published by the AIP Publishing",
            )
            covered_content = document.tobytes()
            document.close()

            class Client:
                def __init__(self, pdf_content):
                    self.client = StreamingHttpClient(pdf_content)

            candidate = replacement_run.Candidate(
                Path("Example_bioRxiv24_Validated.pdf"),
                "preprint",
                title,
                "10.1101/example",
            )
            match = replacement_run.PublishedMatch(
                doi,
                {
                    "DOI": doi,
                    "title": [title],
                    "container-title": ["Magnetic Resonance"],
                    "issued": {"date-parts": [[2025]]},
                    "author": [{"family": "Example"}],
                },
                0.99,
                "test",
            )

            for covered, pdf_content in ((False, content), (True, covered_content)):
                with self.subTest(covered=covered), tempfile.TemporaryDirectory() as directory:
                    normalized, evidence = replacement_run._download_validated_pdf_evidence(
                        Client(pdf_content),
                        candidate,
                        match,
                        "https://example.test/article.pdf",
                        Path(directory),
                    )

                    self.assertEqual(normalized.name, "normalized.pdf")
                    self.assertEqual((Path(directory) / "download.pdf").read_bytes(), pdf_content)
                    self.assertFalse(evidence.needs_rewrite)
                    self.assertEqual(evidence.dois[0].doi, doi)
                    self.assertNotIn("10.1000/wrapper-only", [item.doi for item in evidence.dois])
                    with pymupdf.open(normalized) as checked:
                        self.assertFalse(checked.is_repaired)
                        self.assertFalse(checked.needs_pass)
                        self.assertEqual(len(checked), 1)
                        self.assertIn(title, checked[0].get_text())

        with self.subTest(case="blocked_download_is_not_rewritten_or_accepted"):
            candidate = replacement_run.Candidate(Path("original.pdf"), "preprint", "Example", "10.1000/old")
            match = replacement_run.PublishedMatch("10.1000/new", {"title": ["Example"]}, 1.0, "test")
            for status in pdf_rewrite.REWRITE_BLOCKING_STATUSES:
                with (
                    self.subTest(status=status),
                    tempfile.TemporaryDirectory() as directory,
                    mock.patch.object(replacement_run, "download_pdf_to_path"),
                    mock.patch.object(pdf_evidence, "inspect_pdf", return_value=pdf_evidence.PdfEvidence(
                        [], None, None, None, pdf_status=status,
                    )),
                    mock.patch.object(pdf_rewrite, "write_validated_copy") as rewrite,
                ):
                    with self.assertRaisesRegex(RuntimeError, status):
                        replacement_run.download_validated_pdf(
                            object(), candidate, match, "https://example.test/article.pdf", Path(directory),
                        )
                    rewrite.assert_not_called()


    def test_streamed_download_size_limits(self) -> None:
        with self.subTest(case="streamed_download_rejects_declared_oversize_without_artifact"):
            class Client:
                client = StreamingHttpClient(
                    b"%PDF-small",
                    headers={"Content-Length": "11"},
                )

            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "download.pdf"

                with self.assertRaisesRegex(RuntimeError, "configured limit"):
                    replacement_run.download_pdf_to_path(
                        Client(),
                        "https://example.test/large.pdf",
                        destination,
                        referer="https://doi.org/10.1000/test",
                        max_bytes=10,
                    )

                self.assertFalse(destination.exists())

        with self.subTest(case="streamed_download_removes_owned_partial_when_limit_is_crossed"):
            class Client:
                client = StreamingHttpClient(b"%PDF-" + b"x" * 16)

            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "download.pdf"

                with self.assertRaisesRegex(RuntimeError, "during transfer"):
                    replacement_run.download_pdf_to_path(
                        Client(),
                        "https://example.test/large.pdf",
                        destination,
                        referer="https://doi.org/10.1000/test",
                        max_bytes=10,
                    )

                self.assertFalse(destination.exists())


class PublishedRecordCanonicalizationTests(unittest.TestCase):
    def test_canonical_record_normalization(self) -> None:
        with self.subTest(case="crossref_record_normalizes_collective_author_and_string_venue"):
            record = replacement_run._crossref_record(
                {
                    "DOI": "10.1000/example",
                    "title": ["Collective imaging recommendations"],
                    "type": "journal-article",
                    "issued": {"date-parts": [[2025]]},
                    "container-title": "Imaging Recommendations",
                    "author": [{"name": "International Imaging Consortium"}],
                }
            )

            self.assertEqual(record.journal, "Imaging Recommendations")
            self.assertEqual(record.authors, ("consortium",))

        with self.subTest(case="elife_canonical_record_promotes_exact_vor_doi"):
            canonical = {
                "DOI": "10.7554/eLife.97125",
                "type": "journal-article",
                "title": ["Cingulate cortex shapes early postnatal development"],
                "relation": {
                    "is-same-as": [
                        {"id-type": "doi", "id": "10.7554/eLife.97125.3"}
                    ]
                },
            }
            versioned = {
                "DOI": "10.7554/eLife.97125.3",
                "type": "journal-article",
                "title": ["Cingulate cortex shapes early postnatal development"],
            }
            with mock.patch.object(
                replacement_run,
                "crossref_message",
                side_effect=lambda _client, doi: (
                    versioned if doi.lower().endswith(".3") else canonical
                ),
            ):
                doi, message = replacement_run.canonical_published_record(
                    object(),
                    "10.7554/eLife.97125",
                )
            self.assertEqual(doi, "10.7554/eLife.97125.3")
            self.assertIs(message, versioned)


    def test_ambiguous_identity_stays_unresolved(self) -> None:
        with self.subTest(case="conflicting_direct_relations_are_not_title_searched"):
            candidate = replacement_run.Candidate(
                Path("SmithA_bioRxiv24_Imaging.pdf"),
                "preprint",
                "Quantitative magnetic resonance imaging",
                "10.1101/source",
            )
            source_message = {
                "title": ["Quantitative magnetic resonance imaging"],
                "author": [{"family": "Smith"}],
                "relation": {
                    "is-preprint-of": [
                        {"id-type": "doi", "id": "10.1000/crossref"}
                    ]
                },
            }

            with (
                mock.patch.object(
                    replacement_run,
                    "biorxiv_published_doi",
                    return_value="10.1000/biorxiv",
                ),
                mock.patch.object(
                    replacement_run,
                    "crossref_message",
                    return_value=source_message,
                ),
                mock.patch.object(
                    replacement_run,
                    "canonical_published_record",
                    side_effect=lambda _client, doi: (doi, {"DOI": doi}),
                ),
                mock.patch.object(
                    replacement_run,
                    "journal_title_match",
                ) as title_search,
            ):
                resolution = replacement_run.resolve_published(
                    object(),
                    candidate,
                )

            self.assertEqual(resolution.decision.outcome, "ambiguous")
            self.assertIn(
                "conflicting direct publication relations",
                resolution.decision.reason,
            )
            self.assertIs(resolution.source_message, source_message)
            title_search.assert_not_called()

        with self.subTest(case="ambiguous_decision_reason_is_preserved_in_report_row"):
            candidate = replacement_run.Candidate(
                Path("SmithA_bioRxiv24_Imaging.pdf"),
                "preprint",
                "Quantitative magnetic resonance imaging",
                "10.1101/source",
            )
            decision = record_matching.conflicting_publication_relations(
                ("10.1000/a", "10.1000/b")
            )
            resolution = replacement_run.PublishedResolution(decision, None)

            with mock.patch.object(
                replacement_run,
                "resolve_published",
                return_value=resolution,
            ):
                row = replacement_run.process_candidate(
                    Path("/tmp"),
                    object(),
                    candidate,
                    commit=False,
                )

            self.assertEqual(row["status"], "ambiguous_published_match")
            self.assertEqual(row["match_method"], "direct_publication_relation")
            self.assertEqual(row["note"], decision.reason)


    def test_missing_oa_copy_uses_provider_neutral_status(self) -> None:
        candidate = replacement_run.Candidate(
            Path("SmithA_bioRxiv24_Imaging.pdf"),
            "preprint",
            "Quantitative magnetic resonance imaging",
            "10.1101/source",
        )
        message = {
            "DOI": "10.1000/published",
            "title": ["Quantitative magnetic resonance imaging"],
            "type": "journal-article",
            "issued": {"date-parts": [[2025]]},
            "container-title": ["Magnetic Resonance in Medicine"],
            "author": [{"given": "A", "family": "Smith"}],
        }
        source = record_matching.PreliminaryWork(
            "preprint",
            candidate.title,
            candidate.source_doi,
            "smith",
            2024,
        )
        record = replacement_run._crossref_record(message)
        decision = record_matching.decide_published_relation(
            source,
            record,
            source_identity_verified=True,
        )
        resolution = replacement_run.PublishedResolution(decision, message)

        with (
            mock.patch.object(
                replacement_run,
                "resolve_published",
                return_value=resolution,
            ),
            mock.patch.object(
                replacement_run,
                "oa_locations",
                return_value=[],
            ),
        ):
            row = replacement_run.process_candidate(
                Path("/tmp"),
                object(),
                candidate,
                commit=False,
            )

        self.assertEqual(
            row["status"],
            "published_record_no_eligible_oa_pdf",
        )
        self.assertIn("publishedVersion or acceptedVersion", row["note"])

    def test_known_formal_update_blocks_candidate_before_oa_lookup(self) -> None:
        candidate = replacement_run.Candidate(
            Path("SmithA_bioRxiv24_Imaging.pdf"),
            "preprint",
            "Quantitative magnetic resonance imaging",
            "10.1101/source",
        )
        message = {
            "DOI": "10.1000/published",
            "title": ["Quantitative magnetic resonance imaging"],
            "type": "journal-article",
            "issued": {"date-parts": [[2025]]},
            "container-title": ["Magnetic Resonance in Medicine"],
            "author": [{"given": "A", "family": "Smith"}],
            "updated-by": [
                {
                    "DOI": "10.1000/correction",
                    "type": "correction",
                }
            ],
        }
        decision = record_matching.ReplacementDecision(
            outcome="accepted",
            record=replacement_run._crossref_record(message),
            confidence=0.99,
            method="direct_publication_relation",
            reason="test accepted relation",
        )
        resolution = replacement_run.PublishedResolution(decision, message)

        with (
            mock.patch.object(
                replacement_run,
                "resolve_published",
                return_value=resolution,
            ),
            mock.patch.object(replacement_run, "oa_locations") as locations,
        ):
            row = replacement_run.process_candidate(
                Path("/tmp"),
                object(),
                candidate,
                commit=False,
            )

        self.assertEqual(
            row["status"],
            "candidate_formal_update_requires_review",
        )
        self.assertEqual(
            row["candidate_retraction_status"],
            "correction_or_partial_retraction",
        )
        locations.assert_not_called()


if __name__ == "__main__":
    unittest.main()
