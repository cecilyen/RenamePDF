"""Focused tests consolidated by validation surface."""

import asyncio
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import pymupdf

import RenamePDF
import metadata_providers
import pdf_evidence
import pdf_process
import pdf_rewrite
import record_matching


# Evidence extraction and provider-query safeguards

class CacheDBTests(unittest.TestCase):
    def test_standard_library_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = metadata_providers.MetadataCache(
                Path(directory) / "cache.sqlite"
            )
            cache.open()
            try:
                cache.put_doi2pmid("10.1000/example", "12345")
                cache.put_crossref("10.1000/example", {"DOI": "10.1000/example"})
                self.assertEqual(cache.get_doi2pmid("10.1000/example"), "12345")
                self.assertEqual(
                    cache.get_crossref("10.1000/example"),
                    {"DOI": "10.1000/example"},
                )
            finally:
                cache.close()


class PageExtractionTests(unittest.TestCase):
    def test_blank_first_page_cannot_promote_a_later_reference_doi(self) -> None:
        for image_only in (False, True):
            with self.subTest(image_only=image_only), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "article.pdf"
                with pymupdf.open() as document:
                    first = document.new_page()
                    if image_only:
                        pixmap = pymupdf.Pixmap(pymupdf.csGRAY, 32, 32, bytes([200]) * 1024, False)
                        first.insert_image(first.rect, pixmap=pixmap)
                    document.new_page().insert_textbox(
                        pymupdf.Rect(72, 72, 540, 720),
                        "Discussion. The results extend earlier observations.\n"
                        "References\n1. Other A. Cited work. doi:10.5555/cited.p2",
                        fontsize=9,
                    )
                    document.save(path)
                evidence = pdf_evidence.inspect_pdf(path)
                candidate = next(item for item in evidence.dois if item.doi == "10.5555/cited.p2")
                self.assertLess(candidate.score, 80)
                self.assertFalse(candidate.source.startswith("text:p1:"))

    def test_arxiv_provenance_and_identifier_variants(self) -> None:
        with self.subTest(case="arxiv_citation_on_first_page_is_not_document_provenance"):
            for citation in (
                "Prior work: Smith et al. arXiv:2509.16803v1.",
                "References\narXiv:2509.16803v1 [q-bio.NC]",
                "See https://arxiv.org/abs/2509.16803 for related work.",
                "See https://doi.org/10.48550/arXiv.2509.16803 for related work.",
            ):
                with self.subTest(citation=citation), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "journal.pdf"
                    _create_pdf_with_pages(path, ["A reliable journal article about synthetic imaging\n" + citation])
                    evidence = pdf_evidence.inspect_pdf(path)
                    self.assertIsNone(evidence.arxiv_doi)
                    self.assertTrue(all(candidate.score < 80 for candidate in evidence.dois))

        with self.subTest(case="arxiv_identifier_variants_become_canonical_dois"):
            variants = (
                "arXiv:2509.16803v1 [q-bio.NC]",
                "https://arxiv.org/pdf/2509.16803v2.pdf",
                "https://doi.org/10.48550/arXiv.2509.16803",
            )

            for text in variants:
                with self.subTest(text=text):
                    self.assertEqual(
                        pdf_evidence.extract_arxiv_dois_from_string(text),
                        ["10.48550/arXiv.2509.16803"],
                    )

        with self.subTest(case="multiple_arxiv_ids_are_not_selected_implicitly"):
            self.assertIsNone(
                pdf_evidence._unique_arxiv_doi_from_texts(
                    (
                        "arXiv:2509.16803v1",
                        "arXiv:1810.10973v2",
                    )
                )
            )


    def test_later_page_doi_cannot_gain_first_page_priority_from_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.pdf"
            _create_pdf_with_pages(path, [
                "Regulation of T cell activation by inhibitory receptors in mice",
                "References: doi:10.5555/cited.0001",
            ])
            evidence = pdf_evidence.inspect_pdf(path)
            self.assertEqual(len(evidence.dois), 1)
            self.assertLess(evidence.dois[0].score, 80)

    def test_first_page_pii_evidence(self) -> None:
        with self.subTest(case="pii_normalization_handles_publisher_and_ocr_spacing"):
            expected = "S1352866100000697"
            for value in (
                "S1352-8661(00)00069-7",
                "S 1352-8661 (00)00069-7",
                "s 1 3 5 2 - 8 6 6 1 ( 0 0 ) 0 0 0 6 9 - 7",
                expected,
            ):
                with self.subTest(value=value):
                    self.assertEqual(pdf_evidence.clean_pii(value), expected)

        with self.subTest(case="pii_extraction_prefers_labeled_first_page_identifier"):
            for text in (
                (
                    "1352-8661 00/$ - see front matter\n"
                    "PII: S 1352-8661 (00)00069-7\n"
                    "Open access birdcage coils for microscopic imaging"
                ),
                "PII:\nS 1 3 5 2 - 8 6 6 1 ( 0 0 ) 0 0 0 6 9 - 7",
            ):
                with self.subTest(text=text):
                    self.assertEqual(
                        pdf_evidence.extract_piis_from_string(text),
                        ["S1352866100000697"],
                    )

        with self.subTest(case="multiple_first_page_piis_are_not_selected_implicitly"):
            texts = (
                "PII: S1090-7807(19)30023-0",
                "PII: S1352-8661(00)00069-7",
            )

            self.assertIsNone(pdf_evidence._unique_pii_from_texts(texts))

        with self.subTest(case="issn_and_front_matter_are_not_mistaken_for_pii"):
            self.assertEqual(
                pdf_evidence.extract_piis_from_string(
                    "ISSN 1352-8661; 1352-8661 00/$ - see front matter"
                ),
                [],
            )


    def test_scientific_title_reconstruction(self) -> None:
        with self.subTest(case="scientific_title_subscripts_are_rejoined"):
            self.assertEqual(
                pdf_evidence.clean_title_candidate(
                    "Mapping fat-water separated R 1 , R 2 * , and proton density"
                ),
                "Mapping fat-water separated R1, R2*, and proton density",
            )

        with self.subTest(case="plain_text_title_does_not_append_uppercase_author_byline"):
            text = (
                "JOURNAL OF MAGNETIC RESONANCE 125, 178-184 (1997)\n"
                "A Half-Volume Coil for Efficient Proton Decoupling\n"
                "in Humans at 4 Tesla\n"
                "GREGOR ADRIANY AND ROLF GRUETTER\n"
                "Received October 18, 1996\n"
            )
            self.assertEqual(
                pdf_evidence._title_from_plain_text(text),
                "A Half-Volume Coil for Efficient Proton Decoupling in Humans at 4 Tesla",
            )

        with self.subTest(case="span_title_appends_short_same_size_continuation"):
            class Rect:
                height = 842.0

            class Page:
                rect = Rect()

                @staticmethod
                def get_text(mode: str):
                    if mode != "dict":
                        raise AssertionError(f"unexpected text mode: {mode}")
                    return {
                        "blocks": [
                            {
                                "lines": [
                                    {
                                        "bbox": (57.7, 138.0, 510.0, 152.0),
                                        "spans": [
                                            {
                                                "text": "ROLE OF MAGNETIC RESONANCE IMAGING IN CORONARY",
                                                "size": 14.0,
                                            }
                                        ],
                                    },
                                    {
                                        "bbox": (57.7, 154.0, 180.0, 168.0),
                                        "spans": [
                                            {
                                                "text": "HEART DISEASE.",
                                                "size": 14.0,
                                            }
                                        ],
                                    },
                                    {
                                        "bbox": (57.7, 177.9, 410.0, 188.0),
                                        "spans": [
                                            {
                                                "text": "J. Bogaert, S. Dymarkowski, F.E. Rademakers",
                                                "size": 9.0,
                                            }
                                        ],
                                    },
                                ]
                            }
                        ]
                    }

            self.assertEqual(
                pdf_evidence._title_from_first_page(Page(), []),
                "ROLE OF MAGNETIC RESONANCE IMAGING IN CORONARY HEART DISEASE",
            )

        with self.subTest(case="original_contribution_words_can_start_a_real_title"):
            self.assertTrue(
                pdf_evidence.looks_like_title(
                    "Original Contributions to Magnetic Resonance Infusion Pump Design"
                )
            )

        with self.subTest(case="leading_article_label_is_removed"):
            self.assertEqual(
                pdf_evidence.clean_title_candidate(
                    "A R T I C L E A study on multiple array method"
                ),
                "A study on multiple array method",
            )


    def test_page_representation_reuse(self) -> None:
        with self.subTest(case="only_distinct_pymupdf_representations_are_requested"):
            class Page:
                def __init__(self) -> None:
                    self.calls: list[str] = []

                def get_text(self, mode: str):
                    self.calls.append(mode)
                    if mode == "text":
                        return "text DOI 10.1000/example"
                    if mode == "blocks":
                        return [(0, 0, 1, 1, "block DOI 10.1000/example")]
                    return {
                        "blocks": [
                            {"lines": [{"spans": [{"text": "span DOI 10.1000/example"}]}]}
                        ]
                    }

            page = Page()
            variants = pdf_evidence.page_text_variants(page)
            self.assertEqual(page.calls, ["text", "blocks", "dict"])
            self.assertEqual(len(variants), 3)

        with self.subTest(case="cached_page_variants_form_document_text_samples"):
            samples = pdf_evidence.combine_page_text_variants(
                [
                    ["plain page one", "block page one", "span page one"],
                    ["plain page two", "block page two", "span page two"],
                ]
            )

            self.assertEqual(
                samples,
                (
                    "plain page one\fplain page two",
                    "block page one\fblock page two",
                    "span page one\fspan page two",
                ),
            )


    def test_title_noise_is_rejected(self) -> None:
        with self.subTest(case="bibliographic_header_is_not_a_title"):
            self.assertFalse(
                pdf_evidence.looks_like_title(
                    "Magn Reson Med Sci 2017; 16; 253-258 "
                    "Published Online: September 30, 2016"
                )
            )

        with self.subTest(case="publisher_boilerplate_is_not_a_title"):
            for title in (
                "Attribution 4.0 International CC BY 4.0 License Imaging Neuroscience",
                "Contents lists available at ScienceDirect BBA Molecular Basis of Disease",
                "R e s e a r c h A r t i c l e",
            ):
                with self.subTest(title=title):
                    self.assertFalse(pdf_evidence.looks_like_title(title))

        with self.subTest(case="span_title_skips_ocr_article_type_boilerplate"):
            class Rect:
                height = 792.0

            class Page:
                rect = Rect()

                @staticmethod
                def get_text(mode: str):
                    if mode != "dict":
                        raise AssertionError(f"unexpected text mode: {mode}")
                    return {
                        "blocks": [
                            {
                                "lines": [
                                    {
                                        "bbox": (54.0, 42.0, 218.0, 60.0),
                                        "spans": [
                                            {
                                                "text": "l Original Contribution",
                                                "size": 14.6,
                                            }
                                        ],
                                    },
                                    {
                                        "bbox": (54.0, 94.0, 352.0, 112.0),
                                        "spans": [
                                            {
                                                "text": "A SIMPLE MR-COMPATIBLE INFUSION PUMP",
                                                "size": 12.3,
                                            }
                                        ],
                                    },
                                    {
                                        "bbox": (130.0, 123.0, 500.0, 140.0),
                                        "spans": [
                                            {
                                                "text": "DASIKA L. VARASIMHAM, FRANK J. LONDY",
                                                "size": 11.1,
                                            }
                                        ],
                                    },
                                ]
                            }
                        ]
                    }

            self.assertEqual(
                pdf_evidence._title_from_first_page(Page(), []),
                "A SIMPLE MR-COMPATIBLE INFUSION PUMP",
            )

        with self.subTest(case="scanner_and_access_metadata_are_not_titles"):
            for title in (
                "Scanned doc in PDF format",
                "scanned document from HP9100C",
                "at UNIV OF PITTSBURGH HSLS on June 5, 2008 For personal use only",
            ):
                with self.subTest(title=title):
                    self.assertFalse(pdf_evidence.looks_like_title(title))

        with self.subTest(case="scanner_subject_is_not_used_as_a_title"):
            metadata = {
                "title": "",
                "subject": "A Plausible but Stale Paper Title",
                "creator": "HP 9100C Digital Sender",
            }

            self.assertIsNone(pdf_evidence._pdf_title_from_metadata(metadata))


class PubMedQueryOptimizationTests(unittest.IsolatedAsyncioTestCase):
    def test_normalized_pubmed_query_parameters(self) -> None:
        with self.subTest(case="lowercase_doi_does_not_duplicate_search_terms"):
            terms = metadata_providers.pubmed_doi_terms("10.1000/example")
            self.assertEqual(len(terms), 4)
            self.assertEqual(len(set(terms)), 4)

        with self.subTest(case="ncbi_params_include_only_nonempty_credentials"):
            without_credentials = metadata_providers._ncbi_request_params(
                {"db": "pubmed"},
                api_key="",
                email="",
            )
            with_credentials = metadata_providers._ncbi_request_params(
                {"db": "pubmed"},
                api_key=" test-key ",
                email=" test@example.org ",
            )

            self.assertEqual(without_credentials, {"db": "pubmed"})
            self.assertEqual(
                with_credentials,
                {
                    "db": "pubmed",
                    "api_key": "test-key",
                    "email": "test@example.org",
                },
            )


    async def test_pubmed_request_optimization(self) -> None:
        with self.subTest(case="id_converter_result_skips_redundant_pubmed_searches"):
            class Cache:
                @staticmethod
                def get_doi2pmid(_doi: str):
                    return None

                @staticmethod
                def put_doi2pmid(_doi: str, _pmid: str) -> None:
                    return None

            class Limiter:
                @staticmethod
                async def acquire() -> None:
                    return None

            class Response:
                status_code = 200
                headers: dict[str, str] = {}

                @staticmethod
                def raise_for_status() -> None:
                    return None

                @staticmethod
                def json():
                    return {"records": [{"pmid": "12345"}]}

            class Client:
                def __init__(self) -> None:
                    self.calls: list[str] = []

                async def get(self, url: str, **_kwargs):
                    self.calls.append(url)
                    return Response()

            client = Client()
            adapter = metadata_providers.PubMedAdapter(
                client,
                Cache(),
                "",
                "test@example.org",
                Limiter(),
                {},
                {},
            )
            ids = await adapter.doi_to_pmids("10.1000/example")
            self.assertEqual(ids, ["12345"])
            self.assertEqual(client.calls, [metadata_providers.PMC_IDCONV])

        with self.subTest(case="title_search_requests_relevance_order"):
            class Cache:
                pass

            class Limiter:
                @staticmethod
                async def acquire() -> None:
                    return None

            class Response:
                status_code = 200
                headers: dict[str, str] = {}

                @staticmethod
                def raise_for_status() -> None:
                    return None

                @staticmethod
                def json():
                    return {"esearchresult": {"idlist": ["12083625"]}}

            class Client:
                def __init__(self) -> None:
                    self.params: list[dict[str, object]] = []

                async def get(self, _url: str, **kwargs):
                    self.params.append(kwargs["params"])
                    return Response()

            client = Client()
            adapter = metadata_providers.PubMedAdapter(
                client,
                Cache(),
                "",
                "test@example.org",
                Limiter(),
                {},
                {},
            )

            ids = await adapter.ids_by_title(
                "Role of magnetic resonance imaging in coronary heart disease"
            )

            self.assertEqual(ids, ["12083625"])
            self.assertEqual(client.params[0]["sort"], "relevance")
            self.assertNotIn("api_key", client.params[0])
            self.assertEqual(client.params[0]["email"], "test@example.org")


class TitleLookupGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_titles_are_rejected_before_provider_requests(self) -> None:
        unusable_title = "untitled"
        class Cache:
            @staticmethod
            def get_crossref(_doi: str):
                return None

            @staticmethod
            def put_crossref(_doi: str, _message: dict) -> None:
                return None

        class Limiter:
            @staticmethod
            async def acquire() -> None:
                return None

        class Client:
            async def get(self, *_args, **_kwargs):
                raise AssertionError("provider request should not occur")

        client = Client()
        limiter = Limiter()
        headers: dict[str, str] = {}

        pubmed_adapter = metadata_providers.PubMedAdapter(
            client,
            Cache(),
            "",
            "",
            limiter,
            headers,
            headers,
        )
        crossref_adapter = metadata_providers.CrossrefAdapter(
            client,
            Cache(),
            limiter,
            headers,
            asyncio.Semaphore(1),
        )
        datacite_adapter = metadata_providers.DataCiteAdapter(
            client,
            limiter,
            headers,
            asyncio.Semaphore(1),
        )
        pubmed = await pubmed_adapter.ids_by_title(unusable_title)
        crossref = await crossref_adapter.messages_by_title(unusable_title)
        datacite = await datacite_adapter.attributes_by_title(unusable_title)

        self.assertEqual(pubmed, [])
        self.assertEqual(crossref, [])
        self.assertEqual(datacite, [])


# DOI validation

class DoiValidationTests(unittest.TestCase):
    def test_doi_syntax_and_article_scope(self) -> None:
        with self.subTest(case="journal_level_issn_identifier_is_not_an_article_doi"):
            text = (
                "Journal DOI 10.1002/(ISSN)1522-2594 "
                "Article DOI https://doi.org/10.1002/mrm.30298"
            )

            self.assertEqual(
                pdf_evidence.extract_dois_from_string(text),
                ["10.1002/mrm.30298"],
            )

        with self.subTest(case="xml_metadata_tail_is_removed_from_doi"):
            text = (
                "<prism_2_:doi>10.1002/nbm.4369</prism_2_:doi>"
                "<prism_2_:url>https://doi.org/10.1002/nbm.4369</prism_2_:url>"
            )

            self.assertEqual(
                pdf_evidence.extract_dois_from_string(text),
                ["10.1002/nbm.4369"],
            )

        with self.subTest(case="valid_doi_does_not_absorb_following_column_text"):
            text = (
                "doi:10.2463/mrms.tn.2016-0149\n"
                "confined mode spacing between desired modes"
            )

            self.assertEqual(
                pdf_evidence.extract_dois_from_string(text),
                ["10.2463/mrms.tn.2016-0149"],
            )


    def test_publisher_doi_precedes_equal_score_preprint_doi(self) -> None:
        candidates = {
            "preprint": pdf_evidence.DoiCandidate(
                "10.1101/2024.02.16.579531",
                "link:p1",
                115,
            ),
            "publisher": pdf_evidence.DoiCandidate(
                "10.7554/eLife.97088",
                "link:p1",
                115,
            ),
        }

        ranked = pdf_evidence._rank_doi_candidates(candidates)

        self.assertEqual(ranked[0].doi, "10.7554/eLife.97088")
        self.assertEqual(ranked[1].doi, "10.1101/2024.02.16.579531")

    def test_authorless_journal_record_is_not_enough_metadata(self) -> None:
        record = metadata_providers.parse_crossref_message(
            {
                "DOI": "10.1002/(ISSN)1522-2594",
                "title": ["Magnetic Resonance in Medicine"],
                "issued": {"date-parts": [[2006]]},
                "type": "journal",
            }
        )

        self.assertFalse(record_matching.record_has_enough_metadata(record))


# Spawned PDF processing

def _create_process_pdf(
    path: Path,
    *,
    user_password: str | None = None,
) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Spawned PDF Inspection\nhttps://doi.org/10.1000/spawned",
    )
    options = {}
    if user_password is not None:
        options = {
            "encryption": pymupdf.PDF_ENCRYPT_AES_256,
            "owner_pw": "owner-password",
            "user_pw": user_password,
        }
    document.save(path, **options)
    document.close()


def _crash_once_inspection(path, password, pages):
    marker = Path(path).with_suffix(".crashed")
    if not marker.exists():
        marker.write_text("fixture crash")
        os._exit(73)
    return pdf_evidence.inspect_pdf(Path(path))


def _hanging_inspection(path, password, pages):
    Path(path).with_suffix(".started").write_text(str(os.getpid()))
    if Path(path).stem == "hang":
        time.sleep(60)
    return pdf_evidence.inspect_pdf(Path(path))


def _always_crashing_inspection(path, password, pages):
    os._exit(74)


def _sigint_during_rewrite(path, password, indices, compress, token):
    unlink = pdf_rewrite._unlink_owned
    def signal_before_unlink(candidate, expected):
        if candidate == Path(path):
            os.kill(os.getpid(), signal.SIGINT)
        return unlink(candidate, expected)
    try:
        with mock.patch.object(pdf_rewrite, "_unlink_owned", signal_before_unlink):
            return pdf_rewrite.rewrite_pdf(Path(path), pdf_password=password, expected_source_token=token)
    except KeyboardInterrupt:
        return pdf_rewrite.PdfRewriteOutcome("fixture_interrupted", "SIGINT reached rewrite transaction")


class PdfProcessTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(os.name == "posix", "requires POSIX SIGINT delivery")
    async def test_sigint_in_worker_cannot_interrupt_active_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encrypted.pdf"
            _create_process_pdf(path, user_password="password")
            async with pdf_process.SpawnPdfProcessRunner(max_workers=1) as runner:
                with mock.patch.object(pdf_process, "_rewrite_pdf_worker", _sigint_during_rewrite):
                    result = await runner.rewrite(path, pdf_password="password")
            self.assertEqual(result.status, "pdf_decrypted")
            with pymupdf.open(path) as document:
                self.assertFalse(document.needs_pass)
                for page in document:
                    page.get_text()

    async def test_permanent_worker_crash_stops_after_one_retry(self) -> None:
        async with pdf_process.SpawnPdfProcessRunner(max_workers=1) as runner:
            with mock.patch.object(pdf_process, "_inspect_pdf_worker", _always_crashing_inspection):
                evidence = await runner.inspect(Path("synthetic.pdf"))
        self.assertEqual(evidence.pdf_status, "pdf_processing_error")
        self.assertIn("twice", evidence.pdf_note)
    async def test_broken_inspection_worker_is_replaced_and_task_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.pdf"
            _create_process_pdf(path)
            async with pdf_process.SpawnPdfProcessRunner(max_workers=1) as runner:
                with mock.patch.object(pdf_process, "_inspect_pdf_worker", _crash_once_inspection):
                    evidence = await runner.inspect(path)
                subsequent = await runner.inspect(path)
            self.assertEqual(evidence.pdf_status, "ok")
            self.assertEqual(subsequent.pdf_status, "ok")

    async def test_inspection_timeout_terminates_worker_and_allows_next_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hang.pdf"
            healthy = Path(directory) / "healthy.pdf"
            _create_process_pdf(path)
            _create_process_pdf(healthy)
            async with pdf_process.SpawnPdfProcessRunner(max_workers=1, inspection_timeout=10) as runner:
                with mock.patch.object(pdf_process, "_inspect_pdf_worker", _hanging_inspection):
                    failed, good = await asyncio.gather(runner.inspect(path), runner.inspect(healthy))
            self.assertEqual(failed.pdf_status, "pdf_processing_timeout")
            self.assertTrue(path.with_suffix(".started").exists())
            self.assertEqual(good.pdf_status, "ok", str(list(Path(directory).iterdir())))

    async def test_spawned_worker_does_not_execute_parent_inspector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "paper.pdf"
            _create_process_pdf(pdf)

            async with pdf_process.SpawnPdfProcessRunner(
                max_workers=1
            ) as runner:
                with mock.patch.object(
                    pdf_evidence,
                    "inspect_pdf",
                    side_effect=AssertionError("parent inspector executed"),
                ):
                    evidence = await runner.inspect(pdf)

            self.assertEqual(
                [candidate.doi for candidate in evidence.dois],
                ["10.1000/spawned"],
            )

    async def test_spawned_worker_completes_rewrite_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "encrypted.pdf"
            _create_process_pdf(pdf, user_password="user-password")

            async with pdf_process.SpawnPdfProcessRunner(
                max_workers=1
            ) as runner:
                before = await runner.inspect(
                    pdf,
                    pdf_password="user-password",
                )
                outcome = await runner.rewrite(
                    pdf,
                    pdf_password="user-password",
                )
                after = await runner.inspect(pdf)

            self.assertEqual(before.pdf_status, "pdf_decryption_needed")
            self.assertEqual(outcome.status, "pdf_decrypted")
            self.assertTrue(outcome.changed)
            self.assertEqual(after.pdf_status, "ok")


# Rewrite recovery

def _create_recovery_pdf(path: Path, text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()
    return path.read_bytes()


class PdfRecoveryTests(unittest.TestCase):
    def test_missing_canonical_recovery_modes(self) -> None:
        with self.subTest(case="dry_run_reports_missing_canonical_without_mutating"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                canonical = root / "paper.pdf"
                expected = _create_recovery_pdf(canonical, "Original")
                rollback = root / ".paper.pdf.renamepdf-rollback"
                canonical.rename(rollback)

                outcomes = pdf_rewrite.recover_interrupted_rewrites(root)

                self.assertEqual(len(outcomes), 1)
                self.assertEqual(outcomes[0].status, "pdf_recovery_needed")
                self.assertTrue(outcomes[0].blocks_run)
                self.assertFalse(outcomes[0].changed)
                self.assertFalse(canonical.exists())
                self.assertEqual(rollback.read_bytes(), expected)

        with self.subTest(case="commit_restores_rollback_and_preserves_temporary_copy"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                canonical = root / "paper.pdf"
                expected = _create_recovery_pdf(canonical, "Original")
                rollback = root / ".paper.pdf.renamepdf-rollback"
                canonical.rename(rollback)
                temporary = root / ".paper.pdf.repairing.pdf"
                candidate = _create_recovery_pdf(temporary, "Candidate replacement")

                outcomes = pdf_rewrite.recover_interrupted_rewrites(
                    root,
                    commit=True,
                )

                self.assertEqual(outcomes[0].status, "pdf_recovery_restored")
                self.assertFalse(outcomes[0].blocks_run)
                self.assertTrue(outcomes[0].changed)
                self.assertEqual(canonical.read_bytes(), expected)
                self.assertFalse(rollback.exists())
                self.assertEqual(temporary.read_bytes(), candidate)


    def test_ambiguous_recovery_preserves_every_copy(self) -> None:
        with self.subTest(case="restore_cannot_overwrite_path_appearing_during_recovery"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                canonical = root / "paper.pdf"
                original = _create_recovery_pdf(canonical, "Original")
                rollback = root / ".paper.pdf.renamepdf-rollback"
                canonical.rename(rollback)
                real_copy = pdf_rewrite.create_exclusive_copy
                intruder: bytes | None = None

                def race_copy(source, destination):
                    nonlocal intruder
                    if Path(destination).resolve() == canonical.resolve():
                        intruder = _create_recovery_pdf(canonical, "Concurrent writer")
                    return real_copy(source, destination)

                with mock.patch.object(
                    pdf_rewrite,
                    "create_exclusive_copy",
                    side_effect=race_copy,
                ):
                    outcomes = pdf_rewrite.recover_interrupted_rewrites(
                        root,
                        commit=True,
                    )

                self.assertEqual(outcomes[0].status, "pdf_recovery_failed")
                self.assertTrue(outcomes[0].blocks_run)
                self.assertEqual(canonical.read_bytes(), intruder)
                self.assertEqual(rollback.read_bytes(), original)

        with self.subTest(case="existing_canonical_and_rollback_are_ambiguous"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                canonical = root / "paper.pdf"
                canonical_bytes = _create_recovery_pdf(canonical, "Installed candidate")
                rollback = root / ".paper.pdf.renamepdf-rollback"
                rollback_bytes = _create_recovery_pdf(rollback, "Original")

                outcomes = pdf_rewrite.recover_interrupted_rewrites(
                    root,
                    commit=True,
                )

                self.assertEqual(outcomes[0].status, "pdf_recovery_ambiguous")
                self.assertTrue(outcomes[0].blocks_run)
                self.assertEqual(canonical.read_bytes(), canonical_bytes)
                self.assertEqual(rollback.read_bytes(), rollback_bytes)

        with self.subTest(case="broken_canonical_symlink_and_rollback_are_ambiguous"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                canonical = root / "paper.pdf"
                canonical.symlink_to(root / "missing-target.pdf")
                rollback = root / ".paper.pdf.renamepdf-rollback"
                rollback_bytes = _create_recovery_pdf(rollback, "Original")

                outcomes = pdf_rewrite.recover_interrupted_rewrites(
                    root,
                    commit=True,
                )

                self.assertEqual(outcomes[0].status, "pdf_recovery_ambiguous")
                self.assertTrue(canonical.is_symlink())
                self.assertEqual(rollback.read_bytes(), rollback_bytes)

        with self.subTest(case="missing_canonical_with_only_temporary_copy_is_ambiguous"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                temporary = root / ".paper.pdf.installing.pdf"
                _create_recovery_pdf(temporary, "Candidate replacement")

                outcomes = pdf_rewrite.recover_interrupted_rewrites(
                    root,
                    commit=True,
                )

                self.assertEqual(outcomes[0].status, "pdf_recovery_ambiguous")
                self.assertTrue(temporary.exists())
                self.assertFalse((root / "paper.pdf").exists())


    def test_restore_unlink_failure_leaves_recoverable_copy_and_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "paper.pdf"
            expected = _create_recovery_pdf(canonical, "Original")
            rollback = root / ".paper.pdf.renamepdf-rollback"
            canonical.rename(rollback)
            real_unlink = Path.unlink

            def fail_rollback_unlink(path: Path, *args, **kwargs):
                if path.name == rollback.name:
                    raise PermissionError("simulated cleanup failure")
                return real_unlink(path, *args, **kwargs)

            with mock.patch.object(Path, "unlink", new=fail_rollback_unlink):
                restored = pdf_rewrite.recover_interrupted_rewrites(
                    root,
                    commit=True,
                )

            self.assertEqual(
                restored[0].status,
                "pdf_recovery_cleanup_blocked",
            )
            self.assertTrue(restored[0].blocks_run)
            self.assertTrue(restored[0].changed)
            self.assertEqual(canonical.read_bytes(), expected)
            self.assertEqual(rollback.read_bytes(), expected)
            marker = root / ".paper.pdf.renamepdf-installed-marker"
            self.assertTrue(marker.exists())

            pending = pdf_rewrite.recover_interrupted_rewrites(root)

            self.assertEqual(pending[0].status, "pdf_recovery_cleanup_needed")
            self.assertFalse(pending[0].blocks_run)

            cleaned = pdf_rewrite.recover_interrupted_rewrites(
                root,
                commit=True,
            )

            self.assertEqual(cleaned[0].status, "pdf_recovery_cleanup_completed")
            self.assertFalse(rollback.exists())
            self.assertFalse(marker.exists())

    def test_completion_marker_requires_the_installed_file(self) -> None:
        with self.subTest(case="installed_marker_rejects_replaced_canonical"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                canonical = root / "paper.pdf"
                _create_recovery_pdf(canonical, "Validated installed PDF")
                marker = root / ".paper.pdf.renamepdf-installed-marker"
                marker.write_bytes(pdf_rewrite._marker_payload("install", canonical, 1))
                rollback = root / ".paper.pdf.renamepdf-rollback"
                rollback_bytes = _create_recovery_pdf(rollback, "Original PDF")
                concurrent = root / "concurrent.pdf"
                concurrent_bytes = _create_recovery_pdf(concurrent, "Concurrent replacement")
                concurrent.replace(canonical)

                outcomes = pdf_rewrite.recover_interrupted_rewrites(
                    root,
                    commit=True,
                )

                self.assertEqual(outcomes[0].status, "pdf_recovery_ambiguous")
                self.assertTrue(outcomes[0].blocks_run)
                self.assertEqual(canonical.read_bytes(), concurrent_bytes)
                self.assertEqual(rollback.read_bytes(), rollback_bytes)
                self.assertTrue(marker.exists())

        with self.subTest(case="completed_artifacts_do_not_block_after_canonical_move"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                canonical = root / "paper.pdf"
                installed_bytes = _create_recovery_pdf(canonical, "Validated installed PDF")
                marker = root / ".paper.pdf.renamepdf-installed-marker"
                marker.write_bytes(pdf_rewrite._marker_payload("install", canonical, 1))
                rollback = root / ".paper.pdf.renamepdf-rollback"
                rollback_bytes = _create_recovery_pdf(rollback, "Original PDF")
                moved = root / "renamed-paper.pdf"
                canonical.rename(moved)

                outcomes = pdf_rewrite.recover_interrupted_rewrites(
                    root,
                    commit=True,
                )

                self.assertEqual(
                    outcomes[0].status,
                    "pdf_recovery_completed_artifact_preserved",
                )
                self.assertFalse(outcomes[0].blocks_run)
                self.assertFalse(outcomes[0].changed)
                self.assertEqual(moved.read_bytes(), installed_bytes)
                self.assertEqual(rollback.read_bytes(), rollback_bytes)
                self.assertTrue(marker.exists())


    def test_orphan_temporary_copy_is_preserved_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_recovery_pdf(root / "paper.pdf", "Original")
            temporary = root / ".paper.pdf.repairing_2.pdf"
            _create_recovery_pdf(temporary, "Candidate replacement")

            outcomes = pdf_rewrite.recover_interrupted_rewrites(root)

            self.assertEqual(
                outcomes[0].status,
                "pdf_recovery_artifact_preserved",
            )
            self.assertFalse(outcomes[0].blocks_run)
            self.assertTrue(temporary.exists())

    def test_rewrite_artifact_names_are_recognized(self) -> None:
        artifacts = (
            ".paper.pdf.renamepdf-rollback",
            ".paper.pdf_2.renamepdf-rollback",
            ".paper.pdf.repairing.pdf",
            ".paper.pdf.repairing_2.pdf",
            ".paper.pdf.installing.pdf",
            ".paper.pdf.installing_3.pdf",
            ".paper.pdf.renamepdf-installed-marker",
            ".paper.pdf_2.renamepdf-installed-marker",
        )
        for name in artifacts:
            with self.subTest(name=name):
                self.assertTrue(pdf_rewrite.is_rewrite_artifact(Path(name)))
        self.assertFalse(pdf_rewrite.is_rewrite_artifact(Path("paper.pdf")))


# PDF rewrite and normalization

def _create_repair_pdf(
    path: Path,
    *,
    title: str = "Reliable PDF Repair for Scientific Papers",
    doi: str = "10.1000/repair-test",
    user_password: str | None = None,
) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), f"{title}\nhttps://doi.org/{doi}")
    doc.set_metadata({"title": title})
    save_options = {}
    if user_password is not None:
        save_options = {
            "encryption": pymupdf.PDF_ENCRYPT_AES_256,
            "owner_pw": "owner-password",
            "user_pw": user_password,
        }
    doc.save(path, **save_options)
    doc.close()


def _create_pdf_with_pages(
    path: Path,
    pages: list[str],
    *,
    signature_widget: bool = False,
) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(72, 72, 540, 720),
            text,
            fontsize=10,
        )
    if signature_widget:
        signature = pymupdf.Widget()
        signature.field_name = "Signature1"
        signature.field_type = pymupdf.PDF_WIDGET_TYPE_SIGNATURE
        signature.rect = pymupdf.Rect(72, 650, 220, 690)
        document[0].add_widget(signature)
    document.save(path)
    document.close()


def _make_pdf_large_and_compressible(path: Path) -> None:
    width = height = 1_100
    sample_count = width * height * 3
    pattern = bytes(range(251))
    samples = (pattern * ((sample_count // len(pattern)) + 1))[:sample_count]
    pixmap = pymupdf.Pixmap(
        pymupdf.csRGB,
        width,
        height,
        samples,
        False,
    )
    expanded = path.with_name(f"{path.stem}.expanded.pdf")
    document = pymupdf.open(path)
    document[0].insert_image(
        document[0].rect,
        pixmap=pixmap,
        overlay=False,
    )
    document.save(
        expanded,
        garbage=0,
        deflate=0,
        deflate_images=False,
        expand=1,
    )
    document.close()
    expanded.replace(path)
    if path.stat().st_size <= pdf_rewrite.LOSSLESS_COMPRESSION_THRESHOLD_BYTES:
        raise AssertionError("large-PDF fixture did not exceed 3 MB")


class PdfRepairTests(unittest.TestCase):
    def test_rewrite_rechecks_source_evidence(self) -> None:
        with self.subTest(case="rewrite_rejects_source_changed_since_analysis"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "changed.pdf"
                _create_pdf_with_pages(pdf, ["Original cover", "Original article text"])
                token = pdf_rewrite._file_token(pdf)
                incoming = pdf.with_name("incoming.pdf")
                _create_pdf_with_pages(incoming, ["New content page one", "New content page two"])
                incoming.replace(pdf)
                original = pdf.read_bytes()
                result = pdf_rewrite.rewrite_pdf(
                    pdf, remove_page_indices=(0,), expected_source_token=token,
                )
                self.assertEqual(result.status, "pdf_source_changed")
                self.assertEqual(pdf.read_bytes(), original)

        with self.subTest(case="rewrite_rechecks_cover_detection_even_without_a_token"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "article.pdf"
                _create_pdf_with_pages(pdf, ["Real content page one", "Real content page two"])
                original = pdf.read_bytes()
                result = pdf_rewrite.rewrite_pdf(pdf, remove_page_indices=(0,))
                self.assertEqual(result.status, "pdf_source_changed")
                self.assertEqual(pdf.read_bytes(), original)


    def test_lossless_compression_contract(self) -> None:
        with self.subTest(case="large_pdf_save_uses_verified_lossless_compression_options"):
            class RecordingDocument:
                save_options: dict[str, object] | None = None

                @staticmethod
                def load_page(_page_number: int) -> object:
                    return object()

                @staticmethod
                def subset_fonts(*_args, **_kwargs) -> None:
                    raise AssertionError("font subsetting must not be used")

                def save(self, path: str, **options: object) -> None:
                    self.save_options = options
                    Path(path).write_bytes(b"staged PDF")

                @staticmethod
                def __len__() -> int:
                    return 1

            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "compressed.pdf"
                document = RecordingDocument()

                with mock.patch.object(pdf_rewrite, "_validate_pdf"):
                    pdf_rewrite._save_validated_document(
                        document,
                        destination,
                        use_large_compression=True,
                        allow_save_repair=False,
                    )

                self.assertEqual(
                    document.save_options,
                    {
                        "garbage": 4,
                        "clean": False,
                        "deflate": 1,
                        "deflate_images": True,
                        "deflate_fonts": True,
                        "incremental": False,
                        "expand": 0,
                        "linear": False,
                        "encryption": pymupdf.PDF_ENCRYPT_NONE,
                        "preserve_metadata": True,
                        "use_objstms": True,
                        "compression_effort": 0,
                        "raise_on_repair": True,
                    },
                )

        with self.subTest(case="large_pdf_is_transactionally_compressed_without_content_loss"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "large.pdf"
                _create_repair_pdf(pdf)
                _make_pdf_large_and_compressible(pdf)
                original_size = pdf.stat().st_size
                before = pymupdf.open(pdf)
                try:
                    before_text = tuple(page.get_text() for page in before)
                    before_pixels = tuple(
                        page.get_pixmap(alpha=False).samples for page in before
                    )
                finally:
                    before.close()

                outcome = pdf_rewrite.rewrite_pdf(
                    pdf,
                    compress_if_large=True,
                )

                self.assertEqual(outcome.status, "pdf_compressed")
                self.assertTrue(outcome.changed)
                self.assertLess(pdf.stat().st_size, original_size)
                self.assertIn(b"/ObjStm", pdf.read_bytes())
                after = pymupdf.open(pdf)
                try:
                    self.assertEqual(
                        tuple(page.get_text() for page in after),
                        before_text,
                    )
                    self.assertEqual(
                        tuple(
                            page.get_pixmap(alpha=False).samples
                            for page in after
                        ),
                        before_pixels,
                    )
                finally:
                    after.close()


    def test_compression_preserves_nonreplaceable_originals(self) -> None:
        with self.subTest(case="small_pdf_is_not_rewritten_for_compression"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "small.pdf"
                _create_repair_pdf(pdf)
                original = pdf.read_bytes()

                outcome = pdf_rewrite.rewrite_pdf(
                    pdf,
                    compress_if_large=True,
                )

                self.assertEqual(outcome.status, "ok")
                self.assertFalse(outcome.changed)
                self.assertEqual(pdf.read_bytes(), original)

        with self.subTest(case="signed_large_pdf_is_not_compressed"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "signed.pdf"
                _create_pdf_with_pages(
                    pdf,
                    ["Signed scientific paper"],
                    signature_widget=True,
                )
                _make_pdf_large_and_compressible(pdf)
                original = pdf.read_bytes()

                outcome = pdf_rewrite.rewrite_pdf(
                    pdf,
                    compress_if_large=True,
                )

                self.assertEqual(
                    outcome.status,
                    "pdf_signed_compression_skipped",
                )
                self.assertFalse(outcome.changed)
                self.assertEqual(pdf.read_bytes(), original)

        with self.subTest(case="compression_failure_preserves_large_pdf"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "large.pdf"
                _create_repair_pdf(pdf)
                _make_pdf_large_and_compressible(pdf)
                original = pdf.read_bytes()

                with mock.patch.object(
                    pdf_rewrite,
                    "_write_validated_copy",
                    side_effect=OSError("simulated compression failure"),
                ):
                    outcome = pdf_rewrite.rewrite_pdf(
                        pdf,
                        compress_if_large=True,
                    )

                self.assertEqual(outcome.status, "pdf_compression_failed")
                self.assertFalse(outcome.changed)
                self.assertEqual(pdf.read_bytes(), original)

        with self.subTest(case="nonbeneficial_compression_preserves_large_pdf"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pdf = root / "large.pdf"
                _create_repair_pdf(pdf)
                _make_pdf_large_and_compressible(pdf)
                original = pdf.read_bytes()

                def write_larger_copy(document, pdf_path, **_kwargs):
                    temporary = root / f".{pdf_path.name}.repairing.pdf"
                    temporary.write_bytes(original + b"larger")
                    token = pdf_rewrite._file_token(temporary)
                    if token is None:
                        raise AssertionError("temporary fixture is not a file")
                    return temporary, len(document), token

                with mock.patch.object(
                    pdf_rewrite,
                    "_write_validated_copy",
                    side_effect=write_larger_copy,
                ):
                    outcome = pdf_rewrite.rewrite_pdf(
                        pdf,
                        compress_if_large=True,
                    )

                self.assertEqual(
                    outcome.status,
                    "pdf_compression_not_beneficial",
                )
                self.assertFalse(outcome.changed)
                self.assertEqual(pdf.read_bytes(), original)
                self.assertFalse((root / ".large.pdf.repairing.pdf").exists())


    def test_exclusive_copy_preserves_modification_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            destination = root / "destination.pdf"
            source.write_bytes(b"source")
            expected = source.stat().st_mtime_ns

            pdf_rewrite.create_exclusive_copy(source, destination)

            self.assertEqual(destination.stat().st_mtime_ns, expected)

    def test_exclusive_copy_removes_owned_partial_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            destination = root / "destination.pdf"
            source.write_bytes(b"complete source")

            def fail_after_partial_copy(input_stream, output_stream, **_kwargs):
                input_stream.read(1)
                output_stream.write(b"partial")
                output_stream.flush()
                raise OSError("simulated copy failure")

            with (
                mock.patch.object(
                    pdf_rewrite.shutil,
                    "copyfileobj",
                    side_effect=fail_after_partial_copy,
                ),
                self.assertRaisesRegex(OSError, "simulated copy failure"),
            ):
                pdf_rewrite.create_exclusive_copy(source, destination)

            self.assertEqual(source.read_bytes(), b"complete source")
            self.assertFalse(destination.exists())

    def test_validated_install_refuses_changed_or_symlink_paths(self) -> None:
        with self.subTest(case="validated_copy_preserves_broken_destination_symlink"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.pdf"
                destination = root / "normalized.pdf"
                outside = root / "outside.pdf"
                _create_repair_pdf(source)
                destination.symlink_to(outside)

                with self.assertRaises(FileExistsError):
                    pdf_rewrite.write_validated_copy(source, destination)

                self.assertTrue(destination.is_symlink())
                self.assertFalse(outside.exists())

        with self.subTest(case="validated_copy_preserves_path_created_at_exclusive_install"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.pdf"
                destination = root / "normalized.pdf"
                _create_repair_pdf(source)
                concurrent = b"concurrent owner"
                real_copy = pdf_rewrite.create_exclusive_copy

                def race_copy(copy_source: Path, copy_destination: Path):
                    copy_destination.write_bytes(concurrent)
                    return real_copy(copy_source, copy_destination)

                with (
                    mock.patch.object(
                        pdf_rewrite,
                        "create_exclusive_copy",
                        side_effect=race_copy,
                    ),
                    self.assertRaises(FileExistsError),
                ):
                    pdf_rewrite.write_validated_copy(source, destination)

                self.assertEqual(destination.read_bytes(), concurrent)

        with self.subTest(case="validated_copy_preserves_replacement_before_final_validation"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.pdf"
                destination = root / "normalized.pdf"
                concurrent = root / "concurrent.pdf"
                _create_repair_pdf(source, title="Source")
                _create_repair_pdf(concurrent, title="Concurrent")
                concurrent_bytes = concurrent.read_bytes()
                real_validate = pdf_rewrite._validate_pdf

                def replace_before_validation(path: Path, expected_pages: int) -> None:
                    if Path(path) == destination:
                        concurrent.replace(destination)
                    real_validate(path, expected_pages)

                with (
                    mock.patch.object(
                        pdf_rewrite,
                        "_validate_pdf",
                        side_effect=replace_before_validation,
                    ),
                    self.assertRaisesRegex(RuntimeError, "foreign path preserved"),
                ):
                    pdf_rewrite.write_validated_copy(source, destination)

                self.assertEqual(destination.read_bytes(), concurrent_bytes)


    @unittest.skipUnless(
        sys.platform != "win32",
        "requires POSIX broken-symlink semantics",
    )
    def test_install_validated_copy_does_not_follow_broken_temp_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "paper.pdf"
            replacement = root / "replacement.pdf"
            outside = root / "outside.pdf"
            occupied = root / ".paper.pdf.installing.pdf"
            _create_repair_pdf(canonical, title="Original")
            _create_repair_pdf(replacement, title="Replacement")
            occupied.symlink_to(outside)

            outcome = pdf_rewrite.install_validated_copy(
                canonical,
                replacement,
            )

            self.assertTrue(outcome.changed)
            self.assertTrue(occupied.is_symlink())
            self.assertFalse(outside.exists())

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS metadata")
    def test_rewrite_preserves_macos_extended_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "encrypted.pdf"
            _create_repair_pdf(pdf, user_password="user-password")
            subprocess.run(
                [
                    "/usr/bin/xattr",
                    "-w",
                    "com.example.renamepdf",
                    "retained",
                    str(pdf),
                ],
                check=True,
            )

            outcome = pdf_rewrite.rewrite_pdf(
                pdf,
                pdf_password="user-password",
            )

            value = subprocess.run(
                [
                    "/usr/bin/xattr",
                    "-p",
                    "com.example.renamepdf",
                    str(pdf),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertTrue(outcome.changed)
            self.assertEqual(value, "retained")

    def test_empty_and_image_only_evidence(self) -> None:
        with self.subTest(case="zero_page_document_returns_clean_empty_evidence"):
            class EmptyDocument:
                metadata: dict[str, str] = {}
                is_repaired = False
                is_encrypted = False
                needs_pass = False

                @staticmethod
                def __len__() -> int:
                    return 0

                @staticmethod
                def close() -> None:
                    return None

            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "empty.pdf"
                pdf.write_bytes(b"%PDF-1.7\n%%EOF\n")

                with mock.patch.object(
                    pdf_evidence.pymupdf,
                    "open",
                    return_value=EmptyDocument(),
                ):
                    evidence = pdf_evidence.inspect_pdf(pdf)

                self.assertEqual(evidence.pdf_status, "ok")
                self.assertEqual(evidence.dois, [])
                self.assertIsNone(evidence.title)

        with self.subTest(case="image_only_pdf_reports_that_ocr_is_required"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "scan.pdf"
                document = pymupdf.open()
                page = document.new_page()
                pixmap = pymupdf.Pixmap(
                    pymupdf.csGRAY,
                    100,
                    100,
                    bytes([255]) * 10_000,
                    False,
                )
                page.insert_image(page.rect, pixmap=pixmap)
                document.save(pdf)
                document.close()

                evidence = pdf_evidence.inspect_pdf(pdf)

                self.assertEqual(evidence.pdf_status, "pdf_image_only")
                self.assertIn("OCR is required", evidence.pdf_note)
                self.assertEqual(evidence.dois, [])
                self.assertIsNone(evidence.title)

        with self.subTest(case="image_only_pdf_can_still_use_metadata_doi"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "scan-with-metadata.pdf"
                document = pymupdf.open()
                page = document.new_page()
                pixmap = pymupdf.Pixmap(
                    pymupdf.csGRAY,
                    100,
                    100,
                    bytes([255]) * 10_000,
                    False,
                )
                page.insert_image(page.rect, pixmap=pixmap)
                document.set_metadata({"subject": "doi:10.1172/JCI200318868"})
                document.save(pdf)
                document.close()

                evidence = pdf_evidence.inspect_pdf(pdf)

                self.assertEqual(evidence.pdf_status, "pdf_image_only")
                self.assertEqual(
                    [candidate.doi for candidate in evidence.dois],
                    ["10.1172/JCI200318868"],
                )


    def test_inspection_identifier_provenance(self) -> None:
        with self.subTest(case="inspection_extracts_normalized_pii_from_first_page_text"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "publisher-article.pdf"
                title = "Open Access Birdcage Coils for Microscopic Imaging"
                document = pymupdf.open()
                page = document.new_page()
                page.insert_text((72, 72), title, fontsize=16)
                page.insert_text(
                    (72, 120),
                    "PII: S 1352-8661 (00)00069-7",
                    fontsize=10,
                )
                document.set_metadata({"title": title})
                document.save(pdf)
                document.close()

                evidence = pdf_evidence.inspect_pdf(pdf)

                self.assertEqual(evidence.pii, "S1352866100000697")
                self.assertEqual(evidence.pdf_status, "ok")

        with self.subTest(case="inspection_extracts_arxiv_id_from_first_page_only"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "descriptive-name.pdf"
                _create_pdf_with_pages(
                    pdf,
                    [
                        "Reliable Sparse Imaging\narXiv:2509.16803v1 [q-bio.NC]",
                        "Reference: arXiv:1810.10973v2",
                    ],
                )

                evidence = pdf_evidence.inspect_pdf(pdf)

                self.assertEqual(evidence.arxiv_doi, "10.48550/arXiv.2509.16803")

        with self.subTest(case="metadata_can_override_shared_page_doi_demotion"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "shared-issue-page.pdf"
                preceding_doi = "10.1000/preceding"
                article_doi = "10.1000/current"
                title = "Current Article on a Shared Issue Page"
                document = pymupdf.open()
                page = document.new_page()
                page.insert_textbox(
                    pymupdf.Rect(72, 72, 540, 390),
                    (
                        "Previous article conclusion "
                        + "reliable scientific findings " * 45
                        + f" https://doi.org/{preceding_doi}"
                    ),
                    fontsize=8,
                )
                page.insert_text((72, 470), title, fontsize=18)
                page.insert_text(
                    (72, 520),
                    f"Abstract https://doi.org/{article_doi}",
                    fontsize=10,
                )
                document.set_metadata(
                    {
                        "title": title,
                        "subject": f"doi:{preceding_doi}",
                    }
                )
                document.save(pdf)
                document.close()

                evidence = pdf_evidence.inspect_pdf(pdf)
                candidates = {candidate.doi: candidate for candidate in evidence.dois}

                self.assertEqual(candidates[preceding_doi].source, "metadata:subject")
                self.assertEqual(candidates[preceding_doi].score, 120)
                self.assertGreater(
                    candidates[preceding_doi].score,
                    candidates[article_doi].score,
                )


    def test_cover_detection_boundary(self) -> None:
        with self.subTest(case="inspection_detects_leading_host_and_publisher_covers"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "covered.pdf"
                _create_pdf_with_pages(
                    pdf,
                    [
                        (
                            "See discussions, stats, and author profiles for this "
                            "publication at: https://www.researchgate.net/"
                            "publication/123456\n"
                            "DOI: 10.13140/RG.2.1.1234.5678\n"
                            "CITATIONS 10 READS 100\n"
                            "All content following this page was uploaded by an "
                            "author."
                        ),
                        (
                            "Citation: Journal of Reliable Imaging 12, 34 (2024)\n"
                            "View online: https://doi.org/10.1000/covered\n"
                            "View Table of Contents: https://example.test/"
                            "?ver=pdfcov\nPublished by the AIP Publishing"
                        ),
                        (
                            "Reliable Imaging Without Repository Cover Pages\n"
                            "A. Example and B. Researcher\n"
                            "Abstract\nThis article describes reliable imaging.\n"
                            "Introduction\nhttps://doi.org/10.1000/covered"
                        ),
                    ],
                )
                original = pdf.read_bytes()

                evidence = pdf_evidence.inspect_pdf(pdf)

                self.assertEqual(
                    [
                        (item.page_index, item.kind)
                        for item in evidence.inserted_cover_pages
                    ],
                    [(0, "researchgate"), (1, "aip-pdfcov")],
                )
                self.assertEqual(
                    evidence.title,
                    "Reliable Imaging Without Repository Cover Pages",
                )
                self.assertEqual(
                    evidence.pdf_status,
                    "pdf_cover_page_removal_needed",
                )
                self.assertTrue(evidence.needs_rewrite)
                self.assertEqual(pdf.read_bytes(), original)
                self.assertNotIn(
                    "10.13140/RG.2.1.1234.5678",
                    [item.doi for item in evidence.dois],
                )

        with self.subTest(case="ordinary_article_title_page_is_not_a_removable_cover"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "article.pdf"
                _create_pdf_with_pages(
                    pdf,
                    [
                        (
                            "Reliable Imaging Without Repository Cover Pages\n"
                            "A. Example and B. Researcher\n"
                            "Abstract\nThis article describes reliable imaging.\n"
                            "Introduction\nhttps://doi.org/10.1000/article"
                        ),
                        "Methods\nThe experiment used a calibrated phantom.",
                    ],
                )

                evidence = pdf_evidence.inspect_pdf(pdf)

                self.assertEqual(evidence.inserted_cover_pages, ())
                self.assertFalse(evidence.needs_rewrite)


    def test_cover_removal_preserves_article_pages(self) -> None:
        with self.subTest(case="rewrite_removes_detected_covers_and_preserves_article"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "covered.pdf"
                _create_pdf_with_pages(
                    pdf,
                    [
                        (
                            "HAL Id: hal-123456\n"
                            "https://hal.archives-ouvertes.fr/hal-123456\n"
                            "Submitted on 15 Mar 2020\n"
                            "HAL is a multi-disciplinary open access archive for "
                            "the deposit and dissemination of scientific research."
                        ),
                        (
                            "Reliable Article Preserved After Cover Removal\n"
                            "A. Example and B. Researcher\n"
                            "Abstract\nArticle body and results.\n"
                            "https://doi.org/10.1000/preserved"
                        ),
                        "References\n1. Example A. Reliable methods.",
                    ],
                )
                before = pdf_evidence.inspect_pdf(pdf)

                outcome = pdf_rewrite.rewrite_pdf(
                    pdf,
                    remove_page_indices=tuple(
                        item.page_index for item in before.inserted_cover_pages
                    ),
                )
                after = pdf_evidence.inspect_pdf(pdf)

                self.assertEqual(outcome.status, "pdf_cover_page_removed")
                self.assertTrue(outcome.changed)
                self.assertEqual(after.inserted_cover_pages, ())
                self.assertEqual(
                    after.title,
                    "Reliable Article Preserved After Cover Removal",
                )
                document = pymupdf.open(pdf)
                try:
                    self.assertEqual(len(document), 2)
                    self.assertIn(
                        "Reliable Article Preserved",
                        document[0].get_text(),
                    )
                finally:
                    document.close()

        with self.subTest(case="rewrite_refuses_to_remove_every_page"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "article.pdf"
                _create_repair_pdf(pdf)
                original = pdf.read_bytes()

                outcome = pdf_rewrite.rewrite_pdf(
                    pdf,
                    remove_page_indices=(0,),
                )

                self.assertEqual(outcome.status, "pdf_repair_failed")
                self.assertIn("every page", outcome.note)
                self.assertEqual(pdf.read_bytes(), original)


    def test_preprocessing_failure_blocks_bibliographic_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "covered.pdf"
            _create_pdf_with_pages(
                pdf,
                [
                    (
                        "See discussions, stats, and author profiles for this "
                        "publication at: https://www.researchgate.net/"
                        "publication/123456\n"
                        "All content following this page was uploaded by an "
                        "author."
                    ),
                    (
                        "Reliable Article After Preprocessing\n"
                        "A. Example and B. Researcher\n"
                        "Abstract\nArticle body and results.\n"
                        "https://doi.org/10.1000/preprocessed"
                    ),
                ],
            )
            original = pdf.read_bytes()

            with mock.patch.object(
                pdf_rewrite,
                "write_validated_copy",
                side_effect=OSError("simulated preprocessing failure"),
            ):
                evidence = pdf_evidence.inspect_pdf(pdf)

            self.assertEqual(evidence.pdf_status, "pdf_unrepairable")
            self.assertIsNone(evidence.title)
            self.assertEqual(evidence.dois, [])
            self.assertEqual(pdf.read_bytes(), original)

    def test_validated_copy_normalizes_repaired_pdf_without_changing_source(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "damaged.pdf"
            normalized = root / "normalized.pdf"
            _create_repair_pdf(source)
            damaged = re.sub(
                rb"startxref\s+\d+",
                b"startxref\n0",
                source.read_bytes(),
            )
            source.write_bytes(damaged)

            result = pdf_rewrite.write_validated_copy(source, normalized)

            self.assertEqual(result, normalized)
            self.assertEqual(source.read_bytes(), damaged)
            document = pymupdf.open(normalized)
            try:
                self.assertFalse(document.is_repaired)
                self.assertFalse(document.needs_pass)
                self.assertEqual(len(document), 1)
                self.assertIn(
                    "Reliable PDF Repair",
                    document[0].get_text(),
                )
            finally:
                document.close()

    def test_inspection_is_read_only_and_rewrite_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "encrypted.pdf"
            _create_repair_pdf(pdf, user_password="user-password")
            original = pdf.read_bytes()

            evidence = pdf_evidence.inspect_pdf(
                pdf,
                pdf_password="user-password",
            )

            self.assertEqual(evidence.pdf_status, "pdf_decryption_needed")
            self.assertEqual(pdf.read_bytes(), original)

            outcome = pdf_rewrite.rewrite_pdf(
                pdf,
                pdf_password="user-password",
            )

            self.assertEqual(outcome.status, "pdf_decrypted")
            self.assertNotEqual(pdf.read_bytes(), original)

    def test_explicit_rewrite_preserves_extracted_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "encrypted.pdf"
            _create_repair_pdf(pdf, user_password="user-password")

            before = pdf_evidence.inspect_pdf(
                pdf,
                pdf_password="user-password",
            )
            outcome = pdf_rewrite.rewrite_pdf(
                pdf,
                pdf_password="user-password",
            )
            after = pdf_evidence.inspect_pdf(pdf)

            self.assertEqual(outcome.status, "pdf_decrypted")
            self.assertEqual(before.title, after.title)
            self.assertEqual(
                [(item.doi, item.source, item.score) for item in before.dois],
                [(item.doi, item.source, item.score) for item in after.dois],
            )

    def test_encrypted_pdf_authentication_and_mutation(self) -> None:
        with self.subTest(case="dry_run_authenticates_without_rewriting_encrypted_pdf"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pdf = root / "encrypted.pdf"
                _create_repair_pdf(pdf, user_password="user-password")
                original = pdf.read_bytes()

                hints = RenamePDF.sniff_pdf_hints(
                    pdf,
                    commit=False,
                    pdf_password="user-password",
                )

                self.assertEqual(hints.pdf_status, "pdf_decryption_needed")
                self.assertEqual(hints.title, "Reliable PDF Repair for Scientific Papers")
                self.assertIn("10.1000/repair-test", [item.doi for item in hints.dois])
                self.assertEqual(pdf.read_bytes(), original)
                self.assertFalse((root / "Superseded").exists())

        with self.subTest(case="commit_decrypts_without_permanent_backup"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pdf = root / "encrypted.pdf"
                _create_repair_pdf(pdf, user_password="user-password")
                original_bytes = pdf.read_bytes()

                hints = RenamePDF.sniff_pdf_hints(
                    pdf,
                    commit=True,
                    pdf_password="user-password",
                )

                self.assertEqual(hints.pdf_status, "pdf_decrypted")
                self.assertNotEqual(pdf.read_bytes(), original_bytes)
                self.assertFalse((root / "Superseded").exists())
                self.assertFalse(any(root.glob(".*renamepdf-rollback*")))

                repaired = pymupdf.open(pdf)
                try:
                    self.assertFalse(repaired.needs_pass)
                    self.assertFalse(repaired.is_encrypted)
                    self.assertIsNone((repaired.metadata or {}).get("encryption"))
                    self.assertEqual(len(repaired), 1)
                finally:
                    repaired.close()

        with self.subTest(case="password_failure_is_reported_without_modifying_pdf"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pdf = root / "encrypted.pdf"
                _create_repair_pdf(pdf, user_password="user-password")
                original = pdf.read_bytes()

                hints = RenamePDF.sniff_pdf_hints(
                    pdf,
                    commit=True,
                    pdf_password="wrong-password",
                )

                self.assertEqual(hints.pdf_status, "pdf_password_required")
                self.assertIn("RENAMEPDF_PDF_PASSWORD", hints.pdf_note)
                self.assertEqual(pdf.read_bytes(), original)
                self.assertFalse((root / "Superseded").exists())


    def test_signed_rewrite_is_blocked_consistently_before_matching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "signed-covered.pdf"
            _create_pdf_with_pages(
                pdf,
                [
                    (
                        "See discussions, stats, and author profiles for this "
                        "publication at: https://www.researchgate.net/"
                        "publication/123456\n"
                        "All content following this page was uploaded by an "
                        "author."
                    ),
                    (
                        "Signed Article Behind a Repository Cover\n"
                        "A. Example and B. Researcher\nAbstract\n"
                        "https://doi.org/10.1000/signed-covered"
                    ),
                ],
                signature_widget=True,
            )
            original = pdf.read_bytes()

            dry_run = RenamePDF.sniff_pdf_hints(pdf, commit=False)
            committed = RenamePDF.sniff_pdf_hints(pdf, commit=True)

            for evidence in (dry_run, committed):
                self.assertEqual(
                    evidence.pdf_status,
                    "pdf_signed_rewrite_blocked",
                )
                self.assertEqual(evidence.dois, [])
                self.assertIsNone(evidence.title)
                self.assertIn("digital signature", evidence.pdf_note)
            self.assertEqual(pdf.read_bytes(), original)

    def test_install_failure_restores_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "encrypted.pdf"
            _create_repair_pdf(pdf, user_password="user-password")
            original_bytes = pdf.read_bytes()
            real_copy = pdf_rewrite.create_exclusive_copy

            def fail_repaired_install(source, target, **kwargs):
                if ".repairing" in Path(source).name:
                    raise OSError("simulated installation failure")
                return real_copy(source, target)

            with mock.patch.object(
                pdf_rewrite,
                "create_exclusive_copy",
                side_effect=fail_repaired_install,
            ):
                outcome = pdf_rewrite.rewrite_pdf(
                    pdf,
                    pdf_password="user-password",
                )

            self.assertEqual(outcome.status, "pdf_repair_failed")
            self.assertEqual(pdf.read_bytes(), original_bytes)
            self.assertFalse(any(root.glob(".*renamepdf-rollback*")))

    def test_validation_failure_restores_original_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "encrypted.pdf"
            _create_repair_pdf(pdf, user_password="user-password")
            original_bytes = pdf.read_bytes()
            real_validate = pdf_rewrite._validate_pdf

            def fail_installed_validation(path: Path, expected_pages: int) -> None:
                if Path(path).resolve() == pdf.resolve():
                    raise ValueError("simulated installed-PDF validation failure")
                real_validate(path, expected_pages)

            with mock.patch.object(
                pdf_rewrite,
                "_validate_pdf",
                side_effect=fail_installed_validation,
            ):
                outcome = pdf_rewrite.rewrite_pdf(
                    pdf,
                    pdf_password="user-password",
                )

            self.assertEqual(outcome.status, "pdf_repair_failed")
            self.assertEqual(pdf.read_bytes(), original_bytes)
            self.assertFalse(any(root.glob(".*renamepdf-rollback*")))
            self.assertFalse(any(root.glob(".*installed-marker*")))

    def test_install_does_not_clobber_canonical_path_that_reappears(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "encrypted.pdf"
            _create_repair_pdf(pdf, user_password="user-password")
            original_bytes = pdf.read_bytes()
            concurrent_bytes: bytes | None = None
            real_copy = pdf_rewrite.create_exclusive_copy

            def race_install(source, target):
                nonlocal concurrent_bytes
                if ".repairing" in Path(source).name:
                    concurrent = root / "concurrent.pdf"
                    _create_repair_pdf(concurrent, title="Concurrent canonical PDF")
                    concurrent_bytes = concurrent.read_bytes()
                    concurrent.replace(pdf)
                return real_copy(source, target)

            with mock.patch.object(
                pdf_rewrite,
                "create_exclusive_copy",
                side_effect=race_install,
            ):
                outcome = pdf_rewrite.rewrite_pdf(
                    pdf,
                    pdf_password="user-password",
                )

            self.assertEqual(outcome.status, "pdf_repair_failed")
            self.assertEqual(pdf.read_bytes(), concurrent_bytes)
            self.assertNotEqual(pdf.read_bytes(), original_bytes)
            rollbacks = list(root.glob(".*.renamepdf-rollback"))
            self.assertEqual(len(rollbacks), 1)
            self.assertEqual(rollbacks[0].read_bytes(), original_bytes)

    def test_post_install_cleanup_failure_keeps_success_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "encrypted.pdf"
            _create_repair_pdf(pdf, user_password="user-password")
            original_bytes = pdf.read_bytes()
            real_unlink = Path.unlink

            def fail_rollback_unlink(path: Path, *args, **kwargs):
                if path.name.endswith(".renamepdf-rollback"):
                    raise PermissionError("simulated cleanup failure")
                return real_unlink(path, *args, **kwargs)

            with mock.patch.object(
                Path,
                "unlink",
                new=fail_rollback_unlink,
            ):
                outcome = pdf_rewrite.rewrite_pdf(
                    pdf,
                    pdf_password="user-password",
                )

            installed_markers = list(
                root.glob(".*.renamepdf-installed-marker")
            )
            rollbacks = list(root.glob(".*.renamepdf-rollback"))
            self.assertEqual(outcome.status, "pdf_decrypted")
            self.assertTrue(outcome.changed)
            self.assertIn("cleanup failed", outcome.note)
            self.assertNotEqual(pdf.read_bytes(), original_bytes)
            self.assertEqual(len(installed_markers), 1)
            self.assertEqual(len(rollbacks), 1)
            marker = pdf_rewrite._parse_marker(installed_markers[0])
            self.assertIsNotNone(marker)
            self.assertEqual(marker.operation, "install")
            self.assertEqual(marker.sha256, pdf_rewrite._sha256_file(pdf))

            recovery = pdf_rewrite.recover_interrupted_rewrites(root)

            self.assertEqual(recovery[0].status, "pdf_recovery_cleanup_needed")
            self.assertFalse(recovery[0].blocks_run)

            cleaned = pdf_rewrite.recover_interrupted_rewrites(
                root,
                commit=True,
            )
            self.assertEqual(cleaned[0].status, "pdf_recovery_cleanup_completed")
            self.assertFalse(installed_markers[0].exists())
            self.assertFalse(rollbacks[0].exists())

    def test_concurrent_canonical_replacement_preserves_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "encrypted.pdf"
            _create_repair_pdf(pdf, user_password="user-password")
            original_bytes = pdf.read_bytes()
            concurrent_bytes: bytes | None = None
            real_write_marker = pdf_rewrite._write_exclusive_bytes

            def replace_before_marker(destination, payload):
                nonlocal concurrent_bytes
                if destination.name.endswith(".renamepdf-installed-marker"):
                    concurrent = destination.parent / "concurrent.pdf"
                    _create_repair_pdf(
                        concurrent,
                        title="Concurrent Canonical Replacement",
                    )
                    concurrent_bytes = concurrent.read_bytes()
                    concurrent.replace(pdf)
                return real_write_marker(destination, payload)

            with mock.patch.object(
                pdf_rewrite,
                "_write_exclusive_bytes",
                side_effect=replace_before_marker,
            ):
                outcome = pdf_rewrite.rewrite_pdf(
                    pdf,
                    pdf_password="user-password",
                )

            rollbacks = list(root.glob(".*.renamepdf-rollback"))
            markers = list(root.glob(".*.renamepdf-installed-marker"))
            self.assertEqual(outcome.status, "pdf_repair_failed")
            self.assertIn("canonical path changed", outcome.note)
            self.assertEqual(pdf.read_bytes(), concurrent_bytes)
            self.assertEqual(len(rollbacks), 1)
            self.assertEqual(rollbacks[0].read_bytes(), original_bytes)
            self.assertEqual(len(markers), 0)

            recovery = pdf_rewrite.recover_interrupted_rewrites(
                root,
                commit=True,
            )
            self.assertEqual(recovery[0].status, "pdf_recovery_ambiguous")
            self.assertTrue(recovery[0].blocks_run)

    def test_failed_restore_marker_validation_preserves_original_rollback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "paper.pdf"
            replacement = root / "replacement.pdf"
            concurrent = root / "concurrent.pdf"
            _create_repair_pdf(pdf, title="Original PDF")
            _create_repair_pdf(replacement, title="Replacement PDF")
            _create_repair_pdf(concurrent, title="Concurrent PDF")
            original_bytes = pdf.read_bytes()
            concurrent_bytes = concurrent.read_bytes()
            real_validate = pdf_rewrite._validate_pdf
            real_write_marker = pdf_rewrite._write_exclusive_bytes

            def fail_installed_validation(path: Path, expected_pages: int) -> None:
                if Path(path) == pdf:
                    raise RuntimeError("force installation rollback")
                real_validate(path, expected_pages)

            def replace_restored_before_marker(
                destination: Path,
                payload: bytes,
            ) -> Path:
                if b"operation=restore" in payload:
                    concurrent.replace(pdf)
                return real_write_marker(destination, payload)

            with (
                mock.patch.object(
                    pdf_rewrite,
                    "_validate_pdf",
                    side_effect=fail_installed_validation,
                ),
                mock.patch.object(
                    pdf_rewrite,
                    "_write_exclusive_bytes",
                    side_effect=replace_restored_before_marker,
                ),
                self.assertRaisesRegex(RuntimeError, "rollback preserved") as failure,
            ):
                pdf_rewrite._install_replacement(pdf, replacement, 1)

            self.assertIsNone(failure.exception.source_token)
            rollbacks = list(root.glob(".*.renamepdf-rollback"))
            markers = list(root.glob(".*.renamepdf-installed-marker"))
            self.assertEqual(pdf.read_bytes(), concurrent_bytes)
            self.assertEqual(len(rollbacks), 1)
            self.assertEqual(rollbacks[0].read_bytes(), original_bytes)
            self.assertEqual(len(markers), 1)

            recovery = pdf_rewrite.recover_interrupted_rewrites(root, commit=True)
            self.assertEqual(recovery[0].status, "pdf_recovery_ambiguous")
            self.assertEqual(rollbacks[0].read_bytes(), original_bytes)

    def test_commit_repairs_without_permanent_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "damaged.pdf"
            _create_repair_pdf(pdf)
            damaged = re.sub(
                rb"startxref\s+\d+",
                b"startxref\n0",
                pdf.read_bytes(),
            )
            pdf.write_bytes(damaged)

            before = pymupdf.open(pdf)
            try:
                self.assertTrue(before.is_repaired)
            finally:
                before.close()

            hints = RenamePDF.sniff_pdf_hints(pdf, commit=True)

            self.assertEqual(hints.pdf_status, "pdf_repaired")
            self.assertIn("10.1000/repair-test", [item.doi for item in hints.dois])
            self.assertFalse((root / "Superseded").exists())
            self.assertFalse(any(root.glob(".*renamepdf-rollback*")))

            repaired = pymupdf.open(pdf)
            try:
                self.assertFalse(repaired.is_repaired)
                self.assertEqual(len(repaired), 1)
            finally:
                repaired.close()

    def test_signature_flags_block_rewriting(self) -> None:
        class SignedDocument:
            @staticmethod
            def get_sigflags() -> int:
                return 3

        self.assertTrue(pdf_rewrite.pdf_has_signatures(SignedDocument()))


if __name__ == "__main__":
    unittest.main()
