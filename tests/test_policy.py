"""Focused tests consolidated by validation surface."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import collection_mutation
import document_classification
import filename_policy
import pdf_evidence
import record_matching


# Failed-document classification

class FailedDocumentClassificationTests(unittest.TestCase):
    def test_preprint_provenance(self) -> None:
        with self.subTest(case="inline_arxiv_citation_is_not_a_preprint_classification"):
            result = document_classification.classify_failed_pdf(
                Path("journal.pdf"), title="Reliable synthetic imaging research",
                text_samples=("Prior work: Example et al. arXiv:2509.16803",),
            )
            self.assertEqual(result.category, "Others")

        with self.subTest(case="standalone_arxiv_stamp_still_classifies_a_preprint"):
            result = document_classification.classify_failed_pdf(
                Path("download.pdf"), title="Reliable synthetic imaging research",
                text_samples=("arXiv:2509.16803v1 [q-bio.NC]",),
            )
            self.assertEqual(result.category, "Preprints")

        with self.subTest(case="preprint_identifier_has_high_confidence"):
            result = document_classification.classify_failed_pdf(
                Path("download.pdf"),
                doi_candidates=[
                    ("10.1101/2025.01.01.123456", "metadata:subject", 120)
                ],
            )
            self.assertEqual(result.category, "Preprints")


    def test_proceedings_evidence(self) -> None:
        with self.subTest(case="ismrm_filename_is_a_proceedings_document"):
            result = document_classification.classify_failed_pdf(
                Path("ISMRM2019_0434.pdf")
            )
            self.assertEqual(result.category, "Proceedings")

        with self.subTest(case="conference_paper_label_is_detected"):
            result = document_classification.classify_failed_pdf(
                Path("2015_HP_13C_cewki_Wosik.pdf"),
                title="Cryogenic Receive-only Coil Conference Paper",
            )
            self.assertEqual(result.category, "Proceedings")

        with self.subTest(case="ismrm_proceedings_footer_overrides_reference_noise"):
            result = document_classification.classify_failed_pdf(
                Path("2008_2_4_el_array.pdf"),
                text_samples=(
                    "ResearchGate cover page.\f"
                    "Related doctoral dissertation material. "
                    "443 Proc. Intl. Soc. Mag. Reson. Med. 16 (2008)",
                ),
            )
            self.assertEqual(result.category, "Proceedings")
            self.assertIn("ISMRM", result.reason)

        with self.subTest(case="numbered_digital_poster_is_a_proceedings_document"):
            result = document_classification.classify_failed_pdf(
                Path("4707.pdf"),
                text_samples=(
                    "Program No: 4707 Poster Type: Digital Poster "
                    "Authors: Example Author",
                ),
            )
            self.assertEqual(result.category, "Proceedings")

        with self.subTest(case="numeric_structured_ismrm_abstract_is_detected"):
            result = document_classification.classify_failed_pdf(
                Path("00916.pdf"),
                text_samples=(
                    "INTRODUCTION\nRESULTS\nDISCUSSION\nCONCLUSION\nREFERENCES\n"
                    "Prior work appeared in Proc. ISMRM, 11, p. 4982 (2003).",
                ),
            )
            self.assertEqual(result.category, "Proceedings")

        with self.subTest(case="legacy_p_number_structured_abstract_is_detected"):
            result = document_classification.classify_failed_pdf(
                Path("p2035.pdf"),
                text_samples=(
                    "INTRODUCTION\nMETHODS AND MATERIALS\nRESULTS\nREFERENCES",
                ),
            )
            self.assertEqual(result.category, "Proceedings")

        with self.subTest(case="eucap_front_matter_is_a_proceedings_document"):
            result = document_classification.classify_failed_pdf(
                Path("PID5615597.pdf"),
                text_samples=(
                    "Volume coil for MRI based on metasurface. "
                    "13th European Conference on Antennas and Propagation "
                    "(EuCAP 2019), Mar 2019, Cracovie, Poland.",
                ),
            )
            self.assertEqual(result.category, "Proceedings")
            self.assertIn("EuCAP", result.reason)


    def test_dissertation_evidence(self) -> None:
        with self.subTest(case="dissertation_front_matter_is_detected"):
            result = document_classification.classify_failed_pdf(
                Path("Dis_Malzacher_komplett.pdf"),
                text_samples=(
                    "A dissertation submitted in partial fulfillment of "
                    "the requirements for the degree of Doctor of Philosophy",
                ),
            )
            self.assertEqual(result.category, "Dissertations")

        with self.subTest(case="dissertation_filename_convention_is_detected"):
            result = document_classification.classify_failed_pdf(
                Path("Dis_Malzacher_komplett.pdf")
            )
            self.assertEqual(result.category, "Dissertations")


    def test_classification_requires_document_evidence(self) -> None:
        with self.subTest(case="journal_article_citing_ismrm_is_not_reclassified"):
            result = document_classification.classify_failed_pdf(
                Path("Author_Journal24_Article.pdf"),
                text_samples=(
                    "INTRODUCTION\nRESULTS\nDISCUSSION\nREFERENCES\n"
                    "Prior work appeared in Proc. ISMRM, 11, p. 4982 (2003).",
                ),
            )
            self.assertEqual(result.category, "Others")

        with self.subTest(case="unsubstantiated_document_falls_back_to_others"):
            result = document_classification.classify_failed_pdf(
                Path("Weekly Eyewash equipment Inspection Sheet.pdf"),
                title="Weekly Eyewash Equipment Inspection Sheet",
            )
            self.assertEqual(result.category, "Others")

        with self.subTest(case="journal_filename_with_ismrm_keyword_is_not_preliminary"):
            path = Path(
                "JelescuI_Schilling_MagneticResonanceInMed25_"
                "Recommendations_Preclinical_Ismrm_Study.pdf"
            )
            self.assertIsNone(
                document_classification.classify_preliminary_path(path)
            )


    def test_vendor_evidence(self) -> None:
        with self.subTest(case="vendor_whitepaper_filename_is_detected"):
            result = document_classification.classify_failed_pdf(
                Path("mr_achieva_tx_whitepaper_multitransmit.pdf")
            )
            self.assertEqual(result.category, "VendorDocuments")

        with self.subTest(case="vendor_document_id_requires_vendor_corroboration"):
            result = document_classification.classify_failed_pdf(
                Path("bandwith_in_mri_1800000000456951.pdf"),
                text_samples=("MAGNETOM Flash www.siemens.com/magnetom-world",),
            )
            self.assertEqual(result.category, "VendorDocuments")


# Canonical and duplicate placement

def _set_mtime(path: Path, nanoseconds: int) -> None:
    os.utime(path, ns=(nanoseconds, nanoseconds))


class DuplicatePolicyTests(unittest.TestCase):
    def test_canonical_copy_follows_modification_time(self) -> None:
        with self.subTest(case="newer_incoming_copy_becomes_canonical"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                canonical = root / "Paper.pdf"
                incoming = root / "incoming.pdf"
                canonical.write_bytes(b"old")
                incoming.write_bytes(b"new")
                _set_mtime(canonical, 1_000_000_000)
                _set_mtime(incoming, 2_000_000_000)
                outcome = collection_mutation.place_canonical(
                    incoming,
                    canonical,
                    root / "Duplicated",
                )

                self.assertEqual(canonical.read_bytes(), b"new")
                self.assertEqual((root / "Duplicated" / canonical.name).read_bytes(), b"old")
                self.assertEqual(outcome.action, "canonical_replaced")
                self.assertEqual(outcome.target, canonical)

        with self.subTest(case="older_incoming_copy_is_preserved_as_duplicate"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                canonical = root / "Paper.pdf"
                incoming = root / "incoming.pdf"
                canonical.write_bytes(b"new")
                incoming.write_bytes(b"old")
                _set_mtime(canonical, 2_000_000_000)
                _set_mtime(incoming, 1_000_000_000)
                outcome = collection_mutation.place_canonical(
                    incoming,
                    canonical,
                    root / "Duplicated",
                )

                self.assertEqual(canonical.read_bytes(), b"new")
                self.assertEqual((root / "Duplicated" / canonical.name).read_bytes(), b"old")
                self.assertEqual(outcome.action, "duplicate")


    def test_reconciliation_swaps_files_without_deleting_either(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate_dir = root / "Duplicated"
            duplicate_dir.mkdir()
            canonical = root / "Paper.pdf"
            duplicate = duplicate_dir / "Paper.pdf"
            canonical.write_bytes(b"old")
            duplicate.write_bytes(b"new")
            _set_mtime(canonical, 1_000_000_000)
            _set_mtime(duplicate, 2_000_000_000)

            changed = collection_mutation.prefer_newest_as_canonical(
                canonical,
                duplicate,
            )

            self.assertTrue(changed)
            self.assertEqual(canonical.read_bytes(), b"new")
            self.assertEqual(duplicate.read_bytes(), b"old")
            self.assertFalse(any(path.name.startswith(".") for path in duplicate_dir.iterdir()))


# Filename generation

def _meta() -> record_matching.PaperMeta:
    first = record_matching.Author(given="Élodie", family="García")
    last = record_matching.Author(given="B", family="Smith")
    return record_matching.PaperMeta(
        title="Reliable Quantitative Imaging with Magnetic Resonance",
        year=2024,
        journal_abbrev="Magn Reson Med",
        first_author=first,
        authors=[first, last],
        corresponding_author=last,
        doi="10.1000/reliable",
        pmid=None,
        author_keywords=["quantitative imaging"],
        mesh_keywords=[],
    )


class FilenamePolicyTests(unittest.TestCase):
    def test_portable_and_bounded_filenames(self) -> None:
        with self.subTest(case="bibliographic_filename_is_portable"):
            self.assertEqual(
                filename_policy.make_filename(_meta()),
                "GarciaE_Smith_MagnResonMed24_QuantitativeImaging.pdf",
            )

        with self.subTest(case="windows_reserved_names_are_avoided_case_insensitively"):
            for stem in ("CON", "nul", "Com1", "Lpt9"):
                with self.subTest(stem=stem):
                    self.assertEqual(
                        filename_policy.filesystem_safe_pdf_name(stem),
                        f"{stem}_paper.pdf",
                    )

        with self.subTest(case="forbidden_characters_are_removed"):
            self.assertEqual(
                filename_policy.filesystem_safe_pdf_name('A:B/C*D? "Paper"'),
                "A_B_C_D_Paper.pdf",
            )

        with self.subTest(case="complete_filename_respects_utf8_byte_limit"):
            for stem in ("x" * 300, "研究" * 200):
                with self.subTest(stem=stem[:8]):
                    result = filename_policy.filesystem_safe_pdf_name(stem)

                    self.assertTrue(result.endswith(".pdf"))
                    self.assertLessEqual(
                        len(result.encode("utf-8")),
                        filename_policy.MAX_FILENAME_BYTES,
                    )

        with self.subTest(case="doi_disambiguation_is_stable_safe_and_byte_bounded"):
            first = filename_policy.disambiguate_pdf_name(
                "Paper" * 100 + ".pdf",
                "10.1000/example",
            )
            second = filename_policy.disambiguate_pdf_name(
                "Paper" * 100 + ".pdf",
                "10.1000/example",
            )

            self.assertEqual(first, second)
            self.assertRegex(first, r"_doi[0-9a-f]{10}\.pdf$")
            self.assertLessEqual(
                len(first.encode("utf-8")),
                filename_policy.MAX_FILENAME_BYTES,
            )


    def test_arxiv_journal_fallback(self) -> None:
        with self.subTest(case="arxiv_doi_replaces_missing_journal_label"):
            meta = _meta()
            meta.journal_abbrev = None
            meta.doi = "10.48550/arXiv.2509.16803"

            self.assertIn("_arXiv24_", filename_policy.make_filename(meta))

        with self.subTest(case="explicit_arxiv_evidence_replaces_missing_journal_label"):
            meta = _meta()
            meta.journal_abbrev = None

            self.assertIn(
                "_arXiv24_",
                filename_policy.make_filename(meta, is_arxiv=True),
            )


# Record matching

def _record(
    *,
    title: str = "Reliable Quantitative Imaging with Magnetic Resonance",
    year: int = 2024,
    doi: str | None = "10.1000/reliable",
    family: str = "Example",
    journal: str | None = "Magn Reson Med",
    keywords: list[str] | None = None,
) -> record_matching.PaperMeta:
    author = record_matching.Author(given="A", family=family)
    return record_matching.PaperMeta(
        title=title,
        year=year,
        journal_abbrev=journal,
        first_author=author,
        authors=[author],
        corresponding_author=author,
        doi=doi,
        pmid=None,
        author_keywords=keywords or [],
        mesh_keywords=[],
    )


class DoiDecisionTests(unittest.TestCase):
    def test_identifier_and_title_contradictions(self) -> None:
        with self.subTest(case="notice_cannot_match_original_with_missing_or_different_doi"):
            title = "Synthetic inhibitor X reverses fibrosis in a murine model"
            for prefix in ("Retraction Note", "Correction", "Erratum", "Expression of Concern"):
                for doi in (None, "10.5555/original.0001"):
                    with self.subTest(prefix=prefix, doi=doi):
                        record = record_matching.ProviderRecord("pubmed", _record(title=title, doi=doi))
                        notice = f"{prefix}: {title}"
                        decision = record_matching.decide_doi_match(
                            pdf_evidence.DoiCandidate("10.5555/notice.0001", "link:p1", 115), notice, [record],
                        )
                        self.assertEqual(decision.outcome, "rejected")
                        self.assertEqual(record_matching.decide_title_match(notice, None, [record]).outcome, "rejected")
            notice_record = record_matching.ProviderRecord("crossref", _record(title=f"Retraction Note: {title}", doi="10.5555/notice.0001"))
            decision = record_matching.decide_doi_match(
                pdf_evidence.DoiCandidate("10.5555/notice.0001", "link:p1", 115), notice_record.meta.title, [notice_record],
            )
            self.assertEqual(decision.outcome, "accepted")

        with self.subTest(case="exact_identifier_contradiction_boundary_is_explicit"):
            candidate = pdf_evidence.DoiCandidate("10.1000/reliable", "metadata:doi", 120)
            record = record_matching.ProviderRecord("crossref", _record())
            for score, expected in ((0.249, "rejected"), (0.25, "accepted"), (0.38, "accepted")):
                with self.subTest(score=score), mock.patch.object(record_matching, "title_similarity", return_value=score):
                    result = record_matching.decide_doi_match(candidate, record.meta.title, [record])
                    self.assertEqual(result.outcome, expected)

        with self.subTest(case="pii_has_additional_corroboration_beyond_the_doi_veto"):
            title = "Quantitative optical imaging of plant growth under temperature stress"
            meta = _record(title="Quantitative magnetic resonance imaging of human liver disease")
            records = [record_matching.ProviderRecord("crossref", meta)]
            similarity = record_matching.title_similarity(title, meta.title)
            self.assertGreater(similarity, record_matching.IDENTIFIER_TITLE_CONTRADICTION_SCORE)
            self.assertLess(similarity, record_matching.LOW_SOURCE_TITLE_CONFIRM_SCORE)
            doi = record_matching.decide_doi_match(
                pdf_evidence.DoiCandidate(meta.doi, "metadata:doi", 120), title, records,
            )
            pii = record_matching.decide_pii_match("S1234567824999999", title, records)
            self.assertEqual(doi.outcome, "accepted")
            self.assertEqual(pii.outcome, "rejected")

        with self.subTest(case="usable_contradicting_title_rejects_every_high_score_source"):
            record = record_matching.ProviderRecord("crossref", _record())
            unrelated_title = (
                "Cellular Metabolism and Immune Signaling in Human Liver Disease"
            )

            for source, score in (
                ("metadata:dc:identifier", 120),
                ("link:p1", 115),
                ("filename", 85),
            ):
                with self.subTest(source=source):
                    decision = record_matching.decide_doi_match(
                        pdf_evidence.DoiCandidate(
                            "10.1000/reliable",
                            source,
                            score,
                        ),
                        unrelated_title,
                        [record],
                    )

                    self.assertEqual(decision.outcome, "rejected")
                    self.assertIsNone(decision.resolved)
                    self.assertIn("title similarity", decision.reason)


    def test_doi_priority_controls_title_requirements(self) -> None:
        with self.subTest(case="matching_doi_without_title_hint_still_accepts"):
            decision = record_matching.decide_doi_match(
                pdf_evidence.DoiCandidate(
                    "10.1000/reliable",
                    "metadata:dc:identifier",
                    120,
                ),
                None,
                [record_matching.ProviderRecord("crossref", _record())],
            )

            self.assertEqual(decision.outcome, "accepted")
            self.assertIsNotNone(decision.resolved)

        with self.subTest(case="demoted_leading_article_doi_still_needs_title_confirmation"):
            decision = record_matching.decide_doi_match(
                pdf_evidence.DoiCandidate(
                    "10.1000/reliable",
                    "text:p1",
                    pdf_evidence.LEADING_ARTICLE_DOI_SCORE,
                ),
                "Reliable Quantitative Imaging with Magnetic Resonance",
                [record_matching.ProviderRecord("crossref", _record())],
            )

            self.assertEqual(decision.outcome, "accepted")
            self.assertIsNotNone(decision.resolved)
            self.assertIn("Low-priority DOI source", decision.reason)


    def test_provider_conflict_and_corroboration(self) -> None:
        with self.subTest(case="material_provider_conflict_is_rejected"):
            candidate = pdf_evidence.DoiCandidate(
                "10.1000/reliable",
                "metadata:subject",
                120,
            )
            records = [
                record_matching.ProviderRecord("pubmed", _record(), "123"),
                record_matching.ProviderRecord(
                    "crossref",
                    _record(
                        title="Unrelated Findings from a Different Scientific Study",
                        family="Different",
                    ),
                ),
            ]

            decision = record_matching.decide_doi_match(
                candidate,
                "Reliable Quantitative Imaging with Magnetic Resonance",
                records,
            )

            self.assertEqual(decision.outcome, "conflicting_provider_records")
            self.assertIsNone(decision.resolved)

        with self.subTest(case="unrelated_provider_result_cannot_veto_matching_doi"):
            candidate = pdf_evidence.DoiCandidate(
                "10.1109/TMTT.2018.2869603",
                "metadata:subject",
                120,
            )
            records = [
                record_matching.ProviderRecord(
                    "pubmed",
                    _record(
                        title="Unrelated Findings from a Different Scientific Study",
                        year=2025,
                        doi="10.1103/lt46-nrh4",
                        family="Persoons",
                    ),
                    "123",
                ),
                record_matching.ProviderRecord(
                    "crossref",
                    _record(
                        title=(
                            "Association and Dissociation of Optimal Noise and "
                            "Input Impedance for Low-Noise Amplifiers"
                        ),
                        year=2018,
                        doi="10.1109/TMTT.2018.2869603",
                        family="Johansen",
                    ),
                ),
            ]

            decision = record_matching.decide_doi_match(
                candidate,
                (
                    "Association and Dissociation of Optimal Noise and "
                    "Input Impedance for Low-Noise Amplifiers"
                ),
                records,
            )

            self.assertEqual(decision.outcome, "accepted")
            self.assertIsNotNone(decision.resolved)
            self.assertEqual(decision.resolved.provider, "crossref")

        with self.subTest(case="incomplete_outlier_does_not_override_agreeing_records"):
            candidate = pdf_evidence.DoiCandidate(
                "10.1000/reliable",
                "metadata:subject",
                120,
            )
            incomplete = _record(
                title="Unrelated Findings from a Different Scientific Study",
                year=0,
                family="Unknown",
                journal=None,
            )
            records = [
                record_matching.ProviderRecord("pubmed", _record(), "123"),
                record_matching.ProviderRecord("crossref", _record(keywords=["MRI"])),
                record_matching.ProviderRecord("datacite", incomplete),
            ]

            decision = record_matching.decide_doi_match(
                candidate,
                "Reliable Quantitative Imaging with Magnetic Resonance",
                records,
            )

            self.assertEqual(decision.outcome, "accepted")
            self.assertIsNotNone(decision.resolved)
            self.assertIn("providers agreed", decision.reason)

        with self.subTest(case="provider_trust_is_only_a_final_tie_break"):
            candidate = pdf_evidence.DoiCandidate(
                "10.1000/reliable",
                "metadata:subject",
                120,
            )
            records = [
                record_matching.ProviderRecord("pubmed", _record(), "123"),
                record_matching.ProviderRecord(
                    "crossref",
                    _record(keywords=["MRI", "quantitative", "imaging"]),
                ),
            ]

            decision = record_matching.decide_doi_match(
                candidate,
                "Reliable Quantitative Imaging with Magnetic Resonance",
                records,
            )

            self.assertEqual(decision.resolved.provider, "crossref")


class PiiDecisionTests(unittest.TestCase):
    def test_pii_identity_and_title_boundaries(self) -> None:
        with self.subTest(case="exact_pii_rejects_unrelated_usable_title"):
            decision = record_matching.decide_pii_match(
                "S1090780724000012",
                "Cellular Metabolism and Immune Signaling in Human Liver Disease",
                [record_matching.ProviderRecord("crossref", _record())],
            )

            self.assertEqual(decision.outcome, "rejected")
            self.assertIsNone(decision.resolved)
            self.assertIn("title", decision.reason.lower())

        with self.subTest(case="exact_pii_without_title_hint_still_accepts"):
            decision = record_matching.decide_pii_match(
                "S1090780724000012",
                None,
                [record_matching.ProviderRecord("crossref", _record())],
            )

            self.assertEqual(decision.outcome, "accepted")
            self.assertIsNotNone(decision.resolved)
            self.assertEqual(decision.resolved.source, "filename-pii")

        with self.subTest(case="exact_pii_accepts_moderately_mis_extracted_title"):
            decision = record_matching.decide_pii_match(
                "S1090780724000012",
                "Rel1able Quant1tat1ve Imag1ng w1th Magnet1c Resonance",
                [record_matching.ProviderRecord("crossref", _record())],
            )

            self.assertEqual(decision.outcome, "accepted")
            self.assertIsNotNone(decision.resolved)
            self.assertGreater(decision.resolved.confidence, 0.85)

        with self.subTest(case="exact_pii_rejects_contained_short_title_fragment"):
            decision = record_matching.decide_pii_match(
                "S1090780724000012",
                "Reliable Quantitative Imaging",
                [record_matching.ProviderRecord("crossref", _record())],
            )

            self.assertEqual(decision.outcome, "rejected")
            self.assertIsNone(decision.resolved)
            self.assertIn("short fragment", decision.reason)

        with self.subTest(case="exact_pii_shared_by_distinct_records_is_rejected"):
            decision = record_matching.decide_pii_match(
                "S1090780724000012",
                None,
                [
                    record_matching.ProviderRecord(
                        "crossref",
                        _record(doi="10.1000/one"),
                    ),
                    record_matching.ProviderRecord(
                        "crossref",
                        _record(
                            title="Cellular Metabolism in Human Liver Disease",
                            doi="10.1000/two",
                            family="Different",
                        ),
                    ),
                ],
            )

            self.assertEqual(decision.outcome, "rejected")
            self.assertIsNone(decision.resolved)
            self.assertIn("multiple publications", decision.reason)


class TitleDecisionTests(unittest.TestCase):
    def test_consensus_and_distinct_record_ambiguity(self) -> None:
        with self.subTest(case="same_work_from_multiple_providers_is_not_ambiguous"):
            records = [
                record_matching.ProviderRecord("pubmed", _record(), "123"),
                record_matching.ProviderRecord("crossref", _record()),
                record_matching.ProviderRecord(
                    "datacite",
                    _record(doi=None),
                ),
            ]

            decision = record_matching.decide_title_match(
                "Reliable Quantitative Imaging with Magnetic Resonance",
                None,
                records,
            )

            self.assertEqual(decision.outcome, "accepted")
            self.assertIn("providers agreed", decision.reason)

        with self.subTest(case="provider_consensus_rejects_duplicate_doi_outlier"):
            accepted_doi = (
                "10.1002/(sici)1522-2586(199911)10:5"
                "<590::aid-jmri2>3.0.co;2-s"
            )
            outlier_doi = (
                "10.1002/(sici)1522-2586(199911)10:5"
                "<590::aid-jmri2>3.3.co;2-j"
            )
            records = [
                record_matching.ProviderRecord(
                    "pubmed",
                    _record(doi=accepted_doi),
                    "10548767",
                ),
                record_matching.ProviderRecord(
                    "crossref",
                    _record(doi=accepted_doi),
                ),
                record_matching.ProviderRecord(
                    "crossref",
                    _record(doi=outlier_doi),
                ),
            ]

            decision = record_matching.decide_title_match(
                "Reliable Quantitative Imaging with Magnetic Resonance",
                None,
                records,
            )

            self.assertEqual(decision.outcome, "accepted")
            self.assertEqual(decision.resolved.meta.doi, accepted_doi)
            self.assertIn("duplicate provider DOI outlier", decision.reason)

        with self.subTest(case="provider_consensus_does_not_override_distinct_record"):
            records = [
                record_matching.ProviderRecord(
                    "pubmed",
                    _record(doi="10.1000/consensus"),
                    "12345",
                ),
                record_matching.ProviderRecord(
                    "crossref",
                    _record(doi="10.1000/consensus"),
                ),
                record_matching.ProviderRecord(
                    "crossref",
                    _record(doi="10.1000/distinct", family="Different"),
                ),
            ]

            decision = record_matching.decide_title_match(
                "Reliable Quantitative Imaging with Magnetic Resonance",
                None,
                records,
            )

            self.assertEqual(decision.outcome, "rejected")
            self.assertIn("ambiguous", decision.reason.lower())

        with self.subTest(case="provider_consensus_does_not_override_multiple_outliers"):
            records = [
                record_matching.ProviderRecord(
                    "pubmed",
                    _record(doi="10.1000/consensus"),
                    "12345",
                ),
                record_matching.ProviderRecord(
                    "crossref",
                    _record(doi="10.1000/consensus"),
                ),
                record_matching.ProviderRecord(
                    "crossref",
                    _record(doi="10.1000/outlier-one"),
                ),
                record_matching.ProviderRecord(
                    "datacite",
                    _record(doi="10.1000/outlier-two"),
                ),
            ]

            decision = record_matching.decide_title_match(
                "Reliable Quantitative Imaging with Magnetic Resonance",
                None,
                records,
            )

            self.assertEqual(decision.outcome, "rejected")
            self.assertIn("ambiguous", decision.reason.lower())

        with self.subTest(case="doi_less_records_with_different_parts_remain_ambiguous"):
            part_one = (
                "Reliable Quantitative Imaging with Magnetic Resonance Part 1"
            )
            records = [
                record_matching.ProviderRecord(
                    "crossref",
                    _record(title=part_one, doi=None),
                ),
                record_matching.ProviderRecord(
                    "datacite",
                    _record(
                        title=(
                            "Reliable Quantitative Imaging with Magnetic Resonance "
                            "Part 2"
                        ),
                        doi=None,
                    ),
                ),
            ]

            decision = record_matching.decide_title_match(
                part_one,
                None,
                records,
            )

            self.assertEqual(decision.outcome, "rejected")
            self.assertIn("ambiguous", decision.reason.lower())


    def test_filename_corroboration_breaks_title_ties(self) -> None:
        with self.subTest(case="ambiguous_title_requires_year_and_title_filename_evidence"):
            records = [
                record_matching.ProviderRecord(
                    "crossref",
                    _record(
                        title="Reliable Quantitative Imaging with Magnetic Resonance",
                        doi="10.1000/one",
                    ),
                ),
                record_matching.ProviderRecord(
                    "datacite",
                    _record(
                        title="Reliable Quantitative Imaging using Magnetic Resonance",
                        doi="10.1000/two",
                    ),
                ),
            ]

            decision = record_matching.decide_title_match(
                "Reliable Quantitative Imaging by Magnetic Resonance",
                "Example_MagnResonMed24_Unrelated.pdf",
                records,
            )

            self.assertEqual(decision.outcome, "rejected")
            self.assertIn("ambiguous", decision.reason.lower())

        with self.subTest(case="filename_can_break_tie_with_year_and_title_tokens"):
            records = [
                record_matching.ProviderRecord(
                    "crossref",
                    _record(
                        title="Reliable Quantitative Imaging with Magnetic Resonance",
                        doi="10.1000/one",
                    ),
                ),
                record_matching.ProviderRecord(
                    "datacite",
                    _record(
                        title="Reliable Quantitative Mapping with Magnetic Resonance",
                        doi="10.1000/two",
                    ),
                ),
            ]

            decision = record_matching.decide_title_match(
                "Reliable Quantitative Imaging Mapping with Magnetic Resonance",
                "Example_MagnResonMed24_Reliable_Quantitative_Imaging.pdf",
                records,
            )

            self.assertEqual(decision.outcome, "accepted")
            self.assertEqual(decision.resolved.meta.doi, "10.1000/one")
            self.assertIn("filename year and title corroboration", decision.reason)


if __name__ == "__main__":
    unittest.main()
