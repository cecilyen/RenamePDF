"""Focused tests consolidated by validation surface."""

import csv
import json
import hashlib
import os
import pickle
import sqlite3
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest import mock

import pymupdf

import dedupe_run
import document_classification
import DedupePDF as dedupe_cli
import RenamePDF
import replacement_run
import ReplacePDF as replace_cli
import pdf_identity_cache
import metadata_providers
import organization_run
import pdf_evidence
import pdf_process
import file_lock
import file_metadata


# Duplicate detection and placement

def _identity(
    path: Path,
    *,
    doi: str | None = None,
    doi_score: int = 120,
    title: str | None = None,
) -> dedupe_run.PdfIdentity:
    return dedupe_run.PdfIdentity(
        path=path,
        doi=doi,
        doi_source="metadata:doi" if doi else "",
        doi_score=doi_score if doi else 0,
        title=title,
    )


class DedupePDFTests(unittest.TestCase):
    def test_positive_duplicate_identity(self) -> None:
        with self.subTest(case="same_doi_marks_newest_copy_canonical"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                older = root / "older.pdf"
                newer = root / "newer.pdf"
                older.write_bytes(b"older")
                newer.write_bytes(b"newer")
                os.utime(older, ns=(1_000_000_000, 1_000_000_000))
                os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
                identities = [
                    _identity(older, doi="10.1000/example"),
                    _identity(newer, doi="10.1000/example"),
                ]

                results = dedupe_run.build_results(identities)

                self.assertEqual(results[0]["status"], "duplicate")
                self.assertEqual(results[0]["match_method"], "doi")
                self.assertEqual(results[1]["status"], "canonical")
                self.assertEqual(results[0]["canonical_path"], str(newer))

        with self.subTest(case="similar_extracted_titles_are_duplicates"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                left = root / "ExampleA_Journal24_primary.pdf"
                right = root / "ExampleA_Journal24_copy.pdf"
                left.write_bytes(b"left")
                right.write_bytes(b"right")
                title = "A robust method for detecting duplicate scientific papers"
                identities = [
                    _identity(left, title=title),
                    _identity(right, title=title),
                ]

                results = dedupe_run.build_results(identities)

                self.assertEqual(
                    {row["status"] for row in results},
                    {"canonical", "duplicate"},
                )
                duplicate = next(row for row in results if row["status"] == "duplicate")
                self.assertEqual(duplicate["match_method"], "title")
                self.assertEqual(duplicate["similarity"], "1.000")


    def test_fast_prefix_selection_and_partial_scan_reporting(self) -> None:
        with self.subTest(case="fast_prefix_plan_only_selects_matching_stems"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                left = root / "ShaoX_Wang_one.pdf"
                right = root / "shaox_Wang_two.pdf"
                unique = root / "TaylorP_unique.pdf"

                candidates, skipped = dedupe_run.fast_prefix_plan(
                    [left, right, unique]
                )

                self.assertEqual(candidates, [left, right])
                self.assertEqual(skipped, {unique: "taylorp"})

        with self.subTest(case="fast_skipped_pdf_is_not_reported_as_unique"):
            path = Path("/tmp/TaylorP_unique.pdf")
            identity = dedupe_run.PdfIdentity(
                path=path,
                doi=None,
                doi_source="",
                doi_score=0,
                title=None,
                pdf_status=dedupe_run.FAST_SKIPPED_PDF_STATUS,
                note="fast prefix did not match another PDF",
            )

            result = dedupe_run.build_results([identity])[0]

            self.assertEqual(result["status"], "skipped_fast_prefix")
            self.assertEqual(result["canonical_path"], "")

        with self.subTest(case="fast_mode_does_not_compare_different_prefixes"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                left = root / "ShaoX_Wang_one.pdf"
                right = root / "TaylorP_Wang_two.pdf"
                left.write_bytes(b"left")
                right.write_bytes(b"right")
                identities = [
                    _identity(left, doi="10.1000/shared"),
                    _identity(right, doi="10.1000/shared"),
                ]

                full = dedupe_run.build_results(identities)
                fast = dedupe_run.build_results(identities, fast_prefix=True)

                self.assertEqual(
                    {row["status"] for row in full},
                    {"canonical", "duplicate"},
                )
                self.assertEqual([row["status"] for row in fast], ["unique"] * 2)

        with self.subTest(case="fast_main_only_inspects_matching_prefixes"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                left = root / "ShaoX_Wang_one.pdf"
                right = root / "shaox_Wang_two.pdf"
                unique = root / "TaylorP_unique.pdf"
                report = root / "fast.csv"
                for path in (left, right, unique):
                    path.write_bytes(path.name.encode())

                called_paths: list[Path] = []
                original = dedupe_run.inspect_pdfs

                def inspect_candidates(
                    paths: list[Path],
                    **_kwargs: object,
                ) -> list[dedupe_run.PdfIdentity]:
                    called_paths.extend(paths)
                    return [
                        _identity(path, doi="10.1000/shared")
                        for path in paths
                    ]

                dedupe_run.inspect_pdfs = inspect_candidates
                try:
                    exit_code = dedupe_cli.main(
                        [str(root), "--fast", "--report", str(report)]
                    )
                finally:
                    dedupe_run.inspect_pdfs = original

                with report.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                status_by_file = {Path(row["file"]): row["status"] for row in rows}
                self.assertEqual(exit_code, 0)
                self.assertEqual(called_paths, [left.resolve(), right.resolve()])
                self.assertEqual(
                    status_by_file[unique.resolve()],
                    "skipped_fast_prefix",
                )


    def test_discover_pdfs_accepts_additional_excluded_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            included = root / "Library" / "paper.pdf"
            excluded = root / "0WIP" / "draft.pdf"
            transaction_artifact = root / ".paper.pdf.repairing.pdf"
            included.parent.mkdir()
            excluded.parent.mkdir()
            included.write_bytes(b"included")
            excluded.write_bytes(b"excluded")
            transaction_artifact.write_bytes(b"interrupted replacement")

            paths = dedupe_run.discover_pdfs(
                root,
                excluded_directories=["0wip"],
            )

            self.assertEqual(paths, [included])

    def test_main_halts_before_inspection_when_recovery_is_needed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollback = root / ".paper.pdf.renamepdf-rollback"
            rollback.write_bytes(b"original PDF")

            with mock.patch.object(dedupe_run, "inspect_pdfs") as inspector:
                exit_code = dedupe_cli.main([str(root)])

            self.assertEqual(exit_code, 2)
            inspector.assert_not_called()

    def test_inspection_infrastructure_failures(self) -> None:
        with self.subTest(case="dry_run_inspection_timeout_is_reported"):
            path = Path("/tmp/ShaoX_stalled.pdf")

            def report_timeout(
                paths: list[Path],
                **kwargs: object,
            ) -> list[dedupe_run.pdf_process.PdfBatchInspectionOutcome]:
                outcome = dedupe_run.pdf_process.PdfBatchInspectionOutcome(
                    path=paths[0],
                    evidence=None,
                    error_status="pdf_processing_timeout",
                    error_note="Inspection exceeded 0.01 seconds",
                )
                progress = kwargs["progress"]
                assert callable(progress)
                progress(1, 1, outcome)
                return [outcome]

            with mock.patch.object(
                dedupe_run.pdf_process,
                "inspect_pdf_batch",
                side_effect=report_timeout,
            ):
                identities = dedupe_run.inspect_pdfs(
                    [path],
                    workers=1,
                    inspection_timeout=0.01,
                )

            self.assertEqual(len(identities), 1)
            self.assertEqual(identities[0].path, path)
            self.assertEqual(identities[0].pdf_status, "pdf_processing_timeout")
            self.assertIn("0.01 seconds", identities[0].note)

        with self.subTest(case="dry_run_pool_start_failure_is_reported"):
            path = Path("/tmp/ShaoX_worker-unavailable.pdf")
            with mock.patch.object(
                dedupe_run.pdf_process,
                "inspect_pdf_batch",
                side_effect=OSError("worker process unavailable"),
            ):
                identities = dedupe_run.inspect_pdfs([path], workers=1)

            self.assertEqual(len(identities), 1)
            self.assertEqual(identities[0].path, path)
            self.assertEqual(identities[0].pdf_status, "pdf_processing_error")
            self.assertIn("worker process unavailable", identities[0].note)


    def test_dry_run_inspection_uses_isolated_worker(self) -> None:
        import pymupdf

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ShaoX_worker.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text(
                (72, 72),
                "Isolated PDF Inspection Worker\n"
                "https://doi.org/10.1000/isolated-worker",
            )
            document.save(path)
            document.close()

            identities = dedupe_run.inspect_pdfs(
                [path],
                workers=1,
                inspection_timeout=10,
            )

        self.assertEqual(len(identities), 1)
        self.assertEqual(identities[0].doi, "10.1000/isolated-worker")
        self.assertNotEqual(identities[0].pdf_status, "pdf_processing_error")

    def test_authorized_preprocessing_precedes_identity(self) -> None:
        with self.subTest(case="preprocess_identities_rewrites_only_needed_pdfs"):
            needed = dedupe_run.PdfIdentity(
                path=Path("/tmp/needed.pdf"),
                doi="10.1000/needed",
                doi_source="text:p1",
                doi_score=100,
                title="Needed",
                pdf_status="pdf_decryption_needed",
            )
            unchanged = _identity(
                Path("/tmp/unchanged.pdf"),
                doi="10.1000/unchanged",
            )
            called: list[Path] = []

            def processor(path: Path, _password: str) -> dedupe_run.PdfIdentity:
                called.append(path)
                return dedupe_run.PdfIdentity(
                    path=path,
                    doi="10.1000/needed",
                    doi_source="text:p1",
                    doi_score=100,
                    title="Needed",
                    pdf_status="pdf_decrypted",
                )

            updated = dedupe_run.preprocess_identities(
                [needed, unchanged],
                processor=processor,
            )

            self.assertEqual(called, [needed.path])
            self.assertEqual(updated[0].pdf_status, "pdf_decrypted")
            self.assertIs(updated[1], unchanged)

        with self.subTest(case="commit_inspection_rewrites_before_identity_extraction"):
            import pymupdf
            import re

            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "damaged.pdf"
                document = pymupdf.open()
                page = document.new_page()
                page.insert_text(
                    (72, 72),
                    "A Damaged PDF With Recoverable Identity\n"
                    "https://doi.org/10.1000/dedupe-repair",
                )
                document.save(pdf)
                document.close()
                pdf.write_bytes(
                    re.sub(rb"startxref\s+\d+", b"startxref\n0", pdf.read_bytes())
                )

                identity = dedupe_run.inspect_pdf(pdf, commit=True)

                self.assertEqual(identity.pdf_status, "pdf_repaired")
                self.assertEqual(identity.doi, "10.1000/dedupe-repair")
                checked = pymupdf.open(pdf)
                try:
                    self.assertFalse(checked.is_repaired)
                finally:
                    checked.close()


    def test_title_only_duplicate_corpus(self) -> None:
        with self.subTest(case="title_only_match_rejects_generic_or_partial_titles"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                left = root / "facility.pdf"
                right = root / "coil.pdf"
                third = root / "generic.pdf"
                fourth = root / "review.pdf"
                for path in (left, right, third, fourth):
                    path.write_bytes(path.name.encode())
                identities = [
                    _identity(
                        left,
                        title="Considerations for Setting up a Small Animal Imaging Facility",
                    ),
                    _identity(right, title="Research Article Small"),
                    _identity(third, title="Magnetic Resonance Imaging"),
                    _identity(
                        fourth,
                        title=(
                            "Ultrahigh field magnetic resonance imaging: new frontiers "
                            "and possibilities in human imaging"
                        ),
                    ),
                ]

                results = dedupe_run.build_results(identities)

                self.assertEqual([row["status"] for row in results], ["unique"] * 4)

        with self.subTest(case="title_only_match_rejects_scanner_boilerplate"):
            identities = [
                _identity(
                    Path("/tmp/BendallMR_MRM86_DecoupleTORO.pdf"),
                    title="Scanned doc in PDF format",
                ),
                _identity(
                    Path("/tmp/ChenCN_MRM86_Coil_SNR_Power.pdf"),
                    title="Scanned doc in PDF format",
                ),
            ]

            results = dedupe_run.build_results(identities)

            self.assertEqual([row["status"] for row in results], ["unique"] * 2)

        with self.subTest(case="title_only_match_rejects_reviewed_false_positive_corpus"):
            groups = (
                (
                    ("ConollyS_JMR89_Adiabatic.pdf", "A Selective Adiabatic Spin-Echo Pulse"),
                    ("ConollyS_MacovskiA_MRM91_2AFP.pdf", "A reduced power selective adiabatic spin-echo pulse sequence"),
                ),
                (
                    ("AguirreGK_NeuroImage97_fMRI_Stat2.pdf", "Empirical Analyses of BOLD fMRI Statistics"),
                    ("ZarahnE_NeuroImage97_fMRI_Stat1.pdf", "Empirical Analyses of BOLD fMRI Statistics"),
                ),
                (
                    ("SezanMI_IEEETransMedImg82_POCS2.pdf", "Image Restoration by the Method of Convex and Numerical Results"),
                    ("YoulaDC_IEEETransMedImg82_POCS1.pdf", "Image Restoration by the Method of Convex"),
                ),
                (
                    ("PearsonHA_Blood63_CrRBC1.pdf", "at UNIV OF PITTSBURGH HSLS on June 5, 2008 For personal use only"),
                    ("PearsonHA_Blood63_CrRBC2.pdf", "at UNIV OF PITTSBURGH HSLS on June 5, 2008 For personal use only"),
                ),
                (
                    ("04_Manganese-enhanced magnetic resonance.pdf", "Manganese-enhanced magnetic resonance imaging (MEMRI)"),
                    ("LuHb_PNAS07_Cocaine_MEMRI.pdf", "Cocaine-induced brain activation detected by dynamic manganese-enhanced magnetic resonance imaging (MEMRI)"),
                ),
                tuple(
                    (filename, "Scanned doc in PDF format")
                    for filename in (
                        "BendallMR_MRM86_DecoupleTORO.pdf",
                        "ChenCN_MRM86_Coil_SNR_Power.pdf",
                        "DotyFD_JMR81_Coil.pdf",
                        "LinkJ_NMRBasicPrinciplesandProgress92_HomogeneousCoilDesign.pdf",
                        "Thulborn_BrainMetablicImaging.pdf",
                    )
                ),
                (
                    ("HurlstonSE_IntJImgSysTech96_MRMicroscopy_Coil.pdf", "Optimized Radiofrequency Coils for Increased Signal-to-Noise Ratio in Magnetic Resonance Microscopy"),
                    ("NeubergerT_Webb_NMRBioMed08_Rev_MicroscopyCoil.pdf", "Radiofrequency coils for magnetic resonance microscopy"),
                ),
                (
                    ("GaoJH_MRM96_BOLD_Inflow.pdf", "Quantitative Assessment of Blood Inflow Effects in Functional MRI Signals"),
                    ("GaoJh_NeuroImage12_Rev_Inflow.pdf", "Inflow effects on functional MRI"),
                ),
                (
                    ("JasanoffA_CurrOpinNeurobiol07_Rev_MRI_ContrastAgent_Functional.pdf", "MRI contrast agents for functional molecular imaging"),
                    ("Sorensen AG_Rosen_JMRI97_Rev_Gd_fMRI.pdf", "Contrast Agents In Functional MR Imaging"),
                ),
                (
                    ("MatthewsPM_Jezzard_JNeurolNeurosurgPsychiatry04_Rev_fMRI.pdf", "Functional magnetic resonance imaging"),
                    ("SilvaAC_MethodsMolecularBiology11_Rev_Marmoset_MRI.pdf", "Longitudinal Functional Magnetic Resonance Imaging"),
                ),
                (
                    ("Review_Functional Magnetic Resonance Imaging of Brain Reward Circuitry in Human.pdf", "Functional Magnetic Resonance Imaging of Brain Reward Circuitry in the Human"),
                    ("KimSG_JNeuroSciMethod97_fMRI_Rev.pdf", "Functional magnetic resonance imaging of the human brain"),
                ),
                (
                    ("RichterW_NeuroReport97_MentalRotation.pdf", "scanned document from HP9100C"),
                    ("RichterW_NeuroReport97_Motor.pdf", "scanned document from HP9100C"),
                ),
                (
                    ("OstergaardL_JMRI05_Review_DSCMRI.pdf", "Principles of Cerebral Perfusion Imaging by Bolus Tracking"),
                    ("OstergaardL_TMRI04_Review_DSCMRI.pdf", "Cerebral Perfusion Imaging by Bolus Tracking"),
                ),
                (
                    ("GolayX_TopMRI04_Review_ASL.pdf", "Perfusion Imaging Using Arterial Spin Labeling"),
                    ("WangJj_Detre_MRM02_PASL_CASL.pdf", "Comparison of Quantitative Perfusion Imaging Using Arterial Spin Labeling at 1.5 and 4.0 Tesla"),
                ),
                (
                    ("HennigJ_ConceptMR91_Echo1.pdf", "Echoes - How to Generate, Recognize Use or Avoid Them in MR-Imaging Sequences"),
                    ("HennigJ_ConceptMR91_Echo2.pdf", "Echoes - How to Generate, Recognize Use or Avoid Them in MR-Imaging Sequences"),
                ),
                (
                    ("CallawayEM_AnnualRevNeurosci98_Macaque_V1_LocalCircuits.pdf", "LOCAL CIRCUITS IN PRIMARY VISUAL CORTEX OF THE MACAQUE MONKEY"),
                    ("CallawayEM_JNeurosci98_LaminarCircuits_Macaque_V1.pdf", "Prenatal Development of Layer-Specific Local Circuits in Primary Visual Cortex of the Macaque Monkey"),
                ),
                (
                    ("DeAngelisGC_Freeman_JNeurophysiol93_1.pdf", "Spatiotemporal Organization of Simple-Cell Receptive Fields"),
                    ("DeAngelisGC_Freeman_JNeurophysiol93_2.pdf", "Spatiotemporal Organization of Simple-Cell Receptive Fields in the"),
                ),
                (
                    ("DurnadS_VisNeurosci07_Cat_TemporalFreq_SurroundSuppression.pdf", "Temporal properties of surround suppression in cat primary visual cortex"),
                    ("IshikawaA_EJN10_CatLGNV1_Dynamic_SpatialFrequency_SurroundSuppression.pdf", "Temporal properties of spatial frequency tuning of surround suppression in the primary visual cortex and the lateral geniculate nucleus of the cat"),
                ),
                (
                    ("GirmanSV_JNeuroPhysiol99_RatV1Property.pdf", "Receptive Field Properties of Single Neurons in Rat Primary"),
                    ("LeventhalAG_Hirsh_JNeurophysiol78_CatA17Layer.pdf", "Receptive-Field Properties of Neurons"),
                ),
                (
                    ("KellyJP_JPhysiol74_CatA17LayerStructure.pdf", "VISUAL CORTEX OF THE CAT"),
                    ("PadnickLB_JApplPhysiol99_Cat_LowpO2.pdf", "Oxygenation of the cat primary visual cortex"),
                ),
                (
                    ("MorleyJW_JPhysiol97_Cat21a_SpatioTemporalFreqTuning.pdf", "Spatial and temporal frequency selectivity of cells in area 21a of the cat"),
                    ("ZumbroichTJ_JNeurosci97_CatSuprasylvian_SpatioTemporal.pdf", "Spatial and Temporal Selectivity the Cat"),
                ),
            )

            for group in groups:
                with self.subTest(files=[filename for filename, _title in group]):
                    identities = [
                        _identity(Path("/tmp") / filename, title=title)
                        for filename, title in group
                    ]

                    results = dedupe_run.build_results(identities)

                    self.assertEqual(
                        [row["status"] for row in results],
                        ["unique"] * len(group),
                    )

        with self.subTest(case="title_only_match_preserves_reviewed_true_duplicate_corpus"):
            groups = (
                (
                    ("LeeSP_MRM01_BOLD_CBV_CBF_19F_PFC.pdf", "Relative Changes of Cerebral Arterial and Venous Blood Volumes During Increased Cerebral Blood Flow Implications for BOLD fMRI"),
                    ("LeeSP_MRM01_Arterial_Venous_CBV_CBF_BOLD.pdf", "Relative Changes of Cerebral Arterial and Venous Blood Volumes During Increased Cerebral Blood Flow Implications for BOLD fMRI"),
                ),
                (
                    ("OcaliO_MRM98_UISNR.pdf", "Ultimate Intrinsic Signal-to-Noise Ratio in MRI"),
                    ("OcaliO_MRM98_UISNR_copy.pdf", "Ultimate Intrinsic Signal-to-Noise Ratio in MRI"),
                ),
                (
                    ("AlsopDC_Detre_JCBFM96_ASL_TransitTime.pdf", "Reduced Transit-Time Sensitivity in Noninvasive Magnetic Resonance Imaging of Human Cerebral Blood Flow"),
                    ("AlsopDC_Detre_JCBFM96_TransitTime_CBF.pdf", "Reduced Tr ansit-Time Sensitivity in Noninvasive Magnetic Resonance Imaging of Human Cerebral Blood Flow"),
                ),
                (
                    ("SmithC_Broome_Anaesthesia02_IsofluranceAbsorption_PP_PE.pdf", "Leakage and absorption of isoflurane by different types of anaesthetic circuit and monitoring tubing"),
                    ("SmithC_Broome_Aneasthsia02_Isoflurane_PP_PE_PVC.pdf", "Leakage and absorption of isoflurane by different types of anaesthetic circuit and monitoring tubing"),
                ),
                (
                    ("Hudetz_Microcirculation97_Review_CBF.pdf", "Blood Flow in the Cerebral Capillary Network: A Review Emphasizing Observations with Intravital Microscopy"),
                    ("HudetzAG_Microcirculation97_Rev_CBF_Capillary_IntravitalMicroscopy.pdf", "Blood Flow in the Cerebral Capillary Network: A Review Emphasizing Observations with Intravital Microscopy"),
                ),
                (
                    ("ThompsonJK_Freeman_Science03_NeuronalAct_pO2.pdf", "Single-Neuron Activity and Tissue Oxygenation in the Cerebral Cortex"),
                    ("ThompsonJK_Freeman_Science03_CatV1_ODC_OC_pO2_NeuronalActivity.pdf", "Single-Neuron Activity and Tissue Oxygenation in the"),
                ),
                (
                    ("FriendSM_Baker_VisionRes93_SpatioTemporalFreq_A18_cat.pdf", "Separability in Area 18 Neurons of the Cat"),
                    ("FriendSM_VisionRes93_Cat_SpatioTemporalFreq_A18.pdf", "Spatio-Temporal Frequency Separability in Area 18 Neurons of the Cat"),
                ),
                (
                    ("ThomsonAM_Bannister_CerebCortex03_Rev_InterlaminarConnections.pdf", "Interlaminar Connections in the Neocortex"),
                    ("ThomsonAM_CerebCtx03_Rev_InterlaminarConnections.pdf", "Interlaminar Connections in the Neocortex"),
                ),
            )

            for group in groups:
                with self.subTest(files=[filename for filename, _title in group]):
                    identities = [
                        _identity(Path("/tmp") / filename, title=title)
                        for filename, title in group
                    ]

                    results = dedupe_run.build_results(identities)

                    self.assertEqual(
                        {row["status"] for row in results},
                        {"canonical", "duplicate"},
                    )


    def test_doi_title_corroboration(self) -> None:
        with self.subTest(case="title_does_not_merge_conflicting_dois"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                left = root / "left.pdf"
                right = root / "right.pdf"
                left.write_bytes(b"left")
                right.write_bytes(b"right")
                title = "Correction of a study with the same displayed title"
                identities = [
                    _identity(left, doi="10.1000/original", title=title),
                    _identity(right, doi="10.1000/correction", title=title),
                ]

                results = dedupe_run.build_results(identities)

                self.assertEqual(
                    [row["status"] for row in results],
                    ["unique", "unique"],
                )

        with self.subTest(case="shared_doi_does_not_override_conflicting_titles"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                left = root / "AutheletG_IOP17_Cryostat.pdf"
                right = root / "AutheletG_IOP19_PolymerCryostat.pdf"
                left.write_bytes(b"left")
                right.write_bytes(b"right")
                identities = [
                    _identity(
                        left,
                        doi="10.1000/contaminated",
                        title="Conceptual design of a cryogen-free micro MRI device",
                    ),
                    _identity(
                        right,
                        doi="10.1000/contaminated",
                        title=(
                            "All polymer cryogen free cryostat for micro MRI "
                            "application at clinical field"
                        ),
                    ),
                ]

                results = dedupe_run.build_results(identities)

                self.assertEqual(
                    [row["status"] for row in results],
                    ["unique", "unique"],
                )

        with self.subTest(case="low_priority_doi_requires_title_confirmation"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                left = root / "left.pdf"
                right = root / "right.pdf"
                left.write_bytes(b"left")
                right.write_bytes(b"right")
                title = "A shared title confirms a raw embedded DOI candidate"
                identities = [
                    _identity(
                        left,
                        doi="10.1000/raw",
                        doi_score=30,
                        title=title,
                    ),
                    _identity(
                        right,
                        doi="10.1000/raw",
                        doi_score=30,
                        title=title,
                    ),
                ]

                results = dedupe_run.build_results(identities)

                duplicate = next(row for row in results if row["status"] == "duplicate")
                self.assertEqual(duplicate["match_method"], "doi+title")

        with self.subTest(case="low_priority_doi_without_title_is_not_enough"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                left = root / "left.pdf"
                right = root / "right.pdf"
                left.write_bytes(b"left")
                right.write_bytes(b"right")
                identities = [
                    _identity(left, doi="10.1000/citation", doi_score=30),
                    _identity(right, doi="10.1000/citation", doi_score=30),
                ]

                results = dedupe_run.build_results(identities)

                self.assertEqual(
                    [row["status"] for row in results],
                    ["unique", "unique"],
                )


    def test_embedded_identity_precedes_reference_and_filename_noise(self) -> None:
        with self.subTest(case="inspection_uses_embedded_metadata_not_filename_doi"):
            import pymupdf

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pdf = root / "10.9999_filename-only.pdf"
                document = pymupdf.open()
                page = document.new_page()
                page.insert_text(
                    (72, 72),
                    "A Reliable Embedded Title for Duplicate PDF Detection",
                )
                document.set_metadata({
                    "title": "A Reliable Embedded Title for Duplicate PDF Detection",
                    "subject": "https://doi.org/10.1000/embedded",
                })
                document.save(pdf)
                document.close()

                identity = dedupe_run.inspect_pdf(pdf)

                self.assertEqual(identity.doi, "10.1000/embedded")
                self.assertEqual(
                    identity.title,
                    "A Reliable Embedded Title for Duplicate PDF Detection",
                )
                self.assertNotEqual(identity.doi_source, "filename")

        with self.subTest(case="later_reference_link_does_not_override_first_page_doi"):
            import pymupdf

            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "article.pdf"
                document = pymupdf.open()
                first_page = document.new_page()
                first_page.insert_text(
                    (72, 72),
                    "A Primary Article With Reliable DOI Evidence\n"
                    "https://doi.org/10.1000/primary",
                )
                for _ in range(4):
                    document.new_page()
                last_page = document[4]
                last_page.insert_text((72, 72), "References")
                last_page.insert_link({
                    "kind": pymupdf.LINK_URI,
                    "from": pymupdf.Rect(72, 90, 320, 110),
                    "uri": "https://doi.org/10.1000/referenced",
                })
                document.save(pdf)
                document.close()

                identity = dedupe_run.inspect_pdf(pdf)

                self.assertEqual(identity.doi, "10.1000/primary")

        with self.subTest(case="leading_overlap_doi_does_not_override_selected_article"):
            import pymupdf

            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "overlapping-issue-page.pdf"
                document = pymupdf.open()
                first_page = document.new_page()
                first_page.insert_textbox(
                    pymupdf.Rect(40, 40, 555, 260),
                    "Previous article body text " * 100,
                    fontsize=6,
                )
                first_page.insert_text(
                    (390, 280),
                    "10.1000/previous-article",
                    fontsize=7,
                )
                first_page.insert_textbox(
                    pymupdf.Rect(90, 310, 505, 390),
                    "Target Article Identity Begins Below an Overlapping Issue Page",
                    fontsize=18,
                    align=pymupdf.TEXT_ALIGN_CENTER,
                )
                for _ in range(2):
                    document.new_page()
                last_page = document.new_page()
                last_page.insert_text(
                    (72, 72),
                    "10.1000/target-article",
                )
                document.save(pdf)
                document.close()

                identity = dedupe_run.inspect_pdf(pdf)

                self.assertEqual(identity.doi, "10.1000/target-article")

        with self.subTest(case="filename_only_doi_is_not_embedded_identity"):
            path = Path("/tmp/10.1000_filename.pdf")
            original = pdf_evidence.inspect_pdf
            pdf_evidence.inspect_pdf = lambda *_args, **_kwargs: pdf_evidence.PdfEvidence(
                dois=[pdf_evidence.DoiCandidate("10.1000/filename", "filename", 85)],
                title=None,
                pii=None,
                arxiv_doi=None,
            )
            try:
                identity = dedupe_run.inspect_pdf(path)
            finally:
                pdf_evidence.inspect_pdf = original

            self.assertIsNone(identity.doi)


    def test_unusable_pdfs_are_excluded(self) -> None:
        with self.subTest(case="unreadable_pdf_is_an_error"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "broken.pdf"
                pdf.write_bytes(b"not a PDF")

                identity = dedupe_run.inspect_pdf(pdf)
                results = dedupe_run.build_results([identity])

                self.assertEqual(identity.pdf_status, "pdf_unreadable")
                self.assertEqual(results[0]["status"], "error")

        with self.subTest(case="unrepairable_pdf_is_labeled_and_excluded"):
            with tempfile.TemporaryDirectory() as directory:
                pdf = Path(directory) / "damaged.pdf"
                pdf.write_bytes(b"damaged")
                identity = dedupe_run.PdfIdentity(
                    path=pdf,
                    doi=None,
                    doi_source="",
                    doi_score=0,
                    title=None,
                    pdf_status="pdf_unrepairable",
                    note="normalization failed",
                )

                results = dedupe_run.build_results([identity])

                self.assertEqual(results[0]["status"], "unrepairable")
                self.assertEqual(results[0]["pdf_status"], "pdf_unrepairable")


    def test_commit_moves_duplicate_without_deleting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical.pdf"
            duplicate = root / "copy.pdf"
            canonical.write_bytes(b"canonical")
            duplicate.write_bytes(b"duplicate")
            rows = [{
                "file": str(duplicate),
                "status": "duplicate",
                "note": "",
                "target_path": "",
            }]

            dedupe_run.commit_duplicates(root, rows)

            target = Path(rows[0]["target_path"])
            self.assertFalse(duplicate.exists())
            self.assertEqual(target.parent, root / "Duplicated")
            self.assertEqual(target.read_bytes(), b"duplicate")
            self.assertEqual(canonical.read_bytes(), b"canonical")


# Shared identity cache and worker reuse

"""Consolidated identity-cache and PDF-worker validation tests."""





DOI = "10.1000/reliable"


def _payload(path: Path) -> dict[str, object]:
    del path
    return pdf_evidence.evidence_to_cache_payload(
        pdf_evidence.PdfEvidence(
            dois=[
                pdf_evidence.DoiCandidate(
                    "10.1000/cache-test",
                    "text:p1",
                    100,
                )
            ],
            title="A Persistent PDF Identity Cache Test",
            pii=None,
            arxiv_doi=None,
        )
    )


def _no_match_providers() -> mock.Mock:
    providers = mock.Mock()
    providers.records_by_doi = mock.AsyncMock(
        return_value=metadata_providers.ProviderLookup()
    )
    providers.records_by_title = mock.AsyncMock(
        return_value=metadata_providers.ProviderLookup()
    )
    return providers


class IdentityCacheStorageTests(unittest.TestCase):
    def test_compact_classification_cache_contract(self) -> None:
        with self.subTest(case="varied_document_text_cannot_inflate_the_classification_payload"):
            text = "bioRxiv preprint. " + " ".join(
                hashlib.sha256(str(index).encode()).hexdigest() for index in range(2600)
            )
            evidence = pdf_evidence.PdfEvidence(
                [], "Synthetic cache-size regression", None, None,
                text_samples=(text,),
            )
            payload = pdf_evidence.evidence_to_cache_payload(evidence)
            self.assertLessEqual(len(json.dumps(payload).encode("utf-8")), 4096)

        with self.subTest(case="corrupt_classification_markers_are_rejected"):
            for invalid in ("invalid", {}, {"preprint": True}):
                with self.subTest(invalid=invalid):
                    payload = _payload(Path("paper.pdf"))
                    payload["classification_markers"] = invalid
                    with self.assertRaises(ValueError):
                        pdf_evidence.evidence_from_cache_payload(payload)
            payload = _payload(Path("paper.pdf"))
            payload["classification_markers"]["preprint"] = 1
            with self.assertRaises(ValueError):
                pdf_evidence.evidence_from_cache_payload(payload)

        with self.subTest(case="classification_round_trip_preserves_categories_without_text"):
            cases = (
                ("paper.pdf", "bioRxiv preprint. " + "body " * 40_000, "Preprints"),
                ("paper.pdf", "A dissertation submitted for a doctoral degree", "Dissertations"),
                ("paper.pdf", "Front matter\f" + "body " * 10_000 + "Proc. Intl. Soc. Mag. Reson. Med. 20 (2012)", "Proceedings"),
                ("paper.pdf", "Program No: 1234 Poster Type: Digital Poster", "Proceedings"),
                ("paper.pdf", "European Conference on Antennas and Propagation EuCAP 2014", "Proceedings"),
                ("p01234.pdf", "Introduction Methods Results References", "Proceedings"),
                ("01234.pdf", "Introduction Results Discussion References Proc. ISMRM", "Proceedings"),
                ("paper.pdf", "Philips Healthcare technical guide", "VendorDocuments"),
                ("paper.pdf", "arXiv:2509.16803v1 [q-bio.NC]", "Preprints"),
                ("paper.pdf", "Prior work: Example et al. arXiv:2509.16803", "Others"),
                ("paper.pdf", "Ordinary research text", "Others"),
            )
            for name, text, expected in cases:
                with self.subTest(name=name, expected=expected, text=text[:50]):
                    evidence = pdf_evidence.PdfEvidence(
                        [], "Synthetic manuscript", None, None, text_samples=(text,),
                    )
                    payload = pdf_evidence.evidence_to_cache_payload(evidence)
                    restored = pdf_evidence.evidence_from_cache_payload(payload)
                    fresh = document_classification.classify_failed_pdf(
                        Path(name), title=evidence.title, text_samples=evidence.text_samples,
                    )
                    cached = document_classification.classify_failed_pdf(
                        Path(name), title=restored.title,
                        text_markers=restored.classification_markers,
                    )
                    self.assertEqual(fresh.category, expected)
                    self.assertEqual(cached, fresh)
                    self.assertEqual(restored.text_samples, ())
                    self.assertNotIn("classification_text", payload)
                    self.assertLess(len(json.dumps(payload)), 4096)

        with self.subTest(case="cached_markers_do_not_freeze_filename_classification"):
            evidence = pdf_evidence.PdfEvidence(
                [], "Synthetic manuscript", None, None,
                text_samples=("Introduction Methods Results References",),
            )
            restored = pdf_evidence.evidence_from_cache_payload(
                pdf_evidence.evidence_to_cache_payload(evidence)
            )
            for name, expected in (("p01234.pdf", "Proceedings"), ("paper.pdf", "Others")):
                result = document_classification.classify_failed_pdf(
                    Path(name), title=restored.title, text_markers=restored.classification_markers,
                )
                self.assertEqual(result.category, expected)


    def test_stale_identity_evidence_is_rejected(self) -> None:
        with self.subTest(case="put_cannot_stamp_old_evidence_with_replaced_file_state"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "collection"
                root.mkdir()
                source = root / "paper.pdf"
                source.write_bytes(b"old")
                token = file_metadata.file_token(source)
                evidence = pdf_evidence.PdfEvidence([], "Old inspected title", None, None, source_token=token)
                incoming = root / "incoming.tmp"
                incoming.write_bytes(b"new")
                os.utime(incoming, ns=(token[3], token[3]))
                incoming.replace(source)
                with pdf_identity_cache.IdentityCache(root, extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION, temp_root=Path(directory) / "cache") as cache:
                    cache.put(source, pdf_evidence.evidence_to_cache_payload(evidence))
                    self.assertIsNone(cache.get(source))
                    self.assertEqual(cache.write_count, 0)

        with self.subTest(case="file_change_invalidates_cached_identity"):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = base / "collection"
                root.mkdir()
                pdf = root / "paper.pdf"
                pdf.write_bytes(b"first")
                cache = pdf_identity_cache.IdentityCache(
                    root,
                    extractor_version="test-v1",
                    temp_root=base / "local",
                )

                with cache:
                    cache.put(pdf, _payload(pdf))
                    self.assertIsNotNone(cache.get(pdf))
                    pdf.write_bytes(b"second and different length")
                    self.assertIsNone(cache.get(pdf))

        with self.subTest(case="extractor_version_change_rejects_snapshot"):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = base / "collection"
                root.mkdir()
                pdf = root / "paper.pdf"
                pdf.write_bytes(b"synthetic PDF identity")
                first = pdf_identity_cache.IdentityCache(
                    root,
                    extractor_version="test-v1",
                    temp_root=base / "local-a",
                )
                with first:
                    first.put(pdf, _payload(pdf))
                    first.publish()

                second = pdf_identity_cache.IdentityCache(
                    root,
                    extractor_version="test-v2",
                    temp_root=base / "local-b",
                )
                with second:
                    self.assertIsNone(second.get(pdf))

                self.assertTrue(
                    any("invalid collection" in note for note in second.notes)
                )


    def test_local_cache_publishes_and_seeds_a_new_local_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "collection"
            root.mkdir()
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"synthetic PDF identity")

            first = pdf_identity_cache.IdentityCache(
                root,
                extractor_version="test-v1",
                temp_root=base / "local-a",
            )
            self.assertFalse(first.local_path.is_relative_to(root.resolve()))
            with first:
                first.put(pdf, _payload(pdf))
                snapshot = first.publish()

            self.assertEqual(
                snapshot,
                root.resolve() / pdf_identity_cache.CACHE_FILENAME,
            )
            self.assertTrue(snapshot.is_file())
            if os.name == "posix":
                self.assertEqual(
                    stat.S_IMODE(snapshot.stat().st_mode),
                    pdf_identity_cache.PRIVATE_FILE_MODE,
                )

            second = pdf_identity_cache.IdentityCache(
                root,
                extractor_version="test-v1",
                temp_root=base / "local-b",
            )
            with second:
                cached = second.get(pdf)

            self.assertIsNotNone(cached)
            assert cached is not None
            self.assertEqual(cached["path"], str(pdf))
            evidence = pdf_evidence.evidence_from_cache_payload(cached)
            self.assertEqual(evidence.dois[0].doi, "10.1000/cache-test")
            self.assertTrue(
                any("collection snapshot" in note for note in second.notes)
            )

    def test_prune_missing_removes_an_entry_after_rename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "collection"
            root.mkdir()
            original = root / "original.pdf"
            renamed = root / "renamed.pdf"
            original.write_bytes(b"synthetic PDF identity")
            cache = pdf_identity_cache.IdentityCache(
                root,
                extractor_version="test-v1",
                temp_root=base / "local",
                snapshot_filename=".renamepdf_identity_cache.sqlite",
                local_cache_namespace="renamepdf-identity-cache",
            )

            with cache:
                cache.put(original, _payload(original))
                original.rename(renamed)
                cache.prune_missing()
                snapshot = cache.publish()

            connection = sqlite3.connect(snapshot)
            try:
                count = connection.execute(
                    "SELECT COUNT(*) FROM identity_cache"
                ).fetchone()
            finally:
                connection.close()

            self.assertEqual(count, (0,))

class DedupeIdentityCacheWorkflowTests(unittest.TestCase):
    def test_all_clis_use_one_evidence_cache_configuration(self) -> None:
        self.assertEqual(
            RenamePDF.IDENTITY_CACHE_FILENAME,
            pdf_identity_cache.CACHE_FILENAME,
        )
        self.assertEqual(
            replacement_run.IDENTITY_CACHE_FILENAME,
            pdf_identity_cache.CACHE_FILENAME,
        )
        self.assertEqual(
            RenamePDF.IDENTITY_CACHE_NAMESPACE,
            pdf_identity_cache.DEFAULT_LOCAL_CACHE_NAMESPACE,
        )
        self.assertEqual(
            replacement_run.IDENTITY_CACHE_NAMESPACE,
            pdf_identity_cache.DEFAULT_LOCAL_CACHE_NAMESPACE,
        )
        self.assertEqual(
            dedupe_run.IDENTITY_EXTRACTOR_VERSION,
            pdf_evidence.EVIDENCE_CACHE_VERSION,
        )

    def test_cache_reuse_excludes_processing_failures(self) -> None:
        with self.subTest(case="cache_hit_skips_worker_pool"):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = base / "collection"
                root.mkdir()
                pdf = root / "paper.pdf"
                pdf.write_bytes(b"synthetic PDF identity")
                cache = pdf_identity_cache.IdentityCache(
                    root,
                    extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION,
                    temp_root=base / "local",
                )

                with cache:
                    cache.put(pdf, _payload(pdf))
                    with mock.patch.object(
                        dedupe_run.pdf_process,
                        "inspect_pdf_batch",
                    ) as inspect_batch:
                        identities = dedupe_run.inspect_pdfs(
                            [pdf],
                            workers=1,
                            cache=cache,
                        )

                inspect_batch.assert_not_called()
                self.assertEqual(identities[0].doi, "10.1000/cache-test")
                self.assertEqual(cache.hit_count, 1)

        with self.subTest(case="processing_error_is_not_cached"):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = base / "collection"
                root.mkdir()
                pdf = root / "broken.pdf"
                pdf.write_bytes(b"not a PDF")
                cache = pdf_identity_cache.IdentityCache(
                    root,
                    extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION,
                    temp_root=base / "local",
                )

                def fail_batch(
                    paths: list[Path],
                    **kwargs: object,
                ) -> list[pdf_process.PdfBatchInspectionOutcome]:
                    outcome = pdf_process.PdfBatchInspectionOutcome(
                        path=paths[0],
                        evidence=None,
                        error_status="pdf_processing_error",
                        error_note="synthetic failure",
                    )
                    progress = kwargs["progress"]
                    assert callable(progress)
                    progress(1, 1, outcome)
                    return [outcome]

                with cache:
                    with mock.patch.object(
                        dedupe_run.pdf_process,
                        "inspect_pdf_batch",
                        side_effect=fail_batch,
                    ):
                        identities = dedupe_run.inspect_pdfs(
                            [pdf],
                            workers=1,
                            cache=cache,
                        )
                    self.assertIsNone(cache.get(pdf))

                self.assertEqual(
                    identities[0].pdf_status,
                    "pdf_processing_error",
                )
                self.assertEqual(cache.write_count, 0)


    def test_replace_pdf_freshly_inspects_despite_a_shared_cache_hit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "collection"
            root.mkdir()
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"synthetic PDF identity")

            writer = pdf_identity_cache.IdentityCache(
                root,
                extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION,
                temp_root=base / "rename-local",
            )
            with writer:
                writer.put(pdf, _payload(pdf))
                writer.publish()

            dedupe_cache = pdf_identity_cache.IdentityCache(
                root,
                extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION,
                temp_root=base / "dedupe-local",
            )
            with dedupe_cache:
                with mock.patch.object(
                    dedupe_run.pdf_process,
                    "inspect_pdf_batch",
                ) as inspect_batch:
                    identities = dedupe_run.inspect_pdfs(
                        [pdf],
                        cache=dedupe_cache,
                    )
            inspect_batch.assert_not_called()
            self.assertEqual(identities[0].doi, "10.1000/cache-test")

            replace_cache = pdf_identity_cache.IdentityCache(
                root,
                extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION,
                temp_root=base / "replace-local",
            )
            with replace_cache:
                self.assertIsNotNone(replace_cache.get(pdf))

                def inspect_batch(requested, **kwargs):
                    requested_paths = list(requested)
                    fresh = pdf_evidence.PdfEvidence(
                        dois=[],
                        title="Fresh audit-selected source evidence",
                        pii=None,
                        arxiv_doi=None,
                    )
                    outcome = pdf_process.PdfBatchInspectionOutcome(
                        path=requested_paths[0],
                        evidence=fresh,
                        worker_pid=101,
                    )
                    kwargs["progress"](1, 1, outcome)
                    return [outcome]

                with mock.patch.object(
                    replacement_run.pdf_process,
                    "inspect_pdf_batch",
                    side_effect=inspect_batch,
                ) as inspect_batch:
                    evidence = replacement_run.inspect_candidate_sources(
                        [pdf],
                    )
            inspect_batch.assert_called_once()
            self.assertEqual(
                evidence[pdf].title,
                "Fresh audit-selected source evidence",
            )

    def test_main_publishes_snapshot_only_after_report_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "collection"
            root.mkdir()
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"synthetic PDF identity")
            snapshot = root / pdf_identity_cache.CACHE_FILENAME
            real_cache_class = pdf_identity_cache.IdentityCache
            real_write_report = dedupe_run.write_report

            def cache_factory(
                collection_root: Path,
                *,
                extractor_version: str,
            ) -> pdf_identity_cache.IdentityCache:
                return real_cache_class(
                    collection_root,
                    extractor_version=extractor_version,
                    temp_root=base / "local",
                )

            def inspect_without_pymupdf(
                paths: list[Path],
                **_kwargs: object,
            ) -> list[dedupe_run.PdfIdentity]:
                return [
                    dedupe_run.PdfIdentity(
                        path=path,
                        doi="10.1000/cache-test",
                        doi_source="text:p1",
                        doi_score=100,
                        title="A Persistent PDF Identity Cache Test",
                    )
                    for path in paths
                ]

            def write_then_publish(
                results: list[dict[str, str]],
                path: Path,
            ) -> None:
                self.assertFalse(snapshot.exists())
                real_write_report(results, path)

            with (
                mock.patch.object(
                    dedupe_run.pdf_identity_cache,
                    "IdentityCache",
                    side_effect=cache_factory,
                ),
                mock.patch.object(
                    dedupe_run,
                    "inspect_pdfs",
                    side_effect=inspect_without_pymupdf,
                ),
                mock.patch.object(
                    dedupe_run,
                    "write_report",
                    side_effect=write_then_publish,
                ),
            ):
                exit_code = dedupe_cli.main([str(root)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(snapshot.is_file())


class RenameIdentityCacheWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_post_commit_evidence_tracks_final_file(self) -> None:
        with self.subTest(case="compression_only_commit_refreshes_evidence_for_new_bytes"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "collection"
                root.mkdir()
                target = root / "paper.pdf"
                target.write_bytes(b"compressed bytes")
                old = pdf_evidence.PdfEvidence([], "Before compression", None, None)
                fresh = pdf_evidence.PdfEvidence([], "Fresh compressed evidence", None, None)
                analysis = organization_run._Analysis(organization_run.OrganizationOutcome(target), evidence=old, compress_if_large=True)
                outcome = organization_run.OrganizationOutcome(target, target_path=target, pdf_status="pdf_compressed")
                runner = mock.Mock(inspect=mock.AsyncMock(return_value=fresh))
                with pdf_identity_cache.IdentityCache(root, extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION, temp_root=Path(directory) / "cache") as cache:
                    await organization_run._update_identity_cache_after_commit([analysis], [outcome], organization_run.RunOptions(root, commit=True), runner, cache)
                    self.assertEqual(pdf_evidence.evidence_from_cache_payload(cache.get(target)).title, fresh.title)
                runner.inspect.assert_awaited_once()

        with self.subTest(case="post_commit_moves_clean_evidence_to_final_path"):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = (base / "collection").resolve()
                root.mkdir()
                source = root / "download.pdf"
                target = root / "Author_Journal24_Topic.pdf"
                source.write_bytes(b"stable cached PDF bytes")
                source.rename(target)
                evidence = pdf_evidence.PdfEvidence(
                    dois=[pdf_evidence.DoiCandidate(DOI, "text:p1", 100)],
                    title="Stable paper title",
                    pii=None,
                    arxiv_doi=None,
                )
                analysis = organization_run._Analysis(
                    outcome=organization_run.OrganizationOutcome(source),
                    evidence=evidence,
                )
                outcome = organization_run.OrganizationOutcome(
                    source,
                    target_path=target,
                )
                cache = pdf_identity_cache.IdentityCache(
                    root,
                    extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION,
                    temp_root=base / "local",
                )
                runner = mock.Mock()
                runner.inspect = mock.AsyncMock(
                    side_effect=AssertionError("clean PDF should not be reopened")
                )

                with cache:
                    await organization_run._update_identity_cache_after_commit(
                        [analysis],
                        [outcome],
                        organization_run.RunOptions(root, commit=True),
                        runner,
                        cache,
                    )
                    cached = cache.get(target)

                runner.inspect.assert_not_awaited()
                self.assertIsNotNone(cached)
                assert cached is not None
                self.assertEqual(
                    pdf_evidence.evidence_from_cache_payload(cached).title,
                    "Stable paper title",
                )

        with self.subTest(case="post_commit_reinspects_normalized_final_path"):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = (base / "collection").resolve()
                root.mkdir()
                source = root / "covered.pdf"
                target = root / "Author_Journal24_Normalized.pdf"
                target.write_bytes(b"normalized PDF bytes")
                initial = pdf_evidence.PdfEvidence(
                    dois=[],
                    title="Wrapped title",
                    pii=None,
                    arxiv_doi=None,
                    pdf_status="pdf_cover_page_removal_needed",
                    inserted_cover_pages=(
                        pdf_evidence.InsertedCoverPage(
                            0,
                            "publisher",
                            "synthetic wrapper",
                        ),
                    ),
                )
                refreshed = pdf_evidence.PdfEvidence(
                    dois=[pdf_evidence.DoiCandidate(DOI, "text:p1", 100)],
                    title="Normalized title",
                    pii=None,
                    arxiv_doi=None,
                )
                analysis = organization_run._Analysis(
                    outcome=organization_run.OrganizationOutcome(source),
                    evidence=initial,
                    requires_normalization=True,
                )
                outcome = organization_run.OrganizationOutcome(
                    source,
                    target_path=target,
                )
                cache = pdf_identity_cache.IdentityCache(
                    root,
                    extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION,
                    temp_root=base / "local",
                )
                runner = mock.Mock()
                runner.inspect = mock.AsyncMock(return_value=refreshed)

                with cache:
                    await organization_run._update_identity_cache_after_commit(
                        [analysis],
                        [outcome],
                        organization_run.RunOptions(root, commit=True),
                        runner,
                        cache,
                    )
                    cached = cache.get(target)

                runner.inspect.assert_awaited_once_with(
                    target,
                    pdf_password="",
                )
                self.assertIsNotNone(cached)
                assert cached is not None
                cached_evidence = pdf_evidence.evidence_from_cache_payload(cached)
                self.assertEqual(cached_evidence.pdf_status, "ok")
                self.assertEqual(cached_evidence.title, "Normalized title")


    async def test_cached_evidence_reuse_and_reinspection(self) -> None:
        with self.subTest(case="run_reuses_cached_pdf_evidence"):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = (base / "collection").resolve()
                root.mkdir()
                source = root / "paper.pdf"
                source.write_bytes(b"stable cached PDF bytes")
                evidence = pdf_evidence.PdfEvidence(
                    dois=[
                        pdf_evidence.DoiCandidate(
                            DOI,
                            "metadata:subject",
                            120,
                        )
                    ],
                    title=None,
                    pii=None,
                    arxiv_doi=None,
                )
                cache = pdf_identity_cache.IdentityCache(
                    root,
                    extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION,
                    temp_root=base / "local",
                    snapshot_filename=RenamePDF.IDENTITY_CACHE_FILENAME,
                    local_cache_namespace=RenamePDF.IDENTITY_CACHE_NAMESPACE,
                )
                runner = mock.Mock()
                runner.inspect = mock.AsyncMock(
                    side_effect=AssertionError("worker should not run")
                )
                providers = _no_match_providers()

                with cache:
                    cache.put(
                        source,
                        pdf_evidence.evidence_to_cache_payload(evidence),
                    )
                    outcomes = await organization_run.run(
                        organization_run.RunOptions(root),
                        providers=providers,
                        pdf_runner=runner,
                        identity_cache=cache,
                    )

                runner.inspect.assert_not_awaited()
                providers.records_by_doi.assert_awaited_once_with(DOI)
                self.assertEqual(cache.hit_count, 1)
                self.assertEqual(outcomes[0].doi, DOI)

        with self.subTest(case="run_reinspects_blocking_cached_evidence"):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = (base / "collection").resolve()
                root.mkdir()
                source = root / "paper.pdf"
                source.write_bytes(b"stable cached PDF bytes")
                blocked = pdf_evidence.PdfEvidence(
                    dois=[],
                    title=None,
                    pii=None,
                    arxiv_doi=None,
                    pdf_status="pdf_unreadable",
                    pdf_note="synthetic cached failure",
                )
                valid = pdf_evidence.PdfEvidence(
                    dois=[
                        pdf_evidence.DoiCandidate(
                            DOI,
                            "metadata:subject",
                            120,
                        )
                    ],
                    title=None,
                    pii=None,
                    arxiv_doi=None,
                )
                cache = pdf_identity_cache.IdentityCache(
                    root,
                    extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION,
                    temp_root=base / "local",
                    snapshot_filename=RenamePDF.IDENTITY_CACHE_FILENAME,
                    local_cache_namespace=RenamePDF.IDENTITY_CACHE_NAMESPACE,
                )
                runner = mock.Mock()
                runner.inspect = mock.AsyncMock(return_value=valid)
                providers = _no_match_providers()

                with cache:
                    cache.put(
                        source,
                        pdf_evidence.evidence_to_cache_payload(blocked),
                    )
                    outcomes = await organization_run.run(
                        organization_run.RunOptions(root),
                        providers=providers,
                        pdf_runner=runner,
                        identity_cache=cache,
                    )
                    refreshed = cache.get(source)

                runner.inspect.assert_awaited_once_with(
                    source,
                    pdf_password="",
                )
                self.assertIsNotNone(refreshed)
                assert refreshed is not None
                self.assertEqual(
                    pdf_evidence.evidence_from_cache_payload(
                        refreshed
                    ).pdf_status,
                    "ok",
                )
                providers.records_by_doi.assert_awaited_once_with(DOI)
                self.assertEqual(outcomes[0].doi, DOI)


    def test_main_publishes_identity_cache_after_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "collection"
            root.mkdir()
            report = root / "0_results.csv"
            snapshot = root / RenamePDF.IDENTITY_CACHE_FILENAME
            row = dict.fromkeys(organization_run.CSV_FIELDS, "")
            row.update(
                {
                    "file": str(root / "paper.pdf"),
                    "new_name": "paper.pdf",
                    "status": "no_doi_or_title_found",
                }
            )
            real_cache_class = pdf_identity_cache.IdentityCache
            real_publish = RenamePDF._publish_identity_cache

            def cache_factory(
                collection_root: Path,
                **kwargs: object,
            ) -> pdf_identity_cache.IdentityCache:
                return real_cache_class(
                    collection_root,
                    temp_root=base / "local",
                    **kwargs,
                )

            def publish_after_report(
                cache: pdf_identity_cache.IdentityCache,
            ) -> None:
                self.assertTrue(report.is_file())
                real_publish(cache)

            with (
                mock.patch.object(
                    RenamePDF.identity_cache_storage,
                    "IdentityCache",
                    side_effect=cache_factory,
                ),
                mock.patch.object(
                    RenamePDF,
                    "driver",
                    mock.AsyncMock(return_value=[row]),
                ) as driver,
                mock.patch.object(
                    RenamePDF,
                    "_publish_identity_cache",
                    side_effect=publish_after_report,
                ),
            ):
                exit_code = RenamePDF.main([str(root)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(snapshot.is_file())
            identity_cache = driver.await_args.kwargs["identity_cache"]
            self.assertIsNotNone(identity_cache)


class ReplacePDFIdentityCacheWorkflowTests(unittest.TestCase):
    def test_source_inspection_always_uses_the_worker_pool(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "first.pdf", root / "second.pdf"]
            for path in paths:
                path.write_bytes(b"test PDF placeholder")

            def inspect_batch(requested, **kwargs):
                requested_paths = list(requested)
                self.assertEqual(requested_paths, paths)
                outcomes = []
                for index, path in enumerate(requested_paths, start=1):
                    evidence = pdf_evidence.PdfEvidence(
                        dois=[],
                        title=f"Source title {index}",
                        pii=None,
                        arxiv_doi=None,
                    )
                    outcome = pdf_process.PdfBatchInspectionOutcome(
                        path=path,
                        evidence=evidence,
                        worker_pid=100 + index,
                    )
                    outcomes.append(outcome)
                    kwargs["progress"](
                        index,
                        len(requested_paths),
                        outcome,
                    )
                return outcomes

            with mock.patch.object(
                replacement_run.pdf_process,
                "inspect_pdf_batch",
                side_effect=inspect_batch,
            ) as inspector:
                first = replacement_run.inspect_candidate_sources(paths)
                second = replacement_run.inspect_candidate_sources(paths)

            self.assertEqual(inspector.call_count, 2)
            self.assertEqual(first, second)

    def test_main_publishes_identity_snapshot_after_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = mock.Mock()
            cache.usable = True
            events: list[str] = []

            with (
                mock.patch.object(
                    replacement_run,
                    "_open_identity_cache",
                    return_value=cache,
                ),
                mock.patch.object(
                    replacement_run,
                    "load_audit_candidates",
                    return_value=[],
                ),
                mock.patch.object(
                    replacement_run,
                    "ApiClient",
                ) as client,
                mock.patch.object(
                    replacement_run,
                    "write_report",
                    side_effect=lambda *_args: events.append("report"),
                ),
                mock.patch.object(
                    replacement_run,
                    "_publish_identity_cache",
                    side_effect=lambda *_args: events.append("snapshot"),
                ),
            ):
                exit_code = replace_cli.main(
                    [str(root), "--email", "test@example.test"]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(events, ["report", "snapshot"])
            client.assert_not_called()
            cache.close.assert_called_once_with()


class SnapshotPublicationTests(unittest.TestCase):
    def test_fallback_cache_namespace_is_user_scoped_not_process_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(pdf_identity_cache, "os", wraps=os) as platform,
                mock.patch.object(pdf_identity_cache.getpass, "getuser", return_value="example-user"),
            ):
                del platform.getuid
                platform.getpid.return_value = 123
                first = pdf_identity_cache.IdentityCache(Path(directory), extractor_version="test")
                platform.getpid.return_value = 456
                second = pdf_identity_cache.IdentityCache(Path(directory), extractor_version="test")
                self.assertEqual(first.local_path, second.local_path)

    def test_publication_lock_survives_success_and_failure(self) -> None:
        with self.subTest(case="publication_lock_covers_backup_through_final_copy"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                first_pdf, second_pdf = root / "first.pdf", root / "second.pdf"
                first_pdf.write_bytes(b"first fixture")
                second_pdf.write_bytes(b"second fixture")
                options = dict(extractor_version="test-publication", temp_root=root / "local", commit_interval=1)
                ready, release = threading.Event(), threading.Event()
                def first_publisher():
                    with pdf_identity_cache.IdentityCache(root, **options) as cache:
                        cache.put(first_pdf, {"title": "First fixture"})
                        copy = cache._copy_file_atomic
                        def paused_copy(source, destination):
                            ready.set()
                            if not release.wait(10):
                                raise TimeoutError("test publisher was not released")
                            copy(source, destination)
                        with mock.patch.object(cache, "_copy_file_atomic", side_effect=paused_copy):
                            cache.publish()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    pending = executor.submit(first_publisher)
                    try:
                        self.assertTrue(ready.wait(10))
                        with pdf_identity_cache.IdentityCache(root, **options) as second:
                            second.put(second_pdf, {"title": "Second fixture"})
                            lock = file_lock.local_lock
                            with mock.patch.object(file_lock, "local_lock", side_effect=lambda resource: lock(resource, timeout=0)):
                                with self.assertRaises(TimeoutError):
                                    second.publish()
                            self.assertIsNotNone(second.get(second_pdf))
                            release.set()
                            pending.result(timeout=10)
                            snapshot = second.publish()
                            with closing(sqlite3.connect(snapshot)) as connection:
                                paths = {row[0] for row in connection.execute("SELECT relative_path FROM identity_cache")}
                            self.assertEqual(paths, {"first.pdf", "second.pdf"})
                    finally:
                        release.set()
                        pending.result(timeout=10)

        with self.subTest(case="failed_publication_preserves_snapshot_and_releases_lock"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pdf = root / "paper.pdf"
                pdf.write_bytes(b"fixture")
                with pdf_identity_cache.IdentityCache(root, extractor_version="test", temp_root=root / "local") as cache:
                    cache.put(pdf, {"title": "Original fixture"})
                    snapshot = cache.publish()
                    original = snapshot.read_bytes()
                    cache.put(pdf, {"title": "Updated fixture"})
                    with mock.patch.object(cache, "_copy_file_atomic", side_effect=OSError("injected copy failure")):
                        with self.assertRaises(OSError):
                            cache.publish()
                    self.assertEqual(snapshot.read_bytes(), original)
                    cache.publish()
                    self.assertEqual(cache.get(pdf)["title"], "Updated fixture")


    def test_busy_database_is_not_treated_as_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with pdf_identity_cache.IdentityCache(root, extractor_version="test", temp_root=root / "local") as cache:
                error = sqlite3.OperationalError("database is locked")
                error.sqlite_errorcode = sqlite3.SQLITE_BUSY
                with mock.patch.object(sqlite3, "connect", side_effect=error):
                    with self.assertRaises(sqlite3.OperationalError):
                        cache._metadata_matches(cache.local_path)
                self.assertTrue(cache.local_path.exists())


class PdfWorkerReuseTests(unittest.TestCase):
    def test_compact_worker_payload_preserves_structured_evidence(self) -> None:
        evidence = pdf_evidence.PdfEvidence(
            [pdf_evidence.DoiCandidate("10.1000/example", "metadata:doi", 120)],
            "Synthetic Worker Evidence", None, None,
            needs_repair=True, needs_decryption=True, has_signatures=True,
            inserted_cover_pages=(pdf_evidence.InsertedCoverPage(0, "test", "fixture"),),
            text_samples=("large synthetic extracted text " * 10_000,),
        )
        with mock.patch.object(pdf_evidence, "inspect_pdf", return_value=evidence):
            full = pdf_process._inspect_pdf_batch_worker("paper.pdf", "", 5)
            compact = pdf_process._inspect_pdf_batch_worker("paper.pdf", "", 5, False)
        self.assertIs(full.evidence, evidence)
        self.assertEqual(compact.evidence.text_samples, ())
        self.assertEqual(pdf_evidence.evidence_to_cache_payload(full.evidence), pdf_evidence.evidence_to_cache_payload(compact.evidence))
        self.assertLess(len(pickle.dumps(compact)), len(pickle.dumps(full)) / 10)

    def test_one_spawned_worker_is_reused_for_multiple_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "one.pdf", root / "two.pdf"]
            for index, path in enumerate(paths, start=1):
                document = pymupdf.open()
                page = document.new_page()
                page.insert_text(
                    (72, 72),
                    "Reusable Spawned PDF Worker\n"
                    f"https://doi.org/10.1000/worker-{index}",
                )
                document.save(path)
                document.close()

            outcomes = pdf_process.inspect_pdf_batch(
                paths,
                include_text_samples=False,
                max_workers=1,
                timeout=15,
                max_tasks_per_child=64,
            )

            self.assertEqual(len(outcomes), 2)
            self.assertTrue(all(item.evidence is not None for item in outcomes))
            worker_pids = {item.worker_pid for item in outcomes}
            self.assertEqual(len(worker_pids), 1)
            self.assertNotIn(None, worker_pids)
            self.assertNotIn(os.getpid(), worker_pids)
            self.assertTrue(all(item.evidence.text_samples == () for item in outcomes))
            self.assertEqual([item.evidence.dois[0].doi for item in outcomes], ["10.1000/worker-1", "10.1000/worker-2"])


if __name__ == "__main__":
    unittest.main()
