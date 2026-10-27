"""Focused tests consolidated by validation surface."""

import asyncio
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest import mock

import pymupdf

import RenamePDF
import audit_report
import collection_mutation
import file_lock
import file_metadata
import filename_policy
import publication_status
import metadata_providers
import organization_run
import pdf_evidence
import pdf_identity_cache
import pdf_rewrite
import publication_audit
import record_matching
import report_io


# No-clobber collection mutation

class ReplacementRecoveryTests(unittest.TestCase):
    def test_process_death_at_each_replacement_boundary_is_recoverable(self) -> None:
        script = '''
import os, sys
from pathlib import Path
import collection_mutation as mutation
root, stage, same_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "yes"
source, replacement = root / "source.pdf", root / "candidate.pdf"
target = source if same_path else root / "published.pdf"
move = mutation._rename_path
def interrupted_move(old, new):
    if stage == "before_archive" and old == source:
        os._exit(73)
    move(old, new)
    if (stage == "after_archive" and old == source) or (stage == "after_install" and old == replacement):
        os._exit(73)
mutation._rename_path = interrupted_move
mutation.install_replacement(root, source, replacement, target)
'''
        for same_path in (False, True):
            for stage in ("before_archive", "after_archive", "after_install"):
                with self.subTest(same_path=same_path, stage=stage), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = root / "source.pdf"
                    candidate = root / "candidate.pdf"
                    target = source if same_path else root / "published.pdf"
                    source.write_bytes(b"original synthetic PDF")
                    candidate.write_bytes(b"validated synthetic replacement")
                    process = subprocess.run(
                        [sys.executable, "-c", script, str(root), stage, "yes" if same_path else "no"],
                        cwd=Path(__file__).resolve().parent.parent,
                        capture_output=True, timeout=15,
                    )
                    self.assertEqual(process.returncode, 73, process.stderr)
                    journals = list(root.glob(".renamepdf-replacement-*.json"))
                    self.assertEqual(len(journals), 1)
                    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
                    pending = collection_mutation.recover_interrupted_replacements(root)
                    self.assertTrue(pending[0].blocks_run)
                    self.assertEqual(before, {path: path.read_bytes() for path in root.rglob("*") if path.is_file()})
                    recovered = collection_mutation.recover_interrupted_replacements(root, commit=True)
                    self.assertFalse(recovered[0].blocks_run, recovered[0].note)
                    if stage == "after_install":
                        self.assertEqual(target.read_bytes(), b"validated synthetic replacement")
                        self.assertEqual((root / "Superseded" / source.name).read_bytes(), b"original synthetic PDF")
                    else:
                        self.assertEqual(source.read_bytes(), b"original synthetic PDF")
                        self.assertEqual(candidate.read_bytes(), b"validated synthetic replacement")
                    self.assertEqual(collection_mutation.recover_interrupted_replacements(root, commit=True), ())

    def _interrupt(self, root: Path) -> tuple[Path, Path, Path]:
        source, replacement, target = (root / name for name in ("source.pdf", "candidate.pdf", "published.pdf"))
        source.write_bytes(b"original")
        replacement.write_bytes(b"replacement")
        move = collection_mutation._rename_path
        def interrupt(old, new):
            if old == replacement:
                raise KeyboardInterrupt
            move(old, new)
        with mock.patch.object(collection_mutation, "_rename_path", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                collection_mutation.install_replacement(root, source, replacement, target)
        return source, replacement, target

    def test_recovery_rejects_tampering_symlinks_and_changed_files(self) -> None:
        for damage in ("archive", "target", "symlink", "escaping_path", "malformed"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, candidate, target = self._interrupt(root)
                journal = next(root.glob(".renamepdf-replacement-*.json"))
                archive = root / "Superseded" / source.name
                if damage == "archive":
                    archive.write_bytes(b"changed externally")
                elif damage == "target":
                    target.write_bytes(b"unrelated concurrent file")
                elif damage == "symlink":
                    target.symlink_to(candidate)
                elif damage == "escaping_path":
                    record = json.loads(journal.read_text())
                    record["source"] = "../outside.pdf"
                    journal.write_text(json.dumps(record))
                else:
                    journal.write_text("{")
                before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
                outcome = collection_mutation.recover_interrupted_replacements(root, commit=True)[0]
                self.assertTrue(outcome.blocks_run)
                self.assertTrue(journal.exists())
                self.assertFalse(source.exists())
                self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_failed_recovery_preserves_originals_and_journal(self) -> None:
        with self.subTest(case="failed_restore_keeps_journal_and_original_archive"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, _candidate, _target = self._interrupt(root)
                with mock.patch.object(collection_mutation, "_rename_path", side_effect=OSError("restore blocked")):
                    outcome = collection_mutation.recover_interrupted_replacements(root, commit=True)[0]
                self.assertTrue(outcome.blocks_run)
                self.assertEqual((root / "Superseded" / source.name).read_bytes(), b"original")
                self.assertEqual(len(list(root.glob(".renamepdf-replacement-*.json"))), 1)
                self.assertFalse(collection_mutation.recover_interrupted_replacements(root, commit=True)[0].blocks_run)

        with self.subTest(case="recovery_cannot_overwrite_a_concurrent_original"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, _candidate, _target = self._interrupt(root)
                move = collection_mutation._rename_path
                def concurrent_move(old, new):
                    source.write_bytes(b"concurrent original")
                    move(old, new)
                with mock.patch.object(collection_mutation, "_rename_path", side_effect=concurrent_move):
                    outcome = collection_mutation.recover_interrupted_replacements(root, commit=True)[0]
                self.assertTrue(outcome.blocks_run)
                self.assertEqual(source.read_bytes(), b"concurrent original")
                self.assertEqual((root / "Superseded" / source.name).read_bytes(), b"original")


    def test_parent_scan_recovers_child_journal_but_not_a_live_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "nested-collection"
            child.mkdir()
            source, _candidate, _target = self._interrupt(child)
            with file_lock.local_lock(child / ".renamepdf-replacement-lock"):
                busy = collection_mutation.recover_interrupted_replacements(root, commit=True)
                self.assertEqual(busy[0].status, "replacement_recovery_busy")
                self.assertTrue(busy[0].blocks_run)
                self.assertFalse(source.exists())
            pending = collection_mutation.recover_interrupted_replacements(root)
            self.assertEqual(pending[0].status, "replacement_recovery_needed")
            completed = collection_mutation.recover_interrupted_replacements(root, commit=True)
            self.assertFalse(completed[0].blocks_run, completed[0].note)
            self.assertEqual(source.read_bytes(), b"original")

def _set_mtime(path: Path, nanoseconds: int) -> None:
    os.utime(path, ns=(nanoseconds, nanoseconds))


class CollectionMutationTests(unittest.TestCase):
    def test_native_noreplace_move_is_atomic_on_host_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            target = root / "target.pdf"
            source.write_bytes(b"source")
            try:
                collection_mutation._native_move_noreplace(source, target)
            except collection_mutation._NoReplaceUnavailable as exc:
                self.skipTest(str(exc))
            except OSError as exc:
                if exc.errno in collection_mutation._NATIVE_FALLBACK_ERRNOS:
                    self.skipTest(str(exc))
                raise

            self.assertFalse(source.exists())
            self.assertEqual(target.read_bytes(), b"source")

    def test_native_noreplace_move_refuses_existing_host_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            target = root / "target.pdf"
            source.write_bytes(b"source")
            target.write_bytes(b"target")
            try:
                collection_mutation._native_move_noreplace(source, target)
            except collection_mutation._NoReplaceUnavailable as exc:
                self.skipTest(str(exc))
            except FileExistsError:
                pass
            except OSError as exc:
                if exc.errno in collection_mutation._NATIVE_FALLBACK_ERRNOS:
                    self.skipTest(str(exc))
                raise
            else:
                self.fail("native no-replace move overwrote an existing path")

            self.assertEqual(source.read_bytes(), b"source")
            self.assertEqual(target.read_bytes(), b"target")

    def test_unique_moves_preserve_collisions(self) -> None:
        with self.subTest(case="copy_fallback_refuses_existing_path"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.pdf"
                target = root / "target.pdf"
                source.write_bytes(b"source")
                target.write_bytes(b"target")

                with (
                    mock.patch.object(
                        collection_mutation,
                        "_native_move_noreplace",
                        side_effect=collection_mutation._NoReplaceUnavailable(
                            "simulated unsupported filesystem"
                        ),
                    ),
                    self.assertRaises(FileExistsError),
                ):
                    collection_mutation._move_exclusive(source, target)

                self.assertEqual(source.read_bytes(), b"source")
                self.assertEqual(target.read_bytes(), b"target")

        with self.subTest(case="move_to_unique_preserves_existing_destination"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "incoming.pdf"
                destination = root / "Duplicated"
                existing = destination / "Paper.pdf"
                destination.mkdir()
                source.write_bytes(b"incoming")
                existing.write_bytes(b"existing")

                target = collection_mutation.move_to_unique(
                    source,
                    destination,
                    "Paper.pdf",
                )

                self.assertEqual(target, destination / "Paper_1.pdf")
                self.assertEqual(existing.read_bytes(), b"existing")
                self.assertEqual(target.read_bytes(), b"incoming")

        with self.subTest(case="move_to_unique_retries_collision_created_at_final_move"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "incoming.pdf"
                destination = root / "Duplicated"
                destination.mkdir()
                source.write_bytes(b"incoming")
                raced = False
                real_move = collection_mutation._move_exclusive

                def inject_collision(move_source: Path, move_target: Path) -> None:
                    nonlocal raced
                    if not raced and move_target.name == "Paper.pdf":
                        raced = True
                        move_target.write_bytes(b"concurrent")
                    real_move(move_source, move_target)

                with mock.patch.object(
                    collection_mutation,
                    "_move_exclusive",
                    side_effect=inject_collision,
                ):
                    target = collection_mutation.move_to_unique(
                        source,
                        destination,
                        "Paper.pdf",
                    )

                self.assertEqual(target, destination / "Paper_1.pdf")
                self.assertEqual((destination / "Paper.pdf").read_bytes(), b"concurrent")
                self.assertEqual(target.read_bytes(), b"incoming")


    def test_exclusive_move_preservation(self) -> None:
        with self.subTest(case="failed_exclusive_copy_removes_partial_and_preserves_source"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "incoming.pdf"
                destination = root / "Duplicated"
                source.write_bytes(b"complete source")

                def fail_after_partial_copy(input_stream, output_stream, **_kwargs):
                    input_stream.read(1)
                    output_stream.write(b"partial")
                    output_stream.flush()
                    raise OSError("simulated copy failure")

                with (
                    mock.patch.object(
                        collection_mutation,
                        "_native_move_noreplace",
                        side_effect=collection_mutation._NoReplaceUnavailable(
                            "simulated unsupported filesystem"
                        ),
                    ),
                    mock.patch.object(
                        collection_mutation.shutil,
                        "copyfileobj",
                        side_effect=fail_after_partial_copy,
                    ),
                    self.assertRaises(collection_mutation.CollectionMutationError),
                ):
                    collection_mutation.move_to_unique(source, destination)

                self.assertEqual(source.read_bytes(), b"complete source")
                self.assertFalse((destination / source.name).exists())

        with self.subTest(case="exclusive_move_preserves_modification_time"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "incoming.pdf"
                source.write_bytes(b"incoming")
                _set_mtime(source, 1_234_567_890)
                # Filesystems may round timestamps (notably Windows temp volumes).
                expected = source.stat().st_mtime_ns

                with mock.patch.object(
                    collection_mutation,
                    "_native_move_noreplace",
                    side_effect=collection_mutation._NoReplaceUnavailable(
                        "simulated unsupported filesystem"
                    ),
                ):
                    target = collection_mutation.move_to_unique(
                        source,
                        root / "Duplicated",
                    )

                self.assertEqual(target.stat().st_mtime_ns, expected)


    @unittest.skipUnless(sys.platform == "darwin", "requires macOS metadata")
    def test_copy_fallback_preserves_macos_extended_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "incoming.pdf"
            source.write_bytes(b"incoming")
            subprocess.run(
                [
                    "/usr/bin/xattr",
                    "-w",
                    "com.example.renamepdf",
                    "retained",
                    str(source),
                ],
                check=True,
            )

            with mock.patch.object(
                collection_mutation,
                "_native_move_noreplace",
                side_effect=collection_mutation._NoReplaceUnavailable(
                    "simulated cross-volume move"
                ),
            ):
                target = collection_mutation.move_to_unique(
                    source,
                    root / "Duplicated",
                )

            value = subprocess.run(
                [
                    "/usr/bin/xattr",
                    "-p",
                    "com.example.renamepdf",
                    str(target),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(value, "retained")

    def test_canonical_copy_selection_preserves_both_files(self) -> None:
        with self.subTest(case="canonical_collision_at_final_move_preserves_both_files"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "incoming.pdf"
                canonical = root / "Paper.pdf"
                source.write_bytes(b"incoming")
                real_move = collection_mutation._move_exclusive

                def inject_collision(move_source: Path, move_target: Path) -> None:
                    if move_target == canonical and not canonical.exists():
                        canonical.write_bytes(b"concurrent")
                    real_move(move_source, move_target)

                with (
                    mock.patch.object(
                        collection_mutation,
                        "_move_exclusive",
                        side_effect=inject_collision,
                    ),
                    self.assertRaises(collection_mutation.CollectionMutationError),
                ):
                    collection_mutation.place_canonical(
                        source,
                        canonical,
                        root / "Duplicated",
                    )

                self.assertEqual(source.read_bytes(), b"incoming")
                self.assertEqual(canonical.read_bytes(), b"concurrent")

        with self.subTest(case="newer_source_replaces_canonical_and_preserves_old_copy"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                canonical = root / "Paper.pdf"
                source = root / "incoming.pdf"
                canonical.write_bytes(b"old")
                source.write_bytes(b"new")
                _set_mtime(canonical, 1_000_000_000)
                _set_mtime(source, 2_000_000_000)

                outcome = collection_mutation.place_canonical(
                    source,
                    canonical,
                    root / "Duplicated",
                )

                self.assertEqual(outcome.action, "canonical_replaced")
                self.assertEqual(canonical.read_bytes(), b"new")
                self.assertIsNotNone(outcome.displaced)
                self.assertEqual(outcome.displaced.read_bytes(), b"old")

        with self.subTest(case="older_source_is_preserved_as_duplicate"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                canonical = root / "Paper.pdf"
                source = root / "incoming.pdf"
                canonical.write_bytes(b"new")
                source.write_bytes(b"old")
                _set_mtime(canonical, 2_000_000_000)
                _set_mtime(source, 1_000_000_000)

                outcome = collection_mutation.place_canonical(
                    source,
                    canonical,
                    root / "Duplicated",
                )

                self.assertEqual(outcome.action, "duplicate")
                self.assertEqual(canonical.read_bytes(), b"new")
                self.assertEqual(outcome.target.read_bytes(), b"old")


    def test_same_file_alias_is_normalized_without_creating_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "paper.pdf"
            canonical = root / "Paper.pdf"
            source.write_bytes(b"paper")
            if not canonical.exists():
                os.link(source, canonical)

            outcome = collection_mutation.place_canonical(
                source,
                canonical,
                root / "Duplicated",
            )

            expected_action = "unchanged" if sys.platform == "win32" else "canonical"
            self.assertEqual(outcome.action, expected_action)
            self.assertEqual(canonical.read_bytes(), b"paper")
            pdf_names = [
                path.name for path in root.iterdir() if path.suffix.casefold() == ".pdf"
            ]
            self.assertEqual(len(pdf_names), 1)
            self.assertEqual(pdf_names[0].casefold(), "paper.pdf")
            self.assertFalse((root / "Duplicated").exists())

    def test_canonical_swap_rollback(self) -> None:
        with self.subTest(case="failed_canonical_install_restores_existing_canonical"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                canonical = root / "Paper.pdf"
                source = root / "incoming.pdf"
                canonical.write_bytes(b"old")
                source.write_bytes(b"new")
                _set_mtime(canonical, 1_000_000_000)
                _set_mtime(source, 2_000_000_000)
                real_rename = collection_mutation._rename_path

                def fail_install(rename_source: Path, rename_target: Path) -> None:
                    if rename_source == source and rename_target == canonical:
                        raise OSError("injected install failure")
                    real_rename(rename_source, rename_target)

                with mock.patch.object(
                    collection_mutation,
                    "_rename_path",
                    side_effect=fail_install,
                ):
                    with self.assertRaises(collection_mutation.CollectionMutationError):
                        collection_mutation.place_canonical(
                            source,
                            canonical,
                            root / "Duplicated",
                        )

                self.assertEqual(canonical.read_bytes(), b"old")
                self.assertEqual(source.read_bytes(), b"new")
                self.assertFalse(any((root / "Duplicated").iterdir()))

        with self.subTest(case="failed_swap_restores_both_paths"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                duplicate_directory = root / "Duplicated"
                duplicate_directory.mkdir()
                canonical = root / "Paper.pdf"
                duplicate = duplicate_directory / "Paper.pdf"
                canonical.write_bytes(b"old")
                duplicate.write_bytes(b"new")
                _set_mtime(canonical, 1_000_000_000)
                _set_mtime(duplicate, 2_000_000_000)
                real_rename = collection_mutation._rename_path
                failed = False

                def fail_preserve(rename_source: Path, rename_target: Path) -> None:
                    nonlocal failed
                    if (
                        not failed
                        and rename_source.name.startswith(".Paper.pdf.canonical-swap")
                        and rename_target == duplicate
                    ):
                        failed = True
                        raise OSError("injected preserve failure")
                    real_rename(rename_source, rename_target)

                with mock.patch.object(
                    collection_mutation,
                    "_rename_path",
                    side_effect=fail_preserve,
                ):
                    with self.assertRaises(collection_mutation.CollectionMutationError):
                        collection_mutation.prefer_newest_as_canonical(
                            canonical,
                            duplicate,
                        )

                self.assertEqual(canonical.read_bytes(), b"old")
                self.assertEqual(duplicate.read_bytes(), b"new")
                self.assertFalse(
                    any(path.name.startswith(".") for path in duplicate_directory.iterdir())
                )

        with self.subTest(case="swap_collision_at_promotion_preserves_every_copy"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                duplicate_directory = root / "Duplicated"
                duplicate_directory.mkdir()
                canonical = root / "Paper.pdf"
                duplicate = duplicate_directory / "Paper.pdf"
                canonical.write_bytes(b"old")
                duplicate.write_bytes(b"new")
                _set_mtime(canonical, 1_000_000_000)
                _set_mtime(duplicate, 2_000_000_000)
                real_move = collection_mutation._move_exclusive

                def inject_collision(move_source: Path, move_target: Path) -> None:
                    if move_source == duplicate and move_target == canonical:
                        canonical.write_bytes(b"concurrent")
                    real_move(move_source, move_target)

                with (
                    mock.patch.object(
                        collection_mutation,
                        "_move_exclusive",
                        side_effect=inject_collision,
                    ),
                    self.assertRaises(collection_mutation.CollectionRollbackError),
                ):
                    collection_mutation.prefer_newest_as_canonical(
                        canonical,
                        duplicate,
                    )

                self.assertEqual(canonical.read_bytes(), b"concurrent")
                self.assertEqual(duplicate.read_bytes(), b"new")
                preserved = list(duplicate_directory.glob(".*canonical-swap*"))
                self.assertEqual(len(preserved), 1)
                self.assertEqual(preserved[0].read_bytes(), b"old")


    def test_replacement_rollback_and_no_clobber(self) -> None:
        with self.subTest(case="failed_replacement_install_restores_original"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "preprint.pdf"
                replacement = root / "download.pdf"
                target = root / "published.pdf"
                source.write_bytes(b"preprint")
                replacement.write_bytes(b"published")
                real_rename = collection_mutation._rename_path

                def fail_install(rename_source: Path, rename_target: Path) -> None:
                    if rename_source == replacement and rename_target == target:
                        raise OSError("injected install failure")
                    real_rename(rename_source, rename_target)

                with mock.patch.object(
                    collection_mutation,
                    "_rename_path",
                    side_effect=fail_install,
                ):
                    with self.assertRaises(collection_mutation.CollectionMutationError):
                        collection_mutation.install_replacement(
                            root,
                            source,
                            replacement,
                            target,
                        )

                self.assertEqual(source.read_bytes(), b"preprint")
                self.assertEqual(replacement.read_bytes(), b"published")
                self.assertFalse(target.exists())
                self.assertFalse(any((root / "Superseded").iterdir()))

        with self.subTest(case="replacement_refuses_to_overwrite_existing_target"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "preprint.pdf"
                replacement = root / "download.pdf"
                target = root / "published.pdf"
                source.write_bytes(b"preprint")
                replacement.write_bytes(b"published")
                target.write_bytes(b"existing")

                with self.assertRaises(collection_mutation.CollectionMutationError):
                    collection_mutation.install_replacement(
                        root,
                        source,
                        replacement,
                        target,
                    )

                self.assertEqual(source.read_bytes(), b"preprint")
                self.assertEqual(replacement.read_bytes(), b"published")
                self.assertEqual(target.read_bytes(), b"existing")
                self.assertFalse((root / "Superseded").exists())

        with self.subTest(case="replacement_collision_at_install_restores_original"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "preprint.pdf"
                replacement = root / "download.pdf"
                target = root / "published.pdf"
                source.write_bytes(b"preprint")
                replacement.write_bytes(b"published")
                real_move = collection_mutation._move_exclusive

                def inject_collision(move_source: Path, move_target: Path) -> None:
                    if move_source == replacement and move_target == target:
                        target.write_bytes(b"concurrent")
                    real_move(move_source, move_target)

                with (
                    mock.patch.object(
                        collection_mutation,
                        "_move_exclusive",
                        side_effect=inject_collision,
                    ),
                    self.assertRaises(collection_mutation.CollectionMutationError),
                ):
                    collection_mutation.install_replacement(
                        root,
                        source,
                        replacement,
                        target,
                    )

                self.assertEqual(source.read_bytes(), b"preprint")
                self.assertEqual(replacement.read_bytes(), b"published")
                self.assertEqual(target.read_bytes(), b"concurrent")
                self.assertFalse(any((root / "Superseded").iterdir()))


    def test_destination_name_must_be_a_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            source.write_bytes(b"source")

            with self.assertRaises(ValueError):
                collection_mutation.move_to_unique(
                    source,
                    root / "Duplicated",
                    "../Paper.pdf",
                )


# RenamePDF organization workflow

TITLE = "Reliable Quantitative Imaging with Magnetic Resonance"
DOI = "10.1000/reliable"


def _paper_meta(
    doi: str = DOI,
    *,
    title: str = TITLE,
    family: str = "Example",
) -> record_matching.PaperMeta:
    author = record_matching.Author(given="A", family=family)
    return record_matching.PaperMeta(
        title=title,
        year=2024,
        journal_abbrev="Magn Reson Med",
        first_author=author,
        authors=[author],
        corresponding_author=author,
        doi=doi,
        pmid=None,
        author_keywords=["quantitative imaging"],
        mesh_keywords=[],
    )


def _create_pdf(
    path: Path,
    *,
    doi: str = DOI,
    title: str = TITLE,
    encrypted: bool = False,
) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), title)
    page.insert_text((72, 96), f"doi:{doi}")
    document.set_metadata(
        {
            "title": title,
            "subject": f"doi:{doi}",
            "author": "A Example",
        }
    )
    if encrypted:
        document.save(
            path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="",
        )
    else:
        document.save(path)
    document.close()


def _create_unresolved_pdf(path: Path, *, title: str) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), title)
    document.set_metadata({"title": title})
    document.save(path)
    document.close()


def _create_covered_pdf(path: Path) -> None:
    document = pymupdf.open()
    cover = document.new_page()
    cover.insert_textbox(
        pymupdf.Rect(72, 72, 540, 720),
        (
            "See discussions, stats, and author profiles for this publication "
            "at: https://www.researchgate.net/publication/123456\n"
            "All content following this page was uploaded by an author."
        ),
        fontsize=10,
    )
    article = document.new_page()
    article.insert_textbox(
        pymupdf.Rect(72, 72, 540, 720),
        (
            f"{TITLE}\nA. Example\nAbstract\n"
            f"A complete scientific article.\ndoi:{DOI}"
        ),
        fontsize=10,
    )
    document.set_metadata(
        {
            "title": TITLE,
            "subject": f"doi:{DOI}",
            "author": "A Example",
        }
    )
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


class _PubMedMemory:
    def __init__(self) -> None:
        self.remembered: list[tuple[str, str]] = []

    def remember_doi_pmid(self, doi: str, pmid: str) -> None:
        self.remembered.append((doi, pmid))


class _CrossrefAlternatives:
    @staticmethod
    async def records_by_alternative_id(
        _alternative_id: str,
    ) -> metadata_providers.ProviderLookup:
        return metadata_providers.ProviderLookup()


class FakeProviders:
    def __init__(
        self,
        records_factory=None,
        *,
        on_doi=None,
    ) -> None:
        self.pubmed = _PubMedMemory()
        self.crossref = _CrossrefAlternatives()
        self.calls: list[str] = []
        self.records_factory = records_factory or self._agreeing_records
        self.on_doi = on_doi

    @staticmethod
    def _agreeing_records(doi: str):
        meta = _paper_meta(doi)
        return (
            record_matching.ProviderRecord("pubmed", meta, "12345"),
            record_matching.ProviderRecord("crossref", _paper_meta(doi)),
            record_matching.ProviderRecord("datacite", _paper_meta(doi)),
        )

    async def records_by_doi(
        self,
        doi: str,
    ) -> metadata_providers.ProviderLookup:
        self.calls.append(doi)
        if self.on_doi:
            self.on_doi(doi)
        return metadata_providers.ProviderLookup(
            tuple(self.records_factory(doi))
        )

    @staticmethod
    async def records_by_title(
        _title: str,
    ) -> metadata_providers.ProviderLookup:
        return metadata_providers.ProviderLookup()


class FakeAuditor:
    def __init__(self, result: publication_audit.PublicationAudit) -> None:
        self.result = result
        self.calls: list[tuple[str | None, str | None, str | None]] = []

    async def audit_many(self, identities):
        values = tuple(identities)
        self.calls.extend(
            (identity.doi or None, identity.pmid or None, identity.arxiv_id or None)
            for identity in values
        )
        return {identity: self.result for identity in values}


class InlinePdfProcessRunner:
    def __init__(self) -> None:
        self.inspected: list[Path] = []
        self.rewritten: list[Path] = []

    async def inspect(
        self,
        pdf_path: Path,
        *,
        pdf_password: str = "",
        max_text_pages: int = 5,
    ) -> pdf_evidence.PdfEvidence:
        self.inspected.append(pdf_path)
        return pdf_evidence.inspect_pdf(
            pdf_path,
            pdf_password=pdf_password,
            max_text_pages=max_text_pages,
        )

    async def rewrite(
        self,
        pdf_path: Path,
        *,
        pdf_password: str = "",
        remove_page_indices: tuple[int, ...] = (),
        compress_if_large: bool = False,
        expected_source_token=None,
    ) -> pdf_rewrite.PdfRewriteOutcome:
        self.rewritten.append(pdf_path)
        return pdf_rewrite.rewrite_pdf(
            pdf_path,
            pdf_password=pdf_password,
            remove_page_indices=remove_page_indices,
            compress_if_large=compress_if_large,
            expected_source_token=expected_source_token,
        )


class RenameInspectionTimeoutTests(unittest.TestCase):
    def test_cli_and_library_timeout_validation(self) -> None:
        with self.subTest(case="cli_timeout_is_positive_finite_and_has_a_default"):
            self.assertEqual(RenamePDF._parse_args(["papers"]).inspection_timeout, 120.0)
            self.assertEqual(RenamePDF._parse_args(["papers", "--inspection-timeout", "900"]).inspection_timeout, 900.0)
            for value in ("0", "-1", "nan", "inf", "-inf", "not-a-number"):
                with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        RenamePDF._parse_args(["papers", f"--inspection-timeout={value}"])
                    self.assertEqual(error.exception.code, 2)

        with self.subTest(case="library_options_validate_timeout"):
            for options in (organization_run.RunOptions, organization_run.AuditOptions):
                self.assertEqual(options(Path("papers")).inspection_timeout, 120.0)
                self.assertEqual(options(Path("papers"), inspection_timeout=0.5).inspection_timeout, 0.5)
                for value in (0, -1, float("nan"), float("inf")):
                    with self.subTest(options=options, value=value), self.assertRaises(ValueError):
                        options(Path("papers"), inspection_timeout=value)


    def test_cli_passes_timeout_to_compatibility_driver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(RenamePDF, "_open_identity_cache", return_value=None),
                mock.patch.object(RenamePDF, "driver", new=mock.AsyncMock(return_value=[])) as driver,
            ):
                self.assertEqual(RenamePDF.main([directory, "--inspection-timeout", "900"]), 0)
            self.assertEqual(driver.call_args.kwargs["inspection_timeout"], 900.0)

class OrganizationRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_compression_restore_keeps_failure_and_places_verified_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "large.pdf"
            _create_pdf(source)
            _make_pdf_large_and_compressible(source)
            original = source.read_bytes()
            token = file_metadata.file_token(source)
            marker = pdf_rewrite._marker_payload

            def fail_install_marker(operation, *args, **kwargs):
                if operation == "install":
                    raise OSError("injected install marker failure")
                return marker(operation, *args, **kwargs)

            with mock.patch.object(pdf_rewrite, "_marker_payload", side_effect=fail_install_marker):
                outcome = (await organization_run.run(
                    organization_run.RunOptions(root, commit=True), providers=FakeProviders(),
                    pdf_runner=InlinePdfProcessRunner(),
                ))[0]

            self.assertIn(outcome.status, organization_run.RESOLVED_STATUSES)
            self.assertEqual(outcome.pdf_status, "pdf_compression_failed")
            self.assertFalse(outcome.placement_blocked)
            self.assertIn("injected install marker failure", outcome.pdf_note)
            target = outcome.target_path
            self.assertIsNotNone(target)
            self.assertEqual(target.name, outcome.new_name)
            self.assertEqual(target.read_bytes(), original)
            self.assertNotEqual(file_metadata.file_token(target), token)
            self.assertFalse(source.exists())
            self.assertFalse(list(root.glob(".*.renamepdf-*")))
            with pymupdf.open(target) as document:
                self.assertEqual(len(document), 1)
                for page in document:
                    page.get_text()

    async def test_inspection_timeout_reaches_workflow_runner(self) -> None:
        with self.subTest(case="timeout_reaches_owned_runner_for_organization_and_audit"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for audit in (False, True):
                    runner = mock.Mock(aclose=mock.AsyncMock())
                    with mock.patch.object(organization_run.pdf_process, "SpawnPdfProcessRunner", return_value=runner) as create:
                        if audit:
                            await organization_run.run_audit(
                                organization_run.AuditOptions(root, inspection_timeout=900),
                                providers=FakeProviders(),
                            )
                        else:
                            await organization_run.run(
                                organization_run.RunOptions(root, inspection_timeout=900),
                                providers=FakeProviders(),
                            )
                    create.assert_called_once_with(inspection_timeout=900.0)
                    runner.aclose.assert_awaited_once()

        with self.subTest(case="compatibility_driver_passes_timeout_to_run_options"):
            with mock.patch.object(organization_run, "run", new=mock.AsyncMock(return_value=[])) as run:
                await RenamePDF.driver(Path("papers"), inspection_timeout=900)
            self.assertEqual(run.call_args.args[0].inspection_timeout, 900.0)


    async def test_failed_rewrite_restoration_keeps_its_original_failure_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "covered.pdf"
            _create_covered_pdf(source)
            original = source.read_bytes()
            token = file_metadata.file_token(source)
            marker = pdf_rewrite._marker_payload
            def fail_install_marker(operation, *args, **kwargs):
                if operation == "install":
                    raise OSError("injected install marker failure")
                return marker(operation, *args, **kwargs)
            with mock.patch.object(pdf_rewrite, "_marker_payload", side_effect=fail_install_marker):
                outcome = (await organization_run.run(
                    organization_run.RunOptions(root, commit=True), providers=FakeProviders(),
                    pdf_runner=InlinePdfProcessRunner(),
                ))[0]
            self.assertEqual(outcome.status, "pdf_repair_failed")
            self.assertEqual(outcome.pdf_status, "pdf_repair_failed")
            self.assertTrue(outcome.placement_blocked)
            self.assertEqual(outcome.source_token, file_metadata.file_token(source))
            self.assertIn("injected install marker failure", outcome.note)
            self.assertEqual(source.read_bytes(), original)
            self.assertNotEqual(file_metadata.file_token(source), token)
            self.assertIsNone(outcome.target_path)
            self.assertFalse((root / "Failed").exists())
            with pymupdf.open(source) as document:
                self.assertEqual(len(document), 2)
                for page in document:
                    page.get_text()

    async def test_image_first_page_does_not_accept_a_page_two_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "article.pdf"
            with pymupdf.open() as document:
                first = document.new_page()
                pixmap = pymupdf.Pixmap(pymupdf.csGRAY, 32, 32, bytes([200]) * 1024, False)
                first.insert_image(first.rect, pixmap=pixmap)
                document.new_page().insert_textbox(
                    pymupdf.Rect(72, 72, 540, 720),
                    "Discussion. The results extend earlier observations.\n"
                    "References\n1. Other A. Cited work. doi:10.5555/cited.p2",
                    fontsize=9,
                )
                document.save(source)
            outcome = (await organization_run.run(
                organization_run.RunOptions(root), providers=FakeProviders(),
                pdf_runner=InlinePdfProcessRunner(),
            ))[0]
            self.assertNotIn(outcome.status, organization_run.RESOLVED_STATUSES)
            self.assertIsNone(outcome.confidence)
            self.assertEqual(outcome.pdf_title_hint, "")

    async def test_reference_noise_does_not_select_identity(self) -> None:
        with self.subTest(case="cited_arxiv_id_does_not_change_journal_naming_or_stage"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                source = root / "journal.pdf"
                doc = pymupdf.open()
                page = doc.new_page()
                page.insert_text((72, 72), TITLE)
                page.insert_text((72, 110), "Prior work: Example et al. arXiv:2509.16803v1")
                doc.set_metadata({"title": TITLE, "subject": f"doi:{DOI}"})
                doc.save(source)
                doc.close()
                outcome = (await organization_run.run(
                    organization_run.RunOptions(root), providers=FakeProviders(), pdf_runner=InlinePdfProcessRunner(),
                ))[0]
                row = audit_report.report_row(root, outcome)
                self.assertEqual(row["publication_stage"], "journal_article")
                self.assertNotIn("_arXiv", outcome.new_name)
                self.assertEqual(outcome.arxiv_doi, "")

        with self.subTest(case="reference_doi_falls_back_to_correct_title_record"):
            title = "Regulation of T cell activation by inhibitory receptors in mice"
            cited_title = "T cell exhaustion during chronic viral infection in mice"
            class Providers(FakeProviders):
                title_searches = 0
                async def records_by_doi(self, doi):
                    return metadata_providers.ProviderLookup((
                        record_matching.ProviderRecord("crossref", _paper_meta(doi, title=cited_title)),
                    ))
                async def records_by_title(self, value):
                    self.title_searches += 1
                    return metadata_providers.ProviderLookup((
                        record_matching.ProviderRecord("crossref", _paper_meta("10.5555/self.0001", title=title)),
                    ))
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                source = root / "download.pdf"
                doc = pymupdf.open()
                doc.new_page().insert_text((72, 72), title, fontsize=12)
                doc.new_page().insert_text((72, 72), "References doi:10.5555/cited.0001")
                doc.save(source)
                doc.close()
                providers = Providers()
                result = await organization_run.run(
                    organization_run.RunOptions(root), providers=providers,
                    pdf_runner=InlinePdfProcessRunner(),
                )
                self.assertEqual(result[0].doi, "10.5555/self.0001")
                self.assertEqual(providers.title_searches, 1)


    async def test_cold_and_warm_audit_keep_text_classification_and_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "collection"
            root.mkdir()
            source = root / "manuscript.pdf"
            doc = pymupdf.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Synthetic approaches to membrane protein folding kinetics", fontsize=14)
            page.insert_text((72, 120), "bioRxiv preprint. Not certified by peer review.")
            doc.save(source)
            doc.close()
            cache = pdf_identity_cache.IdentityCache(root, extractor_version=pdf_evidence.EVIDENCE_CACHE_VERSION, temp_root=Path(directory) / "local")
            with cache:
                rows = []
                for _ in range(2):
                    rows.append((await organization_run.run_audit(
                        organization_run.AuditOptions(root), providers=FakeProviders(),
                        pdf_runner=InlinePdfProcessRunner(), identity_cache=cache,
                    ))[0])
            self.assertEqual(rows[0], rows[1])
            self.assertEqual(rows[0]["failure_category"], "Preprints")
            self.assertEqual(rows[0]["replacement_eligible"], "yes")

    async def test_generated_name_collisions_preserve_every_work(self) -> None:
        with self.subTest(case="partial_run_collision_preserves_different_or_unknown_work"):
            for retry_failed in (False, True):
                for existing_doi in ("10.1000/different", None, DOI):
                    with self.subTest(retry_failed=retry_failed, existing_doi=existing_doi), tempfile.TemporaryDirectory() as directory:
                        root = Path(directory).resolve()
                        canonical = root / filename_policy.make_filename(_paper_meta())
                        if existing_doi:
                            _create_pdf(canonical, doi=existing_doi)
                        else:
                            _create_unresolved_pdf(canonical, title=TITLE)
                        _set_mtime(canonical, 1_000_000_000)
                        original = canonical.read_bytes()
                        incoming = root / "Failed" / "Others" / "incoming.pdf" if retry_failed else root / "incoming.pdf"
                        incoming.parent.mkdir(parents=True, exist_ok=True)
                        _create_pdf(incoming)
                        _set_mtime(incoming, 2_000_000_000)
                        outcomes = await organization_run.run(
                            organization_run.RunOptions(root, commit=True, retry_failed=retry_failed,
                                newer_than=None if retry_failed else 1.5),
                            providers=FakeProviders(), pdf_runner=InlinePdfProcessRunner(),
                        )
                        self.assertEqual(len(outcomes), 1)
                        if existing_doi == DOI:
                            self.assertTrue((root / "Duplicated").exists())
                            self.assertNotEqual(canonical.read_bytes(), original)
                        else:
                            self.assertEqual(canonical.read_bytes(), original)
                            self.assertEqual(outcomes[0].target_path.parent, root)
                            self.assertNotEqual(outcomes[0].target_path, canonical)
                            self.assertFalse((root / "Duplicated").exists())

        with self.subTest(case="commit_preserves_older_collision_as_duplicate"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                newer = root / "newer.pdf"
                older = root / "older.pdf"
                _create_pdf(newer)
                _create_pdf(older)
                os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
                os.utime(older, ns=(1_000_000_000, 1_000_000_000))

                outcomes = await organization_run.run(
                    organization_run.RunOptions(root, commit=True),
                    providers=FakeProviders(),
                )

                targets = [outcome.target_path for outcome in outcomes]
                self.assertEqual(
                    sum(
                        target is not None and target.parent == root
                        for target in targets
                    ),
                    1,
                )
                self.assertEqual(
                    sum(
                        target is not None
                        and target.parent == root / "Duplicated"
                        for target in targets
                    ),
                    1,
                )
                self.assertEqual(
                    sum(outcome.status.endswith("_duplicate") for outcome in outcomes),
                    1,
                )
                self.assertTrue((root / "Duplicated").is_dir())
                self.assertFalse((root / "Failed").exists())

        with self.subTest(case="different_dois_disambiguate_generated_name_collision"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                left = root / "left.pdf"
                right = root / "right.pdf"
                left.write_bytes(b"left")
                right.write_bytes(b"right")
                shared_name = "Example_MagnResonMed24_Imaging.pdf"
                outcomes = [
                    organization_run.OrganizationOutcome(
                        source=left,
                        status="crossref_resolved",
                        new_name=shared_name,
                        doi="10.1000/left",
                    ),
                    organization_run.OrganizationOutcome(
                        source=right,
                        status="crossref_resolved",
                        new_name=shared_name,
                        doi="10.1000/right",
                    ),
                ]

                committed = await organization_run._apply_collection_mutations(
                    root,
                    outcomes,
                )

                targets = [outcome.target_path for outcome in committed]
                self.assertTrue(all(target is not None for target in targets))
                self.assertEqual(len(set(targets)), 2)
                self.assertTrue(all(target.parent == root for target in targets))
                self.assertFalse(any(outcome.status.endswith("_duplicate") for outcome in committed))
                self.assertTrue(any("DOI-derived suffix" in outcome.note for outcome in committed))


    async def test_worker_infrastructure_failure_stays_in_place(self) -> None:
        class FailingRunner(InlinePdfProcessRunner):
            async def inspect(self, path, **kwargs):
                if path.name == "crashed.pdf":
                    raise BrokenProcessPool("fixture worker died")
                return await super().inspect(path, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            failed = root / "crashed.pdf"
            _create_pdf(failed)
            _create_pdf(root / "good.pdf")
            result = await organization_run.run(
                organization_run.RunOptions(root, commit=True), providers=FakeProviders(),
                pdf_runner=FailingRunner(),
            )
            self.assertTrue(failed.exists())
            self.assertIsNone(result[0].target_path)
            self.assertIn(result[1].status, organization_run.RESOLVED_STATUSES)
            self.assertTrue(result[1].target_path.exists())

    async def test_changed_source_between_phases_is_not_rewritten_or_moved(self) -> None:
        for covered in (False, True):
            with self.subTest(covered=covered), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                source = root / "paper.pdf"
                (_create_covered_pdf if covered else _create_pdf)(source)
                changed = []
                class ChangingAuditor:
                    async def audit_many(self, _identities):
                        incoming = root / "incoming.tmp"
                        doc = pymupdf.open()
                        for text in ("New content page one", "New content page two"):
                            doc.new_page().insert_text((72, 72), text)
                        doc.save(incoming)
                        doc.close()
                        incoming.replace(source)
                        changed.append(source.read_bytes())
                        return {}
                runner = InlinePdfProcessRunner()
                outcomes = await organization_run.run(
                    organization_run.RunOptions(root, commit=True),
                    providers=FakeProviders(), auditor=ChangingAuditor(), pdf_runner=runner,
                )
                self.assertEqual(outcomes[0].status, "pdf_source_changed")
                self.assertEqual(source.read_bytes(), changed[0])
                self.assertIsNone(outcomes[0].target_path)
                self.assertFalse((root / "Failed").exists())

    def test_resolved_arxiv_evidence_controls_publication_label(self) -> None:
        meta = _paper_meta()
        meta.journal_abbrev = None
        outcome = organization_run.OrganizationOutcome(
            source=Path("paper.pdf"),
            arxiv_doi="10.48550/arXiv.2509.16803",
        )
        resolved = record_matching.ResolvedRecord(
            status="crossref_title_resolved",
            meta=meta,
            pmid=None,
            confidence=0.99,
            source="title",
            reason="test record",
        )

        result = organization_run._resolved_outcome(outcome, resolved)

        self.assertIn("_arXiv24_", result.new_name)

    async def test_pii_resolution_uses_matching_authority_before_title_search(self) -> None:
        with self.subTest(case="pii_resolution_routes_through_match_authority"):
            class ExactPiiCrossref:
                @staticmethod
                async def records_by_alternative_id(
                    _alternative_id: str,
                ) -> metadata_providers.ProviderLookup:
                    return metadata_providers.ProviderLookup(
                        (
                            record_matching.ProviderRecord(
                                "crossref",
                                _paper_meta(),
                            ),
                        )
                    )

            class PiiProviders:
                crossref = ExactPiiCrossref()

            with mock.patch.object(
                record_matching,
                "decide_pii_match",
                wraps=record_matching.decide_pii_match,
            ) as matcher:
                decision = await organization_run._resolve_pii(
                    PiiProviders(),
                    "S1090780724000012",
                    "Cellular Metabolism and Immune Signaling in Liver Disease",
                )

            matcher.assert_called_once()
            self.assertEqual(decision.outcome, "rejected")
            self.assertIsNone(decision.resolved)

        with self.subTest(case="accepted_pii_lookup_precedes_and_skips_title_search"):
            pii = "S1352866100000697"

            class StaticRunner:
                @staticmethod
                async def inspect(
                    _pdf_path: Path,
                    *,
                    pdf_password: str = "",
                    max_text_pages: int = 5,
                ) -> pdf_evidence.PdfEvidence:
                    del pdf_password, max_text_pages
                    return pdf_evidence.PdfEvidence(
                        dois=[],
                        title=TITLE,
                        pii=pii,
                        arxiv_doi=None,
                    )

            class ExactPiiCrossref:
                calls: list[str] = []

                @classmethod
                async def records_by_alternative_id(
                    cls,
                    alternative_id: str,
                ) -> metadata_providers.ProviderLookup:
                    cls.calls.append(alternative_id)
                    return metadata_providers.ProviderLookup(
                        (
                            record_matching.ProviderRecord(
                                "crossref",
                                _paper_meta(),
                            ),
                        )
                    )

            class PiiProviders:
                pubmed = _PubMedMemory()
                crossref = ExactPiiCrossref()

                @staticmethod
                async def records_by_doi(
                    _doi: str,
                ) -> metadata_providers.ProviderLookup:
                    raise AssertionError("DOI lookup should not run")

                @staticmethod
                async def records_by_title(
                    _title: str,
                ) -> metadata_providers.ProviderLookup:
                    raise AssertionError("title lookup should not run")

            analysis = await organization_run._analyze_pdf(
                Path("paper.pdf"),
                "",
                PiiProviders(),
                StaticRunner(),
            )

            self.assertEqual(ExactPiiCrossref.calls, [pii])
            self.assertEqual(analysis.outcome.status, "crossref_resolved")
            self.assertEqual(analysis.outcome.match_source, "filename-pii")


    async def test_runner_injection_and_owned_resources(self) -> None:
        with self.subTest(case="run_accepts_pdf_process_runner"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                source = root / "paper.pdf"
                _create_pdf(source)
                runner = InlinePdfProcessRunner()

                outcomes = await organization_run.run(
                    organization_run.RunOptions(root),
                    providers=FakeProviders(),
                    pdf_runner=runner,
                )

                self.assertEqual(runner.inspected, [source])
                self.assertEqual(len(outcomes), 1)

        with self.subTest(case="run_owned_cache_and_client_are_closed"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                caches: list[metadata_providers.MetadataCache] = []
                clients: list[object] = []

                class TrackingCache(metadata_providers.MetadataCache):
                    def __init__(self, path: Path):
                        super().__init__(path)
                        caches.append(self)

                class TrackingClient:
                    def __init__(self, **kwargs):
                        self.closed = False
                        self.options = kwargs
                        clients.append(self)

                    async def __aenter__(self):
                        return self

                    async def __aexit__(self, *_args):
                        self.closed = True

                with (
                    mock.patch.object(
                        organization_run.metadata_providers,
                        "MetadataCache",
                        TrackingCache,
                    ),
                    mock.patch.object(
                        organization_run.httpx,
                        "AsyncClient",
                        TrackingClient,
                    ),
                ):
                    outcomes = await organization_run.run(
                        organization_run.RunOptions(root)
                    )

                self.assertEqual(outcomes, [])
                self.assertEqual(
                    Path(caches[0].path),
                    root / ".pdf_renamer_cache.sqlite",
                )
                self.assertIsNone(caches[0].con)
                self.assertTrue(clients[0].closed)
                self.assertIs(clients[0].options["follow_redirects"], True)


    async def test_audit_runs_only_after_identity_acceptance(self) -> None:
        with self.subTest(case="resolved_outcome_receives_neutral_audit_fields"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                source = root / "paper.pdf"
                _create_pdf(source)
                audit = publication_audit.PublicationAudit(
                    retraction=publication_status.RetractionAssessment(
                        "reported_retracted",
                        (
                            publication_status.RetractionSignal(
                                "pubmed",
                                "retraction",
                                "PMID:200",
                            ),
                        ),
                        "formal notice",
                    ),
                    commentary=publication_status.CommentaryAssessment(
                        status="critical_citation_found",
                        scite=publication_status.SciteTallies(
                            total=8,
                            contradicting=2,
                        ),
                        note="neutral commentary signal",
                    ),
                    checked_at="2026-09-10T12:00:00Z",
                )
                auditor = FakeAuditor(audit)

                outcomes = await organization_run.run(
                    organization_run.RunOptions(root),
                    providers=FakeProviders(),
                    auditor=auditor,
                )

                self.assertIn(outcomes[0].status, organization_run.RESOLVED_STATUSES)
                self.assertEqual(outcomes[0].retraction_status, "reported_retracted")
                self.assertEqual(
                    outcomes[0].commentary_status,
                    "critical_citation_found",
                )
                self.assertEqual(outcomes[0].scite_contradicting, 2)
                self.assertEqual(auditor.calls, [(DOI, "12345", None)])

        with self.subTest(case="unresolved_identity_is_not_sent_to_audit_sources"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                source = root / "paper.pdf"
                _create_unresolved_pdf(source, title="An unmatched scientific report")
                auditor = FakeAuditor(
                    publication_audit.incomplete_audit(reason="unused")
                )

                outcomes = await organization_run.run(
                    organization_run.RunOptions(root),
                    providers=FakeProviders(),
                    auditor=auditor,
                )

                self.assertEqual(auditor.calls, [])
                self.assertEqual(outcomes[0].retraction_status, "identity_unresolved")
                self.assertEqual(outcomes[0].commentary_status, "identity_unresolved")

        with self.subTest(case="audit_failure_is_reported_without_unresolving_identity"):
            class RaisingAuditor:
                @staticmethod
                async def audit_many(_identities):
                    raise RuntimeError("audit source unavailable")

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                _create_pdf(root / "paper.pdf")

                outcomes = await organization_run.run(
                    organization_run.RunOptions(root),
                    providers=FakeProviders(),
                    auditor=RaisingAuditor(),
                )

                self.assertIn(outcomes[0].status, organization_run.RESOLVED_STATUSES)
                self.assertEqual(outcomes[0].retraction_status, "check_incomplete")
                self.assertIn("audit source unavailable", outcomes[0].retraction_note)


    async def test_retry_incomplete_audits_updates_only_unfinished_side(self) -> None:
        audit = publication_audit.PublicationAudit(
            retraction=publication_status.RetractionAssessment(
                "reported_retracted"
            ),
            commentary=publication_status.CommentaryAssessment(
                status="critical_citation_found",
                scite=publication_status.SciteTallies(
                    total=4,
                    contradicting=1,
                ),
                note="Scite citation statements refreshed",
            ),
            checked_at="2026-09-14T12:00:00Z",
        )
        auditor = FakeAuditor(audit)

        class Scite:
            async def fetch_many(self, dois):
                return {
                    doi: publication_audit.SciteLookup(
                        payload={"total": 4, "contradicting": 1},
                        checked=True,
                    )
                    for doi in dois
                }

        rows: list[dict[str, object]] = [
            {
                "doi": DOI,
                "pmid": "12345",
                "arxiv_doi": "",
                "retraction_status": "no_retraction_signal",
                "commentary_status": "linked_comment_found",
                "audit_evidence": (
                    '{"pubmed_comments":"CommentIn:PMID:300"}'
                ),
                "audit_note": (
                    '{"commentary":"Scite tally was unavailable; '
                    'PubMed comment link found"}'
                ),
                "audit_checked_at": "2026-09-10T12:00:00Z",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            updated, selected = await organization_run.retry_incomplete_audits(
                organization_run.RunOptions(Path(directory)),
                rows,
                auditor=auditor,
                scite=Scite(),
            )

        self.assertEqual(selected, 1)
        self.assertEqual(updated[0]["retraction_status"], "no_retraction_signal")
        self.assertEqual(
            updated[0]["commentary_status"],
            "critical_citation_found",
        )
        self.assertEqual(
            json.loads(str(updated[0]["audit_evidence"]))["scite"]["total"],
            4,
        )
        self.assertEqual(
            json.loads(str(updated[0]["audit_evidence"]))["pubmed_comments"],
            "CommentIn:PMID:300",
        )
        self.assertEqual(auditor.calls, [])

    async def test_retry_failed_case_alias_keeps_existing_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parent = root / "Failed" / "others"
            parent.mkdir(parents=True)
            source = parent / "scan.pdf"
            _create_unresolved_pdf(source, title="Unidentified synthetic document")
            target = root / "Failed" / "Others" / source.name
            if not target.exists():
                self.skipTest("requires a case-insensitive temporary volume")
            original = source.read_bytes()
            outcomes = await organization_run.run(
                organization_run.RunOptions(root, retry_failed=True, commit=True),
                providers=FakeProviders(), pdf_runner=InlinePdfProcessRunner(),
            )
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(outcomes[0].target_path.read_bytes(), original)

    async def test_failed_pdf_is_classified_placed_and_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "ISMRM2019_0434.pdf"
            _create_unresolved_pdf(
                source,
                title="An unresolved conference abstract",
            )

            dry_run = await organization_run.run(
                organization_run.RunOptions(root),
                providers=FakeProviders(),
            )

            self.assertEqual(dry_run[0].failure_category, "Proceedings")
            self.assertIn("conference", dry_run[0].classification_note.lower())
            self.assertIsNone(dry_run[0].target_path)
            self.assertTrue(source.exists())
            self.assertFalse((root / "Failed").exists())

            committed = await organization_run.run(
                organization_run.RunOptions(root, commit=True),
                providers=FakeProviders(),
            )

            target = root / "Failed" / "Proceedings" / source.name
            self.assertEqual(committed[0].target_path, target)
            self.assertTrue(target.exists())
            self.assertFalse((root / "Duplicated").exists())

            retried = await organization_run.run(
                organization_run.RunOptions(root, retry_failed=True),
                providers=FakeProviders(),
            )

            self.assertEqual([item.source for item in retried], [target])
            self.assertEqual(retried[0].failure_category, "Proceedings")

        for condition, expected_status in (
            ("password_protected", "pdf_password_required"),
            ("small_unreadable", "pdf_unreadable"),
            ("large_unreadable", "pdf_unreadable"),
            ("unrepairable", "pdf_unrepairable"),
            ("signed_rewrite_blocked", "pdf_signed_rewrite_blocked"),
        ):
            with self.subTest(case=condition), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                source = root / f"{condition}.pdf"
                runner = InlinePdfProcessRunner()
                providers = FakeProviders()
                with contextlib.ExitStack() as patches:
                    if condition == "password_protected":
                        with pymupdf.open() as document:
                            document.new_page().insert_text((72, 72), "Locked synthetic paper")
                            document.save(
                                source, encryption=pymupdf.PDF_ENCRYPT_AES_256,
                                user_pw="fixture-password", owner_pw="fixture-owner",
                            )
                    elif condition.endswith("unreadable"):
                        source.write_bytes(b"not a PDF\n" * (
                            400_000 if condition == "large_unreadable" else 100
                        ))
                        self.assertEqual(
                            pdf_rewrite.exceeds_lossless_compression_threshold(source),
                            condition == "large_unreadable",
                        )
                    else:
                        _create_covered_pdf(source)
                        if condition == "unrepairable":
                            patches.enter_context(mock.patch.object(
                                pdf_rewrite, "write_validated_copy",
                                side_effect=ValueError("injected normalization failure"),
                            ))
                        else:
                            patches.enter_context(mock.patch.object(
                                pdf_rewrite, "pdf_has_signatures", return_value=True,
                            ))
                    original = source.read_bytes()
                    dry_run = (await organization_run.run(
                        organization_run.RunOptions(root), providers=providers,
                        pdf_runner=runner,
                    ))[0]
                    self.assertEqual(dry_run.pdf_status, expected_status)
                    self.assertIsNone(dry_run.target_path)
                    self.assertEqual(source.read_bytes(), original)
                    self.assertFalse((root / "Failed").exists())
                    committed = (await organization_run.run(
                        organization_run.RunOptions(root, commit=True), providers=providers,
                        pdf_runner=runner,
                    ))[0]

                target = root / "Failed" / "Others" / source.name
                self.assertEqual(committed.status, expected_status)
                self.assertEqual(committed.pdf_status, expected_status)
                self.assertFalse(committed.placement_blocked)
                self.assertEqual(committed.target_path, target)
                self.assertEqual(target.read_bytes(), original)
                self.assertFalse(source.exists())
                self.assertEqual(runner.rewritten, [])
                self.assertEqual(providers.calls, [])

    async def test_dry_run_discovery_and_read_only_contract(self) -> None:
        with self.subTest(case="dry_run_is_read_only_and_returns_stable_path_order"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                second = root / "B.pdf"
                first = root / "a.pdf"
                _create_pdf(second, doi="10.1000/second")
                _create_pdf(first, doi="10.1000/first")
                original = {
                    path: hashlib.sha256(path.read_bytes()).digest()
                    for path in (first, second)
                }
                progress: list[str] = []

                outcomes = await organization_run.run(
                    organization_run.RunOptions(root),
                    providers=FakeProviders(),
                    progress=lambda _done, _total, outcome: progress.append(
                        outcome.source.name
                    ),
                )

                self.assertEqual(
                    [outcome.source.name for outcome in outcomes],
                    ["a.pdf", "B.pdf"],
                )
                self.assertCountEqual(progress, ["a.pdf", "B.pdf"])
                for path, digest in original.items():
                    self.assertEqual(
                        hashlib.sha256(path.read_bytes()).digest(),
                        digest,
                    )
                self.assertFalse((root / "Duplicated").exists())
                self.assertFalse((root / "Failed").exists())

        with self.subTest(case="holding_directories_are_not_selected"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                _create_pdf(root / "paper.pdf")
                superseded = root / "Superseded"
                superseded.mkdir()
                _create_pdf(superseded / "archived.pdf")
                _create_pdf(root / ".paper.pdf.repairing.pdf")

                outcomes = await organization_run.run(
                    organization_run.RunOptions(root),
                    providers=FakeProviders(),
                )

                self.assertEqual(
                    [outcome.source.name for outcome in outcomes],
                    ["paper.pdf"],
                )


    async def test_commit_does_not_create_unused_holding_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "paper.pdf"
            _create_pdf(source)

            outcomes = await organization_run.run(
                organization_run.RunOptions(root, commit=True),
                providers=FakeProviders(),
            )

            self.assertEqual(len(outcomes), 1)
            self.assertIsNotNone(outcomes[0].target_path)
            self.assertTrue(outcomes[0].target_path.is_file())
            self.assertFalse((root / "Duplicated").exists())
            self.assertFalse((root / "Failed").exists())

    async def test_large_pdf_compression_modes(self) -> None:
        with self.subTest(case="dry_run_reports_large_pdf_compression_without_mutation"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                source = root / "paper.pdf"
                _create_pdf(source)
                _make_pdf_large_and_compressible(source)
                original = source.read_bytes()
                runner = InlinePdfProcessRunner()

                outcomes = await organization_run.run(
                    organization_run.RunOptions(root),
                    providers=FakeProviders(),
                    pdf_runner=runner,
                )

                self.assertEqual(outcomes[0].pdf_status, "pdf_compression_needed")
                self.assertIn("3 MB", outcomes[0].pdf_note)
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(runner.rewritten, [])

        with self.subTest(case="commit_compresses_large_pdf_before_placement"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                source = root / "paper.pdf"
                _create_pdf(source)
                _make_pdf_large_and_compressible(source)
                original_size = source.stat().st_size
                runner = InlinePdfProcessRunner()

                outcomes = await organization_run.run(
                    organization_run.RunOptions(root, commit=True),
                    providers=FakeProviders(),
                    pdf_runner=runner,
                )

                target = outcomes[0].target_path
                self.assertIsNotNone(target)
                self.assertEqual(outcomes[0].pdf_status, "pdf_compressed")
                self.assertLess(target.stat().st_size, original_size)
                self.assertEqual(runner.rewritten, [source])


    async def test_analysis_finishes_before_required_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "encrypted.pdf"
            _create_pdf(source, encrypted=True)
            original_digest = hashlib.sha256(source.read_bytes()).digest()
            observed_digests: list[bytes] = []

            def observe_source(_doi: str) -> None:
                observed_digests.append(
                    hashlib.sha256(source.read_bytes()).digest()
                )

            outcomes = await organization_run.run(
                organization_run.RunOptions(root, commit=True),
                providers=FakeProviders(on_doi=observe_source),
            )

            self.assertEqual(observed_digests, [original_digest])
            target = outcomes[0].target_path
            self.assertIsNotNone(target)
            self.assertNotEqual(
                hashlib.sha256(target.read_bytes()).digest(),
                original_digest,
            )
            self.assertEqual(outcomes[0].pdf_status, "pdf_decrypted")

        with self.subTest(case="repair_needed_still_rewrites"), tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "damaged.pdf"
            _create_pdf(source)
            source.write_bytes(re.sub(
                rb"startxref\s+\d+", b"startxref\n0", source.read_bytes(),
            ))
            runner = InlinePdfProcessRunner()
            dry_run = (await organization_run.run(
                organization_run.RunOptions(root), providers=FakeProviders(), pdf_runner=runner,
            ))[0]
            self.assertEqual(dry_run.pdf_status, "pdf_repair_needed")
            self.assertEqual(runner.rewritten, [])

            committed = (await organization_run.run(
                organization_run.RunOptions(root, commit=True), providers=FakeProviders(),
                pdf_runner=runner,
            ))[0]
            self.assertIn(committed.status, organization_run.RESOLVED_STATUSES)
            self.assertEqual(committed.pdf_status, "pdf_repaired")
            self.assertEqual(runner.rewritten, [source])
            self.assertIsNotNone(committed.target_path)
            with pymupdf.open(committed.target_path) as document:
                self.assertFalse(document.is_repaired)
                for page in document:
                    self.assertIn(TITLE, page.get_text())

    async def test_commit_removes_detected_cover_before_placement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "covered.pdf"
            _create_covered_pdf(source)

            outcomes = await organization_run.run(
                organization_run.RunOptions(root, commit=True),
                providers=FakeProviders(),
            )

            target = outcomes[0].target_path
            self.assertIsNotNone(target)
            self.assertEqual(
                outcomes[0].pdf_status,
                "pdf_cover_page_removed",
            )
            document = pymupdf.open(target)
            try:
                self.assertEqual(len(document), 1)
                self.assertIn(TITLE, document[0].get_text())
                self.assertNotIn(
                    "researchgate.net",
                    document[0].get_text().lower(),
                )
            finally:
                document.close()

    async def test_batch_analysis_failure_boundaries(self) -> None:
        with self.subTest(case="analysis_failure_prevents_phase_two_mutation"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                source = root / "encrypted.pdf"
                _create_pdf(source, encrypted=True)
                original_digest = hashlib.sha256(source.read_bytes()).digest()

                def interrupt_progress(
                    _completed: int,
                    _total: int,
                    _outcome: organization_run.OrganizationOutcome,
                ) -> None:
                    raise RuntimeError("injected analysis interruption")

                with self.assertRaisesRegex(
                    RuntimeError,
                    "injected analysis interruption",
                ):
                    await organization_run.run(
                        organization_run.RunOptions(root, commit=True),
                        providers=FakeProviders(),
                        progress=interrupt_progress,
                    )

                self.assertEqual(
                    hashlib.sha256(source.read_bytes()).digest(),
                    original_digest,
                )
                self.assertFalse((root / "Duplicated").exists())
                self.assertFalse((root / "Failed").exists())

        with self.subTest(case="one_unreadable_pdf_does_not_stop_other_outcomes"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                unreadable = root / "a-broken.pdf"
                valid = root / "b-valid.pdf"
                unreadable.write_bytes(b"not a PDF")
                _create_pdf(valid)

                outcomes = await organization_run.run(
                    organization_run.RunOptions(root),
                    providers=FakeProviders(),
                )

                self.assertEqual(len(outcomes), 2)
                self.assertEqual(outcomes[0].source, unreadable)
                self.assertEqual(outcomes[0].status, "pdf_unreadable")
                self.assertIn(
                    outcomes[1].status,
                    organization_run.RESOLVED_STATUSES,
                )


    async def test_provider_conflict_is_reported_through_run_interface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "paper.pdf"
            _create_pdf(source)

            def conflicting(_doi: str):
                return (
                    record_matching.ProviderRecord(
                        "pubmed",
                        _paper_meta(),
                        "12345",
                    ),
                    record_matching.ProviderRecord(
                        "crossref",
                        _paper_meta(
                            title=(
                                "Unrelated Findings from a Different "
                                "Scientific Study"
                            ),
                            family="Different",
                        ),
                    ),
                )

            outcomes = await organization_run.run(
                organization_run.RunOptions(root),
                providers=FakeProviders(conflicting),
            )

            self.assertEqual(
                outcomes[0].status,
                "conflicting_provider_records",
            )
            self.assertIn("conflict", outcomes[0].note.lower())

    async def test_cancellation_waits_for_isolated_transaction(self) -> None:
        started = threading.Event()
        release = threading.Event()
        completed = threading.Event()

        def transaction() -> str:
            started.set()
            release.wait(timeout=5)
            completed.set()
            return "done"

        task = asyncio.create_task(
            organization_run._complete_isolated_transaction(transaction)
        )
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        release.set()

        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(completed.is_set())

    async def test_cancellation_waits_for_async_rewrite_transaction(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        completed = asyncio.Event()

        async def transaction() -> str:
            started.set()
            await release.wait()
            completed.set()
            return "done"

        task = asyncio.create_task(
            organization_run._complete_isolated_async_transaction(transaction)
        )
        await started.wait()
        task.cancel()
        release.set()

        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(completed.is_set())

class CompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_driver_returns_legacy_dictionaries(self) -> None:
        outcome = organization_run.OrganizationOutcome(
            source=Path("/tmp/paper.pdf"),
            status="crossref_resolved",
            new_name="Example_MagnResonMed24_Imaging.pdf",
            confidence=0.95,
        )

        with mock.patch.object(
            organization_run,
            "run",
            mock.AsyncMock(return_value=[outcome]),
        ):
            results = await RenamePDF.driver(Path("/tmp"))

        self.assertEqual(results[0]["file"], str(Path("/tmp/paper.pdf")))
        self.assertEqual(results[0]["confidence"], "0.95")
        self.assertEqual(
            tuple(results[0]),
            organization_run.CSV_FIELDS,
        )
        self.assertIn("audit_evidence", results[0])
        self.assertIn("audit_note", results[0])
        self.assertNotIn("retraction_evidence", results[0])
        self.assertNotIn("scite_total", results[0])

    def test_empty_and_retry_report_handling(self) -> None:
        with self.subTest(case="main_does_not_claim_to_write_an_empty_report"):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                report = root / "0_results.csv"
                report.write_text("existing report\n", encoding="utf-8")
                output = io.StringIO()

                with (
                    mock.patch.object(
                        RenamePDF,
                        "driver",
                        mock.AsyncMock(return_value=[]),
                    ),
                    mock.patch.object(
                        RenamePDF,
                        "_open_identity_cache",
                        return_value=None,
                    ),
                    contextlib.redirect_stdout(output),
                ):
                    exit_code = RenamePDF.main([str(root)])

                self.assertEqual(exit_code, 0)
                self.assertEqual(
                    report.read_text(encoding="utf-8"),
                    "existing report\n",
                )
                self.assertIn("No PDFs selected", output.getvalue())
                self.assertNotIn("CSV saved", output.getvalue())

        with self.subTest(case="main_retries_only_incomplete_report_audits"):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                report = root / "0_results.csv"
                row = dict.fromkeys(organization_run.CSV_FIELDS, "")
                row.update(
                    {
                        "doi": DOI,
                        "retraction_status": "no_retraction_signal",
                        "commentary_status": "check_incomplete",
                    }
                )
                report_io.write_csv_atomic(
                    report,
                    [row],
                    fieldnames=organization_run.CSV_FIELDS,
                )
                updated = dict(row)
                updated["commentary_status"] = "no_commentary_signal"
                output = io.StringIO()

                with (
                    mock.patch.object(
                        organization_run,
                        "retry_incomplete_audits",
                        mock.AsyncMock(return_value=([updated], 1)),
                    ) as retry,
                    mock.patch.object(
                        pdf_rewrite,
                        "recover_interrupted_rewrites",
                    ) as recovery,
                    contextlib.redirect_stdout(output),
                ):
                    exit_code = RenamePDF.main(
                        [str(root), "--retry-audit-incomplete"]
                    )

                self.assertEqual(exit_code, 0)
                retry.assert_awaited_once()
                recovery.assert_not_called()
                _, written = report_io.read_csv(report)
                self.assertEqual(
                    written[0]["commentary_status"],
                    "no_commentary_signal",
                )
                self.assertIn("Retried 1 incomplete audit row", output.getvalue())


    def test_recovery_precedes_compatibility_driver(self) -> None:
        with self.subTest(case="main_halts_on_uncommitted_recovery"):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                rollback = root / ".paper.pdf.renamepdf-rollback"
                rollback.write_bytes(b"original PDF")
                output = io.StringIO()

                with (
                    mock.patch.object(
                        RenamePDF,
                        "driver",
                        mock.AsyncMock(return_value=[]),
                    ) as driver,
                    mock.patch.object(
                        RenamePDF,
                        "_open_identity_cache",
                        return_value=None,
                    ),
                    contextlib.redirect_stdout(output),
                ):
                    exit_code = RenamePDF.main([str(root)])

                self.assertEqual(exit_code, 2)
                driver.assert_not_awaited()
                self.assertIn("pdf_recovery_needed", output.getvalue())
                self.assertIn(str(rollback.resolve()), output.getvalue())

        with self.subTest(case="main_commit_restores_rollback_before_driver"):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                canonical = root / "paper.pdf"
                rollback = root / ".paper.pdf.renamepdf-rollback"
                rollback.write_bytes(b"original PDF")
                output = io.StringIO()

                with (
                    mock.patch.object(
                        RenamePDF,
                        "driver",
                        mock.AsyncMock(return_value=[]),
                    ) as driver,
                    mock.patch.object(
                        RenamePDF,
                        "_open_identity_cache",
                        return_value=None,
                    ),
                    contextlib.redirect_stdout(output),
                ):
                    exit_code = RenamePDF.main([str(root), "--commit"])

                self.assertEqual(exit_code, 0)
                driver.assert_awaited_once()
                self.assertEqual(canonical.read_bytes(), b"original PDF")
                self.assertIn("pdf_recovery_restored", output.getvalue())


if __name__ == "__main__":
    unittest.main()
