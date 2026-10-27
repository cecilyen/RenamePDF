"""Spawn-isolated PyMuPDF inspection and rewrite operations."""

from __future__ import annotations

import asyncio
import math
import multiprocessing
import os
import signal
import time
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing.pool import ApplyResult, Pool
from pathlib import Path
from types import TracebackType
from typing import Protocol

import pdf_evidence
import pdf_rewrite
import file_metadata


DEFAULT_MAX_WORKERS = min(4, max(1, os.cpu_count() or 1))
DEFAULT_MAX_TASKS_PER_CHILD = 64
DEFAULT_INSPECTION_TIMEOUT = 120.0


def inspection_timeout_seconds(value: float | str) -> float:
    """Parse and validate a finite, positive inspection deadline."""
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("inspection timeout must be finite and positive") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("inspection timeout must be finite and positive")
    return seconds


@dataclass(frozen=True, slots=True)
class PdfBatchInspectionOutcome:
    """One read-only inspection result returned by a spawned worker."""

    path: Path
    evidence: pdf_evidence.PdfEvidence | None
    error_status: str = ""
    error_note: str = ""
    worker_pid: int | None = None


@dataclass(frozen=True, slots=True)
class _BatchWorkerResponse:
    path: str
    evidence: pdf_evidence.PdfEvidence
    worker_pid: int


@dataclass(slots=True)
class _PendingBatchInspection:
    path: Path
    result: ApplyResult[_BatchWorkerResponse]
    deadline: float


class PdfProcessRunner(Protocol):
    async def inspect(
        self,
        pdf_path: Path,
        *,
        pdf_password: str = "",
        max_text_pages: int = 5,
    ) -> pdf_evidence.PdfEvidence: ...

    async def rewrite(
        self,
        pdf_path: Path,
        *,
        pdf_password: str = "",
        remove_page_indices: tuple[int, ...] = (),
        compress_if_large: bool = False,
        expected_source_token: file_metadata.FileToken | None = None,
    ) -> pdf_rewrite.PdfRewriteOutcome: ...


def _inspect_pdf_worker(
    pdf_path: str,
    pdf_password: str,
    max_text_pages: int,
) -> pdf_evidence.PdfEvidence:
    return pdf_evidence.inspect_pdf(
        Path(pdf_path),
        pdf_password=pdf_password,
        max_text_pages=max_text_pages,
    )


def _inspect_pdf_batch_worker(
    pdf_path: str,
    pdf_password: str,
    max_text_pages: int,
    include_text_samples: bool = True,
) -> _BatchWorkerResponse:
    """Open one path inside its worker; never receive a Document object."""
    evidence = pdf_evidence.inspect_pdf(
        Path(pdf_path),
        pdf_password=pdf_password,
        max_text_pages=max_text_pages,
    )
    if not include_text_samples:
        evidence = pdf_evidence.without_text_samples(evidence)
    return _BatchWorkerResponse(
        path=pdf_path,
        evidence=evidence,
        worker_pid=os.getpid(),
    )


def _new_pdf_inspection_pool(
    *,
    max_workers: int,
    max_tasks_per_child: int,
) -> Pool:
    return multiprocessing.get_context("spawn").Pool(
        processes=max_workers,
        maxtasksperchild=max_tasks_per_child,
    )


def _terminate_pool(pool: Pool | None) -> None:
    if pool is None:
        return
    pool.terminate()
    pool.join()


def inspect_pdf_batch(
    paths: Iterable[Path],
    *,
    pdf_password: str = "",
    max_text_pages: int = 5,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout: float = DEFAULT_INSPECTION_TIMEOUT,
    max_tasks_per_child: int = DEFAULT_MAX_TASKS_PER_CHILD,
    include_text_samples: bool = True,
    progress: Callable[
        [int, int, PdfBatchInspectionOutcome],
        None,
    ] | None = None,
) -> list[PdfBatchInspectionOutcome]:
    """Inspect paths in a reusable spawned pool with per-task deadlines.

    At most ``max_workers`` tasks are submitted at once, so each deadline starts
    when a worker is available rather than while a request waits in a long
    executor queue. A timeout terminates and recreates the pool: the timed-out
    PDF becomes an error while other interrupted read-only tasks are retried.
    """
    ordered_paths = list(paths)
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if max_tasks_per_child < 1:
        raise ValueError("max_tasks_per_child must be at least 1")
    if not ordered_paths:
        return []

    worker_count = min(max_workers, len(ordered_paths))
    pending = deque(enumerate(ordered_paths))
    in_flight: dict[int, _PendingBatchInspection] = {}
    outcomes: list[PdfBatchInspectionOutcome | None] = [
        None
    ] * len(ordered_paths)
    completed = 0
    pool: Pool | None = None

    def finish(index: int, outcome: PdfBatchInspectionOutcome) -> None:
        nonlocal completed
        outcomes[index] = outcome
        completed += 1
        if progress is not None:
            progress(completed, len(ordered_paths), outcome)

    try:
        while pending or in_flight:
            if pool is None:
                pool = _new_pdf_inspection_pool(
                    max_workers=worker_count,
                    max_tasks_per_child=max_tasks_per_child,
                )
            while pending and len(in_flight) < worker_count:
                index, path = pending.popleft()
                result = pool.apply_async(
                    _inspect_pdf_batch_worker,
                    (str(path), pdf_password, max_text_pages, include_text_samples),
                )
                in_flight[index] = _PendingBatchInspection(
                    path=path,
                    result=result,
                    deadline=time.monotonic() + timeout,
                )

            ready_indices = [
                index
                for index, task in in_flight.items()
                if task.result.ready()
            ]
            for index in ready_indices:
                task = in_flight.pop(index)
                try:
                    response = task.result.get()
                    if Path(response.path) != task.path:
                        raise ValueError(
                            "inspection worker path mismatch: "
                            f"expected {task.path}, received {response.path}"
                        )
                    outcome = PdfBatchInspectionOutcome(
                        path=task.path,
                        evidence=response.evidence,
                        worker_pid=response.worker_pid,
                    )
                except Exception as exc:
                    outcome = PdfBatchInspectionOutcome(
                        path=task.path,
                        evidence=None,
                        error_status="pdf_processing_error",
                        error_note=(
                            "Inspection worker failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                finish(index, outcome)
            if ready_indices:
                continue

            now = time.monotonic()
            expired_indices = [
                index
                for index, task in in_flight.items()
                if task.deadline <= now
            ]
            if expired_indices:
                retry = [
                    (index, task.path)
                    for index, task in in_flight.items()
                    if index not in expired_indices
                ]
                retry.extend(pending)
                for index in expired_indices:
                    task = in_flight[index]
                    finish(
                        index,
                        PdfBatchInspectionOutcome(
                            path=task.path,
                            evidence=None,
                            error_status="pdf_processing_timeout",
                            error_note=(
                                f"Inspection exceeded {timeout:g} seconds"
                            ),
                        ),
                    )
                _terminate_pool(pool)
                pool = None
                in_flight.clear()
                pending = deque(sorted(retry, key=lambda item: item[0]))
                continue

            nearest_deadline = min(
                task.deadline for task in in_flight.values()
            )
            time.sleep(max(0.001, min(0.02, nearest_deadline - now)))

        pool.close()
        pool.join()
        pool = None
    except BaseException:
        _terminate_pool(pool)
        raise

    return [
        outcome
        for outcome in outcomes
        if outcome is not None
    ]


def _rewrite_pdf_worker(
    pdf_path: str,
    pdf_password: str,
    remove_page_indices: tuple[int, ...],
    compress_if_large: bool,
    expected_source_token: file_metadata.FileToken | None,
) -> pdf_rewrite.PdfRewriteOutcome:
    return pdf_rewrite.rewrite_pdf(
        Path(pdf_path),
        pdf_password=pdf_password,
        remove_page_indices=remove_page_indices,
        compress_if_large=compress_if_large,
        expected_source_token=expected_source_token,
    )


def _inspection_failure(status: str, note: str) -> pdf_evidence.PdfEvidence:
    return pdf_evidence.PdfEvidence(
        dois=[], title=None, pii=None, arxiv_doi=None,
        pdf_status=status, pdf_note=note,
    )


def _initialize_worker() -> None:
    # Terminal SIGINT belongs to the parent. It cancels inspections and waits
    # for an active rewrite transaction before shutting down the workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _consume_future_exception(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


def _stop_inspection_executor(executor: ProcessPoolExecutor) -> None:
    """Kill only this runner's read-only worker, never other child processes."""
    executor.kill_workers()


class SpawnPdfProcessRunner:
    """Reuse isolated single-worker lanes; inspection failures affect one lane."""

    def __init__(
        self, *, max_workers: int = DEFAULT_MAX_WORKERS,
        inspection_timeout: float = DEFAULT_INSPECTION_TIMEOUT,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._timeout = inspection_timeout_seconds(inspection_timeout)
        self._closed = False
        self._executors: list[ProcessPoolExecutor | None] = [None] * max_workers
        self._available: asyncio.Queue[int] = asyncio.Queue()
        for index in range(max_workers):
            self._available.put_nowait(index)

    def _active_executor(self, index: int) -> ProcessPoolExecutor:
        if self._closed:
            raise RuntimeError("PDF process runner is closed")
        executor = self._executors[index]
        if executor is None:
            executor = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                max_tasks_per_child=DEFAULT_MAX_TASKS_PER_CHILD,
                initializer=_initialize_worker,
            )
            self._executors[index] = executor
        return executor

    async def _discard_executor(self, index: int) -> None:
        executor = self._executors[index]
        self._executors[index] = None
        if executor is not None:
            await asyncio.to_thread(_stop_inspection_executor, executor)

    async def inspect(
        self,
        pdf_path: Path,
        *,
        pdf_password: str = "",
        max_text_pages: int = 5,
    ) -> pdf_evidence.PdfEvidence:
        loop = asyncio.get_running_loop()
        if self._closed:
            raise RuntimeError("PDF process runner is closed")
        index = await self._available.get()
        try:
            for attempt in range(2):
                try:
                    future = loop.run_in_executor(
                        self._active_executor(index), _inspect_pdf_worker,
                        str(pdf_path), pdf_password, max_text_pages,
                    )
                    # A timed-out or cancelled task is killed below; retrieve its
                    # eventual exception so it does not become an unhandled future.
                    future.add_done_callback(_consume_future_exception)
                    return await asyncio.wait_for(asyncio.shield(future), self._timeout)
                except TimeoutError:
                    await self._discard_executor(index)
                    return _inspection_failure(
                        "pdf_processing_timeout", f"Inspection exceeded {self._timeout:g} seconds",
                    )
                except BrokenProcessPool as exc:
                    await self._discard_executor(index)
                    if attempt:
                        return _inspection_failure("pdf_processing_error", f"Inspection worker failed twice: {exc}")
                except asyncio.CancelledError:
                    await self._discard_executor(index)
                    raise
        finally:
            self._available.put_nowait(index)
        raise AssertionError("inspection retry loop exhausted")

    async def rewrite(
        self,
        pdf_path: Path,
        *,
        pdf_password: str = "",
        remove_page_indices: tuple[int, ...] = (),
        compress_if_large: bool = False,
        expected_source_token: file_metadata.FileToken | None = None,
    ) -> pdf_rewrite.PdfRewriteOutcome:
        loop = asyncio.get_running_loop()
        if self._closed:
            raise RuntimeError("PDF process runner is closed")
        index = await self._available.get()
        try:
            future = loop.run_in_executor(
                self._active_executor(index), _rewrite_pdf_worker,
                str(pdf_path), pdf_password, remove_page_indices,
                compress_if_large, expected_source_token,
            )
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                await asyncio.shield(future)
                raise
        except BrokenProcessPool:
            await self._discard_executor(index)
            raise
        finally:
            self._available.put_nowait(index)

    async def aclose(self) -> None:
        self._closed = True
        executors = [executor for executor in self._executors if executor is not None]
        self._executors = [None] * len(self._executors)
        await asyncio.gather(*(
            asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
            for executor in executors
        ))

    async def __aenter__(self) -> SpawnPdfProcessRunner:
        return self

    async def __aexit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
