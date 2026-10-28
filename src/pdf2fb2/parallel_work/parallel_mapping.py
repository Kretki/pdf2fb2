"""Order-preserving parallel map over independent items, in threads or in processes.

Threads suit work that releases the GIL (OpenCV, PIL encoding, file I/O); processes suit pure-Python or
GIL-holding libraries (PyMuPDF rendering). Results always come back in input order; the progress callback is
called in completion order with the number of finished items.
"""

import multiprocessing
import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import TypeVar

ItemType = TypeVar("ItemType")
ResultType = TypeVar("ResultType")
ProgressCallback = Callable[[int, int], None]
MAX_THREAD_WORKERS = 16
MAX_PROCESS_WORKERS = 8
WORKER_COUNT_VARIABLE = "PDF2FB2_CPU_WORKERS"
# fork is fastest and works from any main module; forking while other threads run can deadlock, so then spawn.
FORK_START_METHOD = "fork"
SPAWN_START_METHOD = "spawn"


def available_workers(worker_ceiling: int) -> int:
    """Number of parallel workers to use on this machine.

    Usage:
        Called before creating a pool; ``PDF2FB2_CPU_WORKERS`` overrides the CPU count (1 disables parallelism).

    Args:
        worker_ceiling: Upper bound of the pool size (>= 1).

    Returns:
        Worker count in 1..worker_ceiling.
    """
    configured_text = os.environ.get(WORKER_COUNT_VARIABLE, "")
    worker_count = int(configured_text) if configured_text.isdigit() else (os.cpu_count() or 1)
    return min(max(worker_count, 1), worker_ceiling)


def process_start_method() -> str:
    """Start method of the worker processes.

    Returns:
        ``fork`` when the caller runs a single thread, else ``spawn``.
    """
    return FORK_START_METHOD if threading.active_count() == 1 else SPAWN_START_METHOD


def _collect_results(task_executor: Executor, work_function: Callable[[ItemType], ResultType],
                     work_items: Sequence[ItemType], progress_callback: ProgressCallback | None) -> list[ResultType]:
    """Submit every item to a pool and gather the results in input order.

    Args:
        task_executor: Open thread or process pool.
        work_function: Function applied to each item (picklable for process pools).
        work_items: Items to process.
        progress_callback: Optional ``callback(finished_count, item_count)`` after each finished item.

    Returns:
        Results aligned with ``work_items``.

    Raises:
        Exception: The first exception raised by ``work_function`` (remaining items are cancelled).
    """
    pending_futures = {task_executor.submit(work_function, work_item): item_position
                       for item_position, work_item in enumerate(work_items)}
    ordered_results: list[ResultType | None] = [None] * len(work_items)
    for finished_count, finished_future in enumerate(as_completed(pending_futures), start=1):
        ordered_results[pending_futures[finished_future]] = finished_future.result()
        if progress_callback is not None:
            progress_callback(finished_count, len(work_items))
    return ordered_results  # type: ignore[return-value]


def _map_serially(work_function: Callable[[ItemType], ResultType], work_items: Sequence[ItemType],
                  progress_callback: ProgressCallback | None) -> list[ResultType]:
    """Apply the function item by item (one worker or a single item: a pool would only add overhead).

    Args:
        work_function: Function applied to each item.
        work_items: Items to process.
        progress_callback: Optional ``callback(finished_count, item_count)``.

    Returns:
        Results aligned with ``work_items``.
    """
    serial_results = []
    for finished_count, work_item in enumerate(work_items, start=1):
        serial_results.append(work_function(work_item))
        if progress_callback is not None:
            progress_callback(finished_count, len(work_items))
    return serial_results


def map_in_threads(work_function: Callable[[ItemType], ResultType], work_items: Sequence[ItemType],
                   progress_callback: ProgressCallback | None = None) -> list[ResultType]:
    """Run ``work_function`` over the items in a thread pool.

    Usage:
        For per-page work in GIL-releasing libraries (OpenCV, PIL); the function must not share mutable state
        with other items, except writing into its own item.

    Args:
        work_function: Function applied to each item.
        work_items: Items to process.
        progress_callback: Optional ``callback(finished_count, item_count)``, called from the calling thread.

    Returns:
        Results aligned with ``work_items``.

    Raises:
        Exception: The first exception raised by ``work_function``.
    """
    worker_count = min(available_workers(MAX_THREAD_WORKERS), len(work_items))
    if worker_count <= 1:
        return _map_serially(work_function, work_items, progress_callback)
    with ThreadPoolExecutor(max_workers=worker_count) as thread_pool:
        return _collect_results(thread_pool, work_function, work_items, progress_callback)


def map_in_processes(work_function: Callable[[ItemType], ResultType], work_items: Sequence[ItemType],
                     progress_callback: ProgressCallback | None = None) -> list[ResultType]:
    """Run ``work_function`` over the items in a process pool.

    Usage:
        For GIL-holding work such as PyMuPDF rendering; the function must be a module-level function and the
        items and results picklable.

    Args:
        work_function: Module-level function applied to each item.
        work_items: Picklable items to process.
        progress_callback: Optional ``callback(finished_count, item_count)``, called in the parent process.

    Returns:
        Results aligned with ``work_items``.

    Raises:
        Exception: The first exception raised by ``work_function``.
    """
    worker_count = min(available_workers(MAX_PROCESS_WORKERS), len(work_items))
    if worker_count <= 1:
        return _map_serially(work_function, work_items, progress_callback)
    with ProcessPoolExecutor(max_workers=worker_count,
                             mp_context=multiprocessing.get_context(process_start_method())) as process_pool:
        return _collect_results(process_pool, work_function, work_items, progress_callback)
