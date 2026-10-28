"""Recovers from CUDA out-of-memory errors between OCR requests and detects an exhausted device."""

import gc
from collections.abc import Callable
from typing import TypeVar

import torch

from ocr_worker.engine_protocol import EngineUnavailableError

GuardedResult = TypeVar("GuardedResult")
ATTEMPTS_PER_CROP = 2
EXHAUSTED_CROP_LIMIT = 3


def release_cuda_cache() -> None:
    """Return the memory of dead tensors and the allocator's free blocks to the driver.

    Usage:
        Called after an out-of-memory error, once its traceback (which keeps the failed
        request's tensors alive) has been dropped.

    Returns:
        None. A no-op when CUDA was never initialised.
    """
    gc.collect()
    torch.cuda.empty_cache()


class CudaMemoryGuard:
    """Runs model calls with one cache flush and retry on CUDA out-of-memory.

    Usage:
        Create one per engine and wrap every model call with ``run_guarded``. A crop that
        still runs out of memory fails on its own; several such crops in a row mean the
        device cannot serve the model any more, and the guard raises ``EngineUnavailableError``
        so the job stops instead of writing empty results for every remaining crop.
    """

    def __init__(self, attempts_per_crop: int = ATTEMPTS_PER_CROP, exhausted_limit: int = EXHAUSTED_CROP_LIMIT,
                 cache_releaser: Callable[[], None] = release_cuda_cache) -> None:
        """Configure the retry policy.

        Args:
            attempts_per_crop: Tries per call, >= 1 (2 = one retry after a cache flush).
            exhausted_limit: Consecutive failed calls that mark the device as exhausted, >= 1.
            cache_releaser: Frees cached device memory between tries.

        Raises:
            ValueError: If a limit is below 1.
        """
        if attempts_per_crop < 1 or exhausted_limit < 1:
            raise ValueError("attempts_per_crop and exhausted_limit must be at least 1.")
        self._attempts_per_crop = attempts_per_crop
        self._exhausted_limit = exhausted_limit
        self._cache_releaser = cache_releaser
        self._failed_streak = 0

    def run_guarded(self, model_call: Callable[[], GuardedResult]) -> GuardedResult:
        """Run a model call, flushing the CUDA cache and retrying after out-of-memory.

        Args:
            model_call: Zero-argument call that runs the model once.

        Returns:
            The call's result.

        Raises:
            torch.cuda.OutOfMemoryError: If every try of this call ran out of memory.
            EngineUnavailableError: If this is the ``exhausted_limit``-th failed call in a row.
        """
        failure_text = ""
        for _ in range(self._attempts_per_crop):
            try:
                guarded_result = model_call()
            except torch.cuda.OutOfMemoryError as memory_error:
                # Keep only the text: the exception's traceback holds the failed request's tensors.
                failure_text = str(memory_error)
            else:
                self._failed_streak = 0
                return guarded_result
            self._cache_releaser()
        return self._report_failure(failure_text)

    def _report_failure(self, failure_text: str) -> GuardedResult:
        """Count a call that ran out of memory on every try and raise the matching error.

        Args:
            failure_text: Message of the last out-of-memory error.

        Returns:
            Never returns.

        Raises:
            EngineUnavailableError: When the failed streak reached ``exhausted_limit``.
            torch.cuda.OutOfMemoryError: Otherwise (only this crop fails).
        """
        self._failed_streak += 1
        if self._failed_streak >= self._exhausted_limit:
            raise EngineUnavailableError(
                f"CUDA out of memory on {self._failed_streak} crops in a row even after freeing the cache; "
                f"the GPU is held by something else or the driver refuses allocations. Last error: {failure_text}")
        raise torch.cuda.OutOfMemoryError(failure_text)
