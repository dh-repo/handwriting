"""
pipeline/training/prefetcher.py
High-Performance Asynchronous Device Batch Prefetcher for Apple Silicon Metal (MPS) and CUDA.
Provides double/triple buffered background data pre-staging with zero host-to-device compute stalls.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Sentinel object indicating end of dataset stream
_SENTINEL = object()


class _ExceptionWrapper:
    """Carries worker exceptions across thread boundary."""

    def __init__(self, exc: Exception) -> None:
        self.exception = exc


class AsyncDevicePrefetcher:
    """
    Zero-overhead asynchronous device batch prefetcher with double/triple buffering.
    
    Spawns a lightweight background daemon thread to fetch batches from PyTorch DataLoader,
    pre-cast floating point tensors to target precision (FP16/BF16/FP32), and transfer
    tensors asynchronously to the target compute device (MPS/CUDA) via non-blocking DMA
    prior to kernel execution.
    """

    def __init__(
        self,
        loader: Any,
        device: Optional[Union[str, torch.device]] = None,
        mixed_precision: str = "fp16",
        queue_size: int = 3,
    ) -> None:
        """
        Initialize the AsyncDevicePrefetcher.

        Args:
            loader: PyTorch DataLoader or batch iterable.
            device: Target compute device ('mps', 'cuda', 'cpu', or torch.device).
            mixed_precision: Precision mode ('fp16', 'bf16', 'fp32', or 'none').
            queue_size: Bounded queue capacity (default 3 for double/triple buffering).
        """
        self.loader = loader
        if device is None:
            self.device: Optional[torch.device] = None
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(str(device))

        self.mixed_precision = str(mixed_precision or "none").lower()
        self.queue_size = max(1, queue_size)
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_exception: Optional[Exception] = None
        self._iter_active: bool = False

    def _cast_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast floating-point tensor to target mixed-precision dtype."""
        if tensor.is_floating_point():
            if self.mixed_precision == "fp16" and tensor.dtype != torch.float16:
                return tensor.half()
            elif self.mixed_precision == "bf16" and tensor.dtype != torch.bfloat16:
                return tensor.bfloat16()
            elif self.mixed_precision in ("fp32", "none") and tensor.dtype not in (torch.float32, torch.float64):
                return tensor.float()
        return tensor

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transfer tensor to device non-blockingly."""
        if self.device is not None:
            return tensor.to(self.device, non_blocking=True)
        return tensor

    def _prepare_batch(self, batch: Any) -> Any:
        """Recursively process batch tensors, converting precision and moving to device."""
        if isinstance(batch, torch.Tensor):
            casted = self._cast_tensor(batch)
            return self._to_device(casted)
        elif isinstance(batch, dict):
            prepared: Dict[str, Any] = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    casted = self._cast_tensor(v)
                    prepared[k] = self._to_device(casted)
                elif isinstance(v, dict):
                    prepared[k] = self._prepare_batch(v)
                elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                    prepared[k] = [self._to_device(self._cast_tensor(t)) for t in v]
                else:
                    prepared[k] = v
            return prepared
        elif isinstance(batch, (list, tuple)):
            processed = [self._prepare_batch(item) for item in batch]
            return type(batch)(processed)
        return batch

    def _worker_loop(self) -> None:
        """Background thread routine fetching batches and populating queue."""
        try:
            for batch in self.loader:
                if self._stop_event.is_set():
                    break
                prepared_batch = self._prepare_batch(batch)

                # Push to queue with periodic check of stop_event to avoid hanging on exit
                pushed = False
                while not pushed and not self._stop_event.is_set():
                    try:
                        self._queue.put(prepared_batch, timeout=0.1)
                        pushed = True
                    except queue.Full:
                        continue

                if self._stop_event.is_set():
                    break
        except Exception as e:
            if not self._stop_event.is_set():
                logger.debug(f"AsyncDevicePrefetcher worker encountered error: {e}")
                self._worker_exception = e
                # Push exception wrapper so consumer knows error occurred
                try:
                    self._queue.put(_ExceptionWrapper(e), timeout=0.5)
                except Exception:
                    pass
        finally:
            # Signal end of stream
            while not self._stop_event.is_set():
                try:
                    self._queue.put(_SENTINEL, timeout=0.1)
                    break
                except queue.Full:
                    # Consumer might be slow or stopped
                    if self._stop_event.is_set():
                        break
                    continue

    def __iter__(self) -> AsyncDevicePrefetcher:
        """Start prefetching worker thread and return iterator."""
        if self._iter_active and self._worker_thread is not None and self._worker_thread.is_alive():
            return self
        self.close()
        self._stop_event.clear()
        self._worker_exception = None
        self._iter_active = True

        # Drain any residual items in queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Exception:
                break

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AsyncDevicePrefetcherWorker",
        )
        self._worker_thread.start()
        return self

    def __next__(self) -> Dict[str, Any]:
        """Fetch next pre-staged batch already on target device."""
        if self._worker_thread is None or (not self._worker_thread.is_alive() and self._queue.empty()):
            if self._worker_thread is None:
                # Auto-start if iterated directly with next()
                self.__iter__()

        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._worker_thread is not None and not self._worker_thread.is_alive() and self._queue.empty():
                    self._iter_active = False
                    if self._worker_exception is not None:
                        exc = self._worker_exception
                        self._worker_exception = None
                        raise exc
                    raise StopIteration
                continue

            try:
                if item is _SENTINEL:
                    self._iter_active = False
                    if self._worker_exception is not None:
                        exc = self._worker_exception
                        self._worker_exception = None
                        raise exc
                    raise StopIteration
                elif isinstance(item, _ExceptionWrapper):
                    self._iter_active = False
                    raise item.exception
                return item
            finally:
                self._queue.task_done()

    def __len__(self) -> int:
        """Return total number of batches in loader if defined."""
        if hasattr(self.loader, "__len__"):
            return len(self.loader)
        return 0

    def close(self) -> None:
        """Gracefully terminate worker thread and flush internal queues."""
        self._iter_active = False
        self._stop_event.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Exception:
                break

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self._worker_thread = None

    def __enter__(self) -> AsyncDevicePrefetcher:
        """Context manager entry starts prefetcher."""
        return self.__iter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit guarantees thread termination."""
        self.close()
