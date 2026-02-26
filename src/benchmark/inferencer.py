"""
TensorRT engine inference.

TRTInferencer loads a compiled .engine file and runs single-sample inference
using asynchronous CUDA streams.

Why async streams?
    All three GPU operations — copy input to device, run network, copy output
    back — are queued onto a CUDA stream in one go before we block on
    synchronize().  This avoids repeated CPU↔GPU round-trips and is how the
    model would run in a real embedded deployment, so it gives a fair latency
    measurement.

Why pinned (page-locked) memory?
    Normal numpy arrays live in pageable RAM.  The GPU can only DMA from
    page-locked memory, so CUDA would silently stage a copy to a temporary
    pinned buffer first.  Allocating pinned buffers upfront removes that hidden
    copy from the critical path.
"""

import time
from pathlib import Path
from typing import Tuple

import numpy as np


class TRTInferencer:
    """Loads a TensorRT engine and runs timed batch inference."""

    def __init__(self, engine_path: Path):
        # TensorRT and pycuda are only available on the Jetson, so we import
        # them here rather than at module level to avoid import errors on
        # development machines.
        import pycuda.autoinit  # noqa: F401 — initialises the CUDA context once
        import pycuda.driver as cuda
        import tensorrt as trt

        self._cuda = cuda

        logger  = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()

        # Query binding indices and shapes from the engine rather than
        # hard-coding them, so this works with all five model architectures.
        input_idx  = self._engine.get_binding_index("input")
        output_idx = self._engine.get_binding_index("capacity")

        input_shape  = self._engine.get_binding_shape(input_idx)   # (batch, 120)
        output_shape = self._engine.get_binding_shape(output_idx)  # (batch,) or scalar

        # batch_size is baked into the engine at compile time (static).
        # batch=1  → single-cell engine
        # batch=32 → 32-cell pack engine
        self.batch_size = int(input_shape[0])

        input_size  = int(np.prod(input_shape))
        output_size = max(self.batch_size, int(np.prod(output_shape)))

        # Allocate pinned host buffers and matching GPU device buffers
        self._host_input  = cuda.pagelocked_empty(input_size,  dtype=np.float32)
        self._host_output = cuda.pagelocked_empty(output_size, dtype=np.float32)
        self._dev_input   = cuda.mem_alloc(self._host_input.nbytes)
        self._dev_output  = cuda.mem_alloc(self._host_output.nbytes)
        self._stream      = cuda.Stream()

        # execute_async_v2 expects device pointers indexed by binding number
        self._bindings = [None, None]
        self._bindings[input_idx]  = int(self._dev_input)
        self._bindings[output_idx] = int(self._dev_output)

    def infer(self, input_array: np.ndarray) -> Tuple[float, float]:
        """
        Run one inference and return (prediction, latency_ms).

        The timer wraps the full GPU round-trip: host→device copy, network
        execution, and device→host copy.  stream.synchronize() ensures we
        don't stop the clock until all GPU work is actually complete.
        """
        np.copyto(self._host_input, input_array.ravel())

        t_start = time.perf_counter()
        self._cuda.memcpy_htod_async(self._dev_input,   self._host_input,  self._stream)
        self._context.execute_async_v2(
            bindings=self._bindings, stream_handle=self._stream.handle
        )
        self._cuda.memcpy_dtoh_async(self._host_output, self._dev_output,  self._stream)
        self._stream.synchronize()
        t_end = time.perf_counter()

        return float(self._host_output[0]), (t_end - t_start) * 1000.0
