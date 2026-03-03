"""
Script to run inference on .engine files
"""

# =========================================================
#                           Imports
# =========================================================

import time
import numpy as np
from pathlib import Path
from typing import Tuple

from src.benchmark.power_log import TegrastatsLogger



# =========================================================
#                      Inference Function
# =========================================================

class TRTWrapper:
    def __init__(self, engine_path: Path):
        # Scoped imports to protect non-Jetson environments
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt

        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()

        # Dynamically allocate buffers based on engine 
        # bindings and create a stream
        self._bindings = [int(0)] * self._engine.num_bindings
        
        for i in range(self._engine.num_bindings):
            shape = self._engine.get_binding_shape(i)
            size = trt.volume(shape)
            dtype = trt.nptype(self._engine.get_binding_dtype(i))
            
            # Allocate pinned memory and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self._bindings[i] = int(device_mem)
            
            if self._engine.binding_is_input(i):
                self._host_input = host_mem
                self._dev_input = device_mem
                self.batch_size = shape[0] # Dynamically capture batch size
            else:
                self._host_output = host_mem
                self._dev_output = device_mem

        self._stream = cuda.Stream()

    def infer(self, input_array: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run one inference and return (prediction_array, latency_ms).
        """
        # Ravel and copy input into pinned host memory
        np.copyto(self._host_input, input_array.ravel())

        t_start = time.perf_counter()
        
        # Async round-trip
        self._cuda.memcpy_htod_async(self._dev_input, self._host_input, self._stream)
        self._context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)
        self._cuda.memcpy_dtoh_async(self._host_output, self._dev_output, self._stream)
        
        # Wait for GPU to finish
        self._stream.synchronize()
        
        t_end = time.perf_counter()

        # Return a COPY of the full output array 
        return self._host_output.copy(), (t_end - t_start) * 1000.0

    def __del__(self):
        """Clean up pycuda memory to prevent VRAM leaks across 30 models."""
        try:
            if hasattr(self, '_dev_input'):
                self._dev_input.free()
            if hasattr(self, '_dev_output'):
                self._dev_output.free()
            if hasattr(self, '_context'):
                del self._context
            if hasattr(self, '_engine'):
                del self._engine
        except Exception:
            pass
