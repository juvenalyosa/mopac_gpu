// Molecular Orbital PACkage (MOPAC)
// Copyright 2025-2026 Juvenal Yosa, PhD (juvenal.yosa@gmail.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Device buffers that live for one call of a kernel wrapper (pair gradient, hcore pairs, dispersion,
// H-bonds), taken from CUDA's stream-ordered pool on the default stream.  With cudaMalloc / cudaFree a
// QM/MM step of trypsin-BPTI (A100) made ~100 allocations, and every cudaFree waits for the whole device.
// The pool keeps up to 256 MB between calls.  MOPAC_GPU_ASYNC_ALLOC=0: cudaMalloc / cudaFree.
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

inline bool mopac_stream_alloc_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOPAC_GPU_ASYNC_ALLOC");
    if (v && (*v == '0' || *v == 'n' || *v == 'N' || *v == 'f' || *v == 'F')) return false;
    int dev = 0;
    int pools = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, dev) != cudaSuccess || !pools) {
      cudaGetLastError();
      return false;
    }
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) {
      std::uint64_t keep = std::uint64_t(256) << 20;
      cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep);
    }
    cudaGetLastError();
    return true;
  }();
  return enabled;
}

inline cudaError_t mopac_stream_malloc(void **ptr, std::size_t bytes) {
  return mopac_stream_alloc_enabled() ? cudaMallocAsync(ptr, bytes, 0) : cudaMalloc(ptr, bytes);
}

inline void mopac_stream_free(void *ptr) {
  if (!ptr) return;
  if (mopac_stream_alloc_enabled()) {
    cudaFreeAsync(ptr, 0);
  } else {
    cudaFree(ptr);
  }
}
