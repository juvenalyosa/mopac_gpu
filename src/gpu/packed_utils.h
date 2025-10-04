#ifndef MOPAC_GPU_PACKED_UTILS_H
#define MOPAC_GPU_PACKED_UTILS_H

#include <cstddef>
#include <cstdint>

namespace mopac_gpu {

// Convert Fortran-style 1-based indices (i,j) with i >= j into
// a 0-based packed lower-triangular index.
__host__ __device__ inline std::size_t packed_index(int i, int j) {
  int ii = i - 1;
  int jj = j - 1;
  if (ii >= jj) {
    return static_cast<std::size_t>(ii) * (ii + 1) / 2 + jj;
  }
  return static_cast<std::size_t>(jj) * (jj + 1) / 2 + ii;
}

__host__ __device__ inline int span_count(int first, int last) {
  return (last >= first) ? (last - first + 1) : 0;
}

__host__ __device__ inline int pair_count(int span) {
  return (span > 0) ? (span * (span + 1)) / 2 : 0;
}

} // namespace mopac_gpu

#endif // MOPAC_GPU_PACKED_UTILS_H
