# GPU SCF Disk-Streaming Plan

## Motivation
- Enable GPU Fock/SCF on systems that require integral disk-out ("use_disk"), typically large biomolecules.
- Avoid the current all-or-nothing CPU fallback which disables GPU acceleration entirely when integrals spill to disk.

## CPU Reference Path
- `fock2.F90` orchestrates blockwise evaluation using atom ranges (`ia..ib`, `ja..jb`).
- Integral blocks (`w`, `wj`, `wk`) are read sequentially from disk via helper routines (see `fockdorbs`, `den_in_out`).
- Each block contributes to the packed Fock matrix `f` and depends on the Coulson matrices `p`/`ptot` residing in memory.

## Proposed Streaming Architecture

### Responsibilities
1. **Fortran layer**: retain ownership of disk IO, expose a lightweight streaming API that feeds integral blocks to the GPU driver.
2. **CUDA driver (`scf_driver.cu`)**: consume streamed blocks, stage them in pinned memory, launch existing GPU kernels, and accumulate results into the device Fock buffer.
3. **Synchronization**: ensure deterministic accumulation order and safe transition back to CPU when the GPU path is not available.

### Interface Sketch
```
module gpu_scf_stream_if
  interface
    subroutine mopac_cuda_scf_stream_register(cookie) bind(C)
      import :: c_ptr
      type(c_ptr), value :: cookie
    end subroutine

    subroutine mopac_cuda_scf_stream_publish(cookie, ia, ib, ja, jb, len,
                                             wj, wk, status) bind(C)
      import :: c_ptr, c_int, c_double
      type(c_ptr), value :: cookie
      integer(c_int), value :: ia, ib, ja, jb, len
      real(c_double), intent(in) :: wj(len), wk(len)
      integer(c_int), intent(out) :: status
    end subroutine

    subroutine mopac_cuda_scf_stream_finalize(cookie, status) bind(C)
      import :: c_ptr, c_int
      type(c_ptr), value :: cookie
      integer(c_int), intent(out) :: status
    end subroutine
  end interface
end module gpu_scf_stream_if
```
- `cookie` is an opaque handle owned by Fortran (e.g. a derived type with disk state).
- `publish` is called per block; status codes bubble up errors to Fortran, which can then fall back to CPU.

### CUDA Side
- Introduce `DiskStreamContext` in `scf_driver.cu`:
  - Holds staging buffers (`PinnedHostBuffer`, `DeviceBuffer`).
  - Tracks current accumulation pointers, packed-index helpers, and error state.
- Add helper `bool scf_stream_begin(MopacGpuScfContext&, DiskStreamContext&, StreamCallbacks&)` that validates context and prepares buffers.
- Each `publish` call copies integrals to device and launches the existing pair kernels (`fock_kernels.cu`) with parameters derived from `(ia, ib, ja, jb)`.
- Upon `finalize`, flush device reductions (if any), ensure the Fock buffer is marked valid, and hand control back to the SCF loop.

### Accumulation Strategy
- Reuse the packed-index utilities already present (`packed_index_zero`) to map block contributions.
- Implement deterministic accumulation via:
  1. Device atomic adds into the packed Fock buffer, or
  2. Per-block dense tiles followed by deterministic host merge (preferred for reproducibility).
- Maintain debug hooks (e.g. `MOPAC_GPU_STREAM_DEBUG`) to log block statistics for validation.

## Migration Strategy
1. **Phase 0 (current)**: CPU-only fallback for `use_disk`; no behavioural change.
2. **Phase 1**: Implement registration and staging stubs (returning `false`) so Fortran can detect availability without altering SCF semantics.
3. **Phase 2**: Wire the streaming callbacks through `fock2.F90` (guarded by `MOPAC_GPU_STREAM=on`), execute GPU kernels per block, and validate against CPU on mid-sized benchmarks.
4. **Phase 3**: Extend to DIIS/resident caches, ensure gradients respect streamed Fock matrices, and update regression coverage.

## Open Questions
- Optimum block size for PCIe transfer vs kernel occupancy.
- Whether to pre-sort blocks by atom pair type to reduce divergence.
- Handling MOZYME/solid-state variants that already use disk.

## Next Steps
- Implement the stub registration API (Phase 1).
- Capture real block metadata from `fock2` to confirm parameter ordering.
- Prototype streaming on a reduced test (e.g. 200-atom protein fragment).
