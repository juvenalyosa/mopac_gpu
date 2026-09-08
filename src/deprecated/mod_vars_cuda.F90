! Molecular Orbital PACkage (MOPAC)
! Copyright 2021 Virginia Polytechnic Institute and State University
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!    http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

module mod_vars_cuda
  public :: lgpu, ngpus, gpu_id, mozyme_gpu, mozyme_gpu_min_block, mozyme_force_2gpu
  public :: mozyme_gpu_requested, mozyme_gpu_plan_ready, mozyme_gpu_enabled
  public :: mozyme_gpu_disable_reason, mozyme_fock1_batch_gpu, mozyme_fock2_4x1_batch_gpu
  public :: mozyme_resident_fock_gpu, mozyme_fock_gpu, mozyme_f2_gpu, mozyme_check_gpu
  public :: resident_scf, gpu_scf_stream_available
  public :: MOZYME_GPU_REASON_NONE, MOZYME_GPU_REASON_NOT_REQUESTED, MOZYME_GPU_REASON_NO_DEVICE
  public :: MOZYME_GPU_REASON_NO_PRODUCTION_WORK, MOZYME_GPU_REASON_DEVICE_POLICY

  integer, parameter :: nthreads_gpu = 256, nblocks_gpu = 256
  logical :: lgpu = .false.
  ! MOZYME GPU state is split deliberately:
  ! requested: user/default policy asked for MOZYME GPU work.
  ! mozyme_gpu: production-safe MOZYME GPU work is currently enabled.
  ! plan_ready: the MOZYME sparse workload has been profiled after setup.
  logical :: mozyme_gpu = .false.
  logical :: mozyme_gpu_requested = .false.
  logical :: mozyme_gpu_plan_ready = .false.
  logical :: mozyme_gpu_enabled = .false.
  integer, parameter :: MOZYME_GPU_REASON_NONE = 0
  integer, parameter :: MOZYME_GPU_REASON_NOT_REQUESTED = 1
  integer, parameter :: MOZYME_GPU_REASON_NO_DEVICE = 2
  integer, parameter :: MOZYME_GPU_REASON_NO_PRODUCTION_WORK = 3
  integer, parameter :: MOZYME_GPU_REASON_DEVICE_POLICY = 4
  integer :: mozyme_gpu_disable_reason = MOZYME_GPU_REASON_NOT_REQUESTED
  logical :: mozyme_force_2gpu = .false.
  ! Production-safe first MOZYME Fock offload: batch all one-center atom tasks
  ! into one CUDA launch and scatter exact FP64 deltas back to the packed Fock.
  logical :: mozyme_fock1_batch_gpu = .false.
  ! Production-safe two-center slice for the common p-block/H branches.  These
  ! 4x1 and 1x4 interactions dominate biomolecular MOZYME pair counts.
  logical :: mozyme_fock2_4x1_batch_gpu = .false.
  ! GPU-resident sparse Fock path.  This is the production direction for large
  ! MOZYME systems: upload the sparse plan/integrals once per geometry and run
  ! fused kernels over the packed density/Fock arrays each SCF iteration.
  logical :: mozyme_resident_fock_gpu = .false.
  ! The legacy MOZYME Fock GPU wrappers are not production-safe yet.  They are
  ! kept behind an explicit development switch while the GPU-first batched
  ! MOZYME path is designed.
  logical :: mozyme_fock_gpu = .false.
  logical :: mozyme_f2_gpu = .false.
  ! Experimental MOZYME LMO validation/retry path.  This does not launch CUDA
  ! work, so production GPU enablement must not depend on it.
  logical :: mozyme_check_gpu = .false.
  ! MOZYME atom blocks are usually very small (H=1, C/N/O=4, heavier atoms larger).
  ! Default above the common 1-4 orbital blocks so GPU offload is opt-in for
  ! cases large enough to amortize transfer and library-call overhead.
  integer :: mozyme_gpu_min_block = 16
  logical :: resident_scf = .false.
  integer, parameter :: GPU_SCF_TASK_AUTO = 0
  integer, parameter :: GPU_SCF_TASK_CPU  = 1
  integer, parameter :: GPU_SCF_TASK_GPU  = 2
  integer :: gpu_scf_task_mode = GPU_SCF_TASK_AUTO
  logical :: gpu_scf_stream_available = .false.
  real, parameter :: real_cuda = selected_real_kind(8)
  integer, parameter :: prec = 8
  integer :: ngpus,gpu_id
  logical, parameter :: exe_gpu_kepler = .true.
end module mod_vars_cuda
