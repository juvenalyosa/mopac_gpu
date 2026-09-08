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

module mozyme_gpu_plan
  implicit none
  integer, parameter :: max_resident_plan_basis = 9
  logical, save :: plan_requested = .false.
  logical, save :: plan_disabled_early = .false.
contains

  subroutine apply_mozyme_gpu_early_plan()
    use molkst_C, only: numat
    use MOZYME_C, only: iorbs
#ifdef GPU
    use mod_vars_cuda, only: lgpu, mozyme_gpu, mozyme_gpu_requested, mozyme_gpu_plan_ready, &
      mozyme_gpu_enabled, mozyme_gpu_disable_reason, mozyme_gpu_min_block, mozyme_fock1_batch_gpu, &
      mozyme_fock2_4x1_batch_gpu, mozyme_resident_fock_gpu, mozyme_fock_gpu, mozyme_f2_gpu, resident_scf, &
      gpu_scf_stream_available, MOZYME_GPU_REASON_NOT_REQUESTED, &
      MOZYME_GPU_REASON_NO_DEVICE, MOZYME_GPU_REASON_NO_PRODUCTION_WORK
#endif
    implicit none
    integer :: i, j, eligible_density_pairs
    logical :: strict_required
#ifdef GPU
    strict_required = mozyme_plan_env_enabled('MOPAC_MOZYME_SCF_STRICT_RESIDENT') .or. &
      mozyme_plan_env_enabled('MOPAC_MOZYME_SCF_GPU') .or. &
      mozyme_plan_env_enabled('MOPAC_MOZYME_GPU_STRICT') .or. &
      mozyme_plan_env_enabled('MOPAC_MOZYME_FULL_SCF_GPU')
    if (.not. mozyme_gpu_requested .and. .not. mozyme_gpu) then
      plan_requested = .false.
      plan_disabled_early = .false.
      mozyme_gpu_plan_ready = .false.
      mozyme_gpu_enabled = .false.
      mozyme_gpu_disable_reason = MOZYME_GPU_REASON_NOT_REQUESTED
      return
    end if

    plan_requested = mozyme_gpu_requested .or. mozyme_gpu
    plan_disabled_early = .false.
    mozyme_gpu_plan_ready = .false.
    mozyme_gpu_enabled = mozyme_gpu .and. lgpu
    if (.not. lgpu) then
      mozyme_gpu = .false.
      mozyme_gpu_enabled = .false.
      mozyme_gpu_disable_reason = MOZYME_GPU_REASON_NO_DEVICE
      plan_disabled_early = .true.
      return
    end if
    if (.not. mozyme_gpu) then
      plan_disabled_early = .true.
      return
    end if
    eligible_density_pairs = 0
    do i = 1, numat
      do j = 1, i
        if (iorbs(i) >= mozyme_gpu_min_block .and. iorbs(j) >= mozyme_gpu_min_block) then
          eligible_density_pairs = eligible_density_pairs + 1
        end if
      end do
    end do

    if (.not. mozyme_resident_fock_gpu .and. .not. mozyme_fock1_batch_gpu .and. .not. mozyme_fock2_4x1_batch_gpu .and. &
        .not. (mozyme_fock_gpu .and. mozyme_f2_gpu) .and. eligible_density_pairs == 0) then
      if (strict_required) then
        resident_scf = .true.
        gpu_scf_stream_available = .true.
      else
        ! Disable GPU before MOZYME sparse setup.  The full profiler runs after
        ! fillij(.false.), when the sparse interaction map is actually available.
        mozyme_gpu = .false.
        lgpu = .false.
        resident_scf = .false.
        gpu_scf_stream_available = .false.
        mozyme_gpu_enabled = .false.
        mozyme_gpu_disable_reason = MOZYME_GPU_REASON_NO_PRODUCTION_WORK
        plan_disabled_early = .true.
      end if
    end if
#endif
  end subroutine apply_mozyme_gpu_early_plan

  subroutine report_mozyme_gpu_plan()
    use chanel_C, only: iw
    use molkst_C, only: numat, id
    use MOZYME_C, only: direct, iorbs, lijbo, nijbo
#ifdef GPU
    use mod_vars_cuda, only: lgpu, mozyme_gpu, mozyme_gpu_requested, mozyme_gpu_plan_ready, &
      mozyme_gpu_enabled, mozyme_gpu_disable_reason, mozyme_gpu_min_block, mozyme_fock1_batch_gpu, &
      mozyme_fock2_4x1_batch_gpu, mozyme_resident_fock_gpu, &
      mozyme_fock_gpu, mozyme_f2_gpu, mozyme_check_gpu, resident_scf, &
      gpu_scf_stream_available, MOZYME_GPU_REASON_NONE, &
      MOZYME_GPU_REASON_NOT_REQUESTED, MOZYME_GPU_REASON_NO_DEVICE, &
      MOZYME_GPU_REASON_NO_PRODUCTION_WORK
#endif
    implicit none
    integer :: i, j, bs_i, bs_j, bs_lo, bs_hi, max_block, ij_addr, plan_ione
    integer :: tri_i, tri_j
    integer :: atom_hist(0:9), pair_hist(0:9,0:9), fock_pair_hist(0:9,0:9)
    integer :: eligible_density_pairs
    integer :: env_len, env_status
    integer :: fock_one_center_tasks, fock_two_center_pairs, fock_skipped_pairs, fock_point_charge_pairs
    integer :: fock_point_dipole_pairs, fock_point_monopole_pairs
    integer :: fock_d_pairs, fock_candidate_gpu_tasks, fock_production_gpu_tasks
    integer :: fock_resident_executable_tasks
    integer :: fock_resident_supported_one_center, fock_resident_unsupported_one_center
    integer :: fock_resident_supported_pairs, fock_resident_unsupported_pairs
    integer :: fock_resident_supported_point_pairs, fock_resident_unsupported_point_pairs
    integer :: fock_resident_noop_pairs
    integer :: fock_resident_basis_limit_pairs, fock_resident_direct_basis_pairs
    integer :: fock_resident_other_pairs
    integer :: fock_resident_basis_limit_point_pairs, fock_resident_direct_basis_point_pairs
    integer :: fock_resident_other_point_pairs
    integer(kind=8) :: fock_one_center_terms, fock_two_center_terms
    logical :: trace_plan, disabled_no_work, preflight_stop
    logical :: fock_resident_full_coverage_planned
    logical :: strict_required
    character(len=16) :: env_value, value
    integer, external :: ijbo
    external :: mopend, mozyme_gpu_strict_abort

#ifdef GPU
    if (.not. mozyme_gpu_requested .and. .not. mozyme_gpu .and. .not. plan_requested) return

    strict_required = mozyme_plan_env_enabled('MOPAC_MOZYME_SCF_STRICT_RESIDENT') .or. &
      mozyme_plan_env_enabled('MOPAC_MOZYME_SCF_GPU') .or. &
      mozyme_plan_env_enabled('MOPAC_MOZYME_GPU_STRICT') .or. &
      mozyme_plan_env_enabled('MOPAC_MOZYME_FULL_SCF_GPU')
    trace_plan = .false.
    preflight_stop = .false.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_PROFILE', env_value, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') trace_plan = .true.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_DEBUG', env_value, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') trace_plan = .true.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_VERBOSE', env_value, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') trace_plan = .true.
    env_value = ' '
    call get_environment_variable('MOPAC_MOZYME_GPU_PREFLIGHT_STOP', env_value, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') then
      preflight_stop = strict_required
      trace_plan = .true.
    end if

    atom_hist = 0
    pair_hist = 0
    fock_pair_hist = 0
    eligible_density_pairs = 0
    max_block = 0
    fock_one_center_tasks = 0
    fock_two_center_pairs = 0
    fock_skipped_pairs = 0
    fock_point_charge_pairs = 0
    fock_point_dipole_pairs = 0
    fock_point_monopole_pairs = 0
    fock_d_pairs = 0
    fock_candidate_gpu_tasks = 0
    fock_production_gpu_tasks = 0
    fock_resident_executable_tasks = 0
    fock_resident_supported_one_center = 0
    fock_resident_unsupported_one_center = 0
    fock_resident_supported_pairs = 0
    fock_resident_unsupported_pairs = 0
    fock_resident_supported_point_pairs = 0
    fock_resident_unsupported_point_pairs = 0
    fock_resident_noop_pairs = 0
    fock_resident_basis_limit_pairs = 0
    fock_resident_direct_basis_pairs = 0
    fock_resident_other_pairs = 0
    fock_resident_basis_limit_point_pairs = 0
    fock_resident_direct_basis_point_pairs = 0
    fock_resident_other_point_pairs = 0
    fock_resident_full_coverage_planned = .false.
    fock_one_center_terms = 0_8
    fock_two_center_terms = 0_8

    do i = 1, numat
      bs_i = max(0, min(9, iorbs(i)))
      atom_hist(bs_i) = atom_hist(bs_i) + 1
      max_block = max(max_block, bs_i)
      if (iorbs(i) > 0) then
        tri_i = (iorbs(i) * (iorbs(i) + 1)) / 2
        fock_one_center_tasks = fock_one_center_tasks + 1
        fock_one_center_terms = fock_one_center_terms + int(tri_i, kind=8) * int(tri_i, kind=8)
        if (mozyme_plan_resident_basis_supported(iorbs(i))) then
          fock_resident_supported_one_center = fock_resident_supported_one_center + 1
        else
          fock_resident_unsupported_one_center = fock_resident_unsupported_one_center + 1
        end if
      end if
    end do

    do i = 1, numat
      bs_i = max(0, min(9, iorbs(i)))
      do j = 1, i
        bs_j = max(0, min(9, iorbs(j)))
        bs_hi = max(bs_i, bs_j)
        bs_lo = min(bs_i, bs_j)
        pair_hist(bs_hi, bs_lo) = pair_hist(bs_hi, bs_lo) + 1
        if (bs_i >= mozyme_gpu_min_block .and. bs_j >= mozyme_gpu_min_block) then
          eligible_density_pairs = eligible_density_pairs + 1
        end if
      end do
    end do

    if (id == 0) then
      plan_ione = 1
    else
      plan_ione = 0
    end if
    do i = 1, numat
      bs_i = max(0, min(9, iorbs(i)))
      tri_i = (iorbs(i) * (iorbs(i) + 1)) / 2
      do j = 1, i - plan_ione
        bs_j = max(0, min(9, iorbs(j)))
        if (lijbo .and. allocated(nijbo)) then
          ij_addr = nijbo(i, j)
        else
          ij_addr = ijbo(i, j)
        end if
        if (ij_addr >= 0) then
          tri_j = (iorbs(j) * (iorbs(j) + 1)) / 2
          bs_hi = max(bs_i, bs_j)
          bs_lo = min(bs_i, bs_j)
          fock_pair_hist(bs_hi, bs_lo) = fock_pair_hist(bs_hi, bs_lo) + 1
          fock_two_center_pairs = fock_two_center_pairs + 1
          fock_two_center_terms = fock_two_center_terms + int(tri_i, kind=8) * int(tri_j, kind=8)
          if (iorbs(i) > 5 .or. iorbs(j) > 5) fock_d_pairs = fock_d_pairs + 1
          if (mozyme_plan_resident_pair_supported(iorbs(i), iorbs(j))) then
            fock_resident_supported_pairs = fock_resident_supported_pairs + 1
          else if (mozyme_plan_resident_pair_noop(iorbs(i), iorbs(j))) then
            fock_resident_noop_pairs = fock_resident_noop_pairs + 1
          else
            fock_resident_unsupported_pairs = fock_resident_unsupported_pairs + 1
            if (mozyme_plan_resident_basis_limit_fallback(iorbs(i), iorbs(j))) then
              fock_resident_basis_limit_pairs = fock_resident_basis_limit_pairs + 1
            else if (mozyme_plan_resident_direct_basis_fallback(iorbs(i), iorbs(j))) then
              fock_resident_direct_basis_pairs = fock_resident_direct_basis_pairs + 1
            else
              fock_resident_other_pairs = fock_resident_other_pairs + 1
            end if
          end if
        else
          fock_skipped_pairs = fock_skipped_pairs + 1
          if (bs_i * bs_j > 0) then
            fock_point_charge_pairs = fock_point_charge_pairs + 1
            if (mozyme_plan_resident_point_supported(iorbs(i), iorbs(j), ij_addr)) then
              fock_resident_supported_point_pairs = fock_resident_supported_point_pairs + 1
            else
              fock_resident_unsupported_point_pairs = fock_resident_unsupported_point_pairs + 1
              if (mozyme_plan_resident_basis_limit_fallback(iorbs(i), iorbs(j))) then
                fock_resident_basis_limit_point_pairs = fock_resident_basis_limit_point_pairs + 1
              else if (mozyme_plan_resident_direct_basis_fallback(iorbs(i), iorbs(j))) then
                fock_resident_direct_basis_point_pairs = fock_resident_direct_basis_point_pairs + 1
              else
                fock_resident_other_point_pairs = fock_resident_other_point_pairs + 1
              end if
            end if
            if (ij_addr == -2) then
              fock_point_dipole_pairs = fock_point_dipole_pairs + 1
            else
              fock_point_monopole_pairs = fock_point_monopole_pairs + 1
            end if
          end if
        end if
      end do
    end do
    fock_candidate_gpu_tasks = fock_one_center_tasks + fock_two_center_pairs + fock_point_charge_pairs
    fock_resident_full_coverage_planned = &
      fock_resident_unsupported_one_center == 0 .and. &
      fock_resident_unsupported_pairs == 0 .and. fock_resident_unsupported_point_pairs == 0
    if (fock_resident_full_coverage_planned) then
      fock_resident_executable_tasks = fock_resident_supported_one_center + fock_resident_supported_pairs + &
        fock_resident_supported_point_pairs
    else
      fock_resident_executable_tasks = 0
    end if
    if (mozyme_resident_fock_gpu) then
      fock_production_gpu_tasks = fock_resident_executable_tasks
    else
      if (mozyme_fock1_batch_gpu) fock_production_gpu_tasks = fock_one_center_tasks
      if (mozyme_fock2_4x1_batch_gpu) fock_production_gpu_tasks = fock_production_gpu_tasks + fock_pair_hist(4,1)
    end if
    if (mozyme_fock_gpu .and. mozyme_f2_gpu) fock_production_gpu_tasks = fock_candidate_gpu_tasks

    disabled_no_work = plan_disabled_early
    if (fock_production_gpu_tasks == 0 .and. eligible_density_pairs == 0) then
      if (strict_required) then
        call mozyme_gpu_strict_abort('strict_no_gpu_work', &
          'Strict MOZYME full-SCF GPU requested, but the MOZYME GPU plan has no GPU work')
      end if
      ! No production MOZYME GPU path can execute for this molecule.  Disable the
      ! global GPU switch too because generic BLAS/eigensolver helpers also key
      ! off lgpu and are not a valid accelerator for sparse MOZYME in this state.
      mozyme_gpu = .false.
      lgpu = .false.
      resident_scf = .false.
      gpu_scf_stream_available = .false.
      disabled_no_work = .true.
      mozyme_gpu_disable_reason = MOZYME_GPU_REASON_NO_PRODUCTION_WORK
    else if (.not. mozyme_gpu_requested) then
      mozyme_gpu_disable_reason = MOZYME_GPU_REASON_NOT_REQUESTED
    else if (.not. lgpu) then
      mozyme_gpu_disable_reason = MOZYME_GPU_REASON_NO_DEVICE
    else
      mozyme_gpu_disable_reason = MOZYME_GPU_REASON_NONE
    end if
    if ((mozyme_resident_fock_gpu .or. mozyme_fock1_batch_gpu .or. mozyme_fock2_4x1_batch_gpu) .and. &
        .not. (mozyme_fock_gpu .and. mozyme_f2_gpu) .and. &
        eligible_density_pairs == 0) then
      if (strict_required) then
        mozyme_gpu = .true.
        resident_scf = .true.
        gpu_scf_stream_available = .true.
      else
        ! Keep only explicit batched sparse Fock kernels enabled.  Generic
        ! lgpu/MOZYME_GPU paths are too broad for sparse MOZYME and can activate
        ! unrelated experimental code after the planner has already approved the
        ! isolated production kernels.  Explicitly requested DIAGG stage
        ! offloads are the exception: they need mozyme_gpu to stay on.
        mozyme_gpu = mozyme_plan_env_enabled('MOPAC_MOZYME_DIAGG1_CONSTRUCT_GPU') .or. &
          mozyme_plan_env_enabled('MOPAC_MOZYME_DIAGG2_ROTATE_GPU')
        if (mozyme_resident_fock_gpu) then
          resident_scf = .true.
          gpu_scf_stream_available = .true.
        else
          lgpu = .false.
          resident_scf = .false.
          gpu_scf_stream_available = .false.
        end if
      end if
    end if
    mozyme_gpu_enabled = mozyme_resident_fock_gpu .or. mozyme_fock1_batch_gpu .or. mozyme_fock2_4x1_batch_gpu .or. &
      (mozyme_gpu .and. lgpu)
    mozyme_gpu_plan_ready = .true.

    if (.not. trace_plan) return

    write(iw,'(1x,a)') '[MOZYME_GPU_PLAN_BEGIN]'
    write(iw,'(1x,a)') 'version=1'
    write(iw,'(1x,a)') 'phase=full'
    write(iw,'(1x,a,l1)') 'requested=', mozyme_gpu_requested
    write(iw,'(1x,a,l1)') 'plan_ready=', mozyme_gpu_plan_ready
    write(iw,'(1x,a,l1)') 'enabled=', mozyme_gpu_enabled
    write(iw,'(1x,a,l1)') 'lgpu=', lgpu
    write(iw,'(1x,a,l1)') 'mozyme_gpu=', mozyme_gpu
    write(iw,'(1x,a,l1)') 'resident_fock_gpu=', mozyme_resident_fock_gpu
    write(iw,'(1x,a,l1)') 'direct=', direct
    write(iw,'(1x,a,l1)') 'fock1_batch_gpu=', mozyme_fock1_batch_gpu
    write(iw,'(1x,a,l1)') 'fock2_4x1_batch_gpu=', mozyme_fock2_4x1_batch_gpu
    write(iw,'(1x,a,l1)') 'fock_gpu=', mozyme_fock_gpu
    write(iw,'(1x,a,l1)') 'f2_gpu=', mozyme_f2_gpu
    write(iw,'(1x,a,l1)') 'check_gpu=', mozyme_check_gpu
    write(iw,'(1x,a,l1)') 'disabled_no_work=', disabled_no_work
    write(iw,'(1x,a,i0)') 'disable_reason=', mozyme_gpu_disable_reason
    write(iw,'(1x,a,i0)') 'minblk=', mozyme_gpu_min_block
    write(iw,'(1x,a,i0)') 'max_block=', max_block
    write(iw,'(1x,a,i0)') 'atom_pairs=', numat * (numat + 1) / 2
    write(iw,'(1x,a,i0)') 'atoms_1=', atom_hist(1)
    write(iw,'(1x,a,i0)') 'atoms_4=', atom_hist(4)
    write(iw,'(1x,a,i0)') 'atoms_9=', atom_hist(9)
    write(iw,'(1x,a,i0)') 'density_pairs_meeting_minblk=', eligible_density_pairs
    write(iw,'(1x,a,i0)') 'fock_one_center_tasks=', fock_one_center_tasks
    write(iw,'(1x,a,i0)') 'fock_resident_supported_one_center=', fock_resident_supported_one_center
    write(iw,'(1x,a,i0)') 'fock_resident_unsupported_one_center=', fock_resident_unsupported_one_center
    write(iw,'(1x,a,i0)') 'fock_two_center_tasks=', fock_two_center_pairs
    write(iw,'(1x,a,i0)') 'fock_skipped_pairs=', fock_skipped_pairs
    write(iw,'(1x,a,i0)') 'fock_point_charge_pairs=', fock_point_charge_pairs
    write(iw,'(1x,a,i0)') 'fock_point_dipole_pairs=', fock_point_dipole_pairs
    write(iw,'(1x,a,i0)') 'fock_point_monopole_pairs=', fock_point_monopole_pairs
    write(iw,'(1x,a,i0)') 'fock_d_pairs=', fock_d_pairs
    write(iw,'(1x,a,i0)') 'fock_pairs_4x4=', fock_pair_hist(4,4)
    write(iw,'(1x,a,i0)') 'fock_pairs_4x1=', fock_pair_hist(4,1)
    write(iw,'(1x,a,i0)') 'fock_pairs_9x4=', fock_pair_hist(9,4)
    write(iw,'(1x,a,i0)') 'fock_pairs_9x9=', fock_pair_hist(9,9)
    write(iw,'(1x,a,i0)') 'fock_resident_supported_pairs=', fock_resident_supported_pairs
    write(iw,'(1x,a,i0)') 'fock_resident_unsupported_pairs=', fock_resident_unsupported_pairs
    write(iw,'(1x,a,i0)') 'fock_resident_noop_pairs=', fock_resident_noop_pairs
    write(iw,'(1x,a,i0)') 'fock_resident_basis_limit_unsupported_pairs=', fock_resident_basis_limit_pairs
    write(iw,'(1x,a,i0)') 'fock_resident_direct_unsupported_pairs=', fock_resident_direct_basis_pairs
    write(iw,'(1x,a,i0)') 'fock_resident_other_unsupported_pairs=', fock_resident_other_pairs
    write(iw,'(1x,a,i0)') 'fock_resident_supported_point_pairs=', fock_resident_supported_point_pairs
    write(iw,'(1x,a,i0)') 'fock_resident_unsupported_point_pairs=', fock_resident_unsupported_point_pairs
    write(iw,'(1x,a,i0)') 'fock_resident_basis_limit_unsupported_point_pairs=', &
      fock_resident_basis_limit_point_pairs
    write(iw,'(1x,a,i0)') 'fock_resident_direct_unsupported_point_pairs=', &
      fock_resident_direct_basis_point_pairs
    write(iw,'(1x,a,i0)') 'fock_resident_other_unsupported_point_pairs=', &
      fock_resident_other_point_pairs
    write(iw,'(1x,a,l1)') 'fock_resident_full_coverage_planned=', &
      fock_resident_full_coverage_planned
    write(iw,'(1x,a,i0)') 'fock_resident_executable_tasks=', fock_resident_executable_tasks
    write(iw,'(1x,a,i0)') 'fock_candidate_gpu_tasks=', fock_candidate_gpu_tasks
    write(iw,'(1x,a,i0)') 'fock_production_gpu_tasks=', fock_production_gpu_tasks
    write(iw,'(1x,a,i0)') 'fock_one_center_terms=', fock_one_center_terms
    write(iw,'(1x,a,i0)') 'fock_two_center_terms=', fock_two_center_terms
    write(iw,'(1x,a)') '[MOZYME_GPU_PLAN_END]'
    write(iw,'(1x,a,l1,a,l1,a,i0,a,i0)') '[MOZYME GPU plan] lgpu=', lgpu, ' fock_gpu=', &
      mozyme_fock_gpu, ' minblk=', mozyme_gpu_min_block, ' max_block=', max_block
    write(iw,'(1x,a,l1)') '[MOZYME GPU plan] check_gpu=', mozyme_check_gpu
    write(iw,'(1x,a,l1)') '[MOZYME GPU plan] resident_fock_gpu=', mozyme_resident_fock_gpu
    write(iw,'(1x,a,l1)') '[MOZYME GPU plan] direct=', direct
    write(iw,'(1x,a,l1)') '[MOZYME GPU plan] fock1_batch_gpu=', mozyme_fock1_batch_gpu
    write(iw,'(1x,a,l1)') '[MOZYME GPU plan] fock2_4x1_batch_gpu=', mozyme_fock2_4x1_batch_gpu
    write(iw,'(1x,a,l1,a,l1)') &
      '[MOZYME GPU plan] mozyme_gpu_after_plan=', mozyme_gpu, 'disabled_no_work=', disabled_no_work
    write(iw,'(1x,a,i0,a,i0,a,i0,a,i0)') '[MOZYME GPU plan] atoms_1=', atom_hist(1), ' atoms_4=', &
      atom_hist(4), ' atoms_9=', atom_hist(9), ' atom_pairs=', numat * (numat + 1) / 2
    write(iw,'(1x,a,i0,a,i0,a,i0,a,i0)') '[MOZYME GPU plan] pairs_1x1=', pair_hist(1,1), &
      ' pairs_4x1=', pair_hist(4,1), ' pairs_4x4=', pair_hist(4,4), ' pairs_9x9=', pair_hist(9,9)
    write(iw,'(1x,a,i0)') '[MOZYME GPU plan] density_pairs_meeting_minblk=', eligible_density_pairs
    write(iw,'(1x,a,i0,a,i0,a,i0,a,i0)') '[MOZYME GPU fock plan] one_center=', fock_one_center_tasks, &
      ' two_center=', fock_two_center_pairs, ' skipped=', fock_skipped_pairs, ' d_pairs=', fock_d_pairs
    write(iw,'(1x,a,i0,a,i0,a,i0)') '[MOZYME GPU fock plan] point_charge_pairs=', fock_point_charge_pairs, &
      ' point_dipole_pairs=', fock_point_dipole_pairs, ' point_monopole_pairs=', fock_point_monopole_pairs
    write(iw,'(1x,a,i0,a,i0,a,i0,a,i0)') '[MOZYME GPU fock plan] pairs_4x4=', fock_pair_hist(4,4), &
      ' pairs_4x1=', fock_pair_hist(4,1), ' pairs_9x4=', fock_pair_hist(9,4), ' pairs_9x9=', fock_pair_hist(9,9)
    write(iw,'(1x,a,i0,a,i0,a,i0,a,i0)') '[MOZYME GPU fock plan] resident_supported_real_pairs=', &
      fock_resident_supported_pairs, ' resident_unsupported_real_pairs=', fock_resident_unsupported_pairs, &
      ' resident_supported_point_pairs=', fock_resident_supported_point_pairs, &
      ' resident_unsupported_point_pairs=', fock_resident_unsupported_point_pairs
    write(iw,'(1x,a,i0,a,i0)') '[MOZYME GPU fock plan] resident_one_center supported=', &
      fock_resident_supported_one_center, ' unsupported=', fock_resident_unsupported_one_center
    write(iw,'(1x,a,i0,a,i0,a,i0,a,i0)') '[MOZYME GPU fock plan] fallback_real_pairs basis_limit=', &
      fock_resident_basis_limit_pairs, ' direct_basis=', fock_resident_direct_basis_pairs, &
      ' other=', fock_resident_other_pairs, ' noop=', fock_resident_noop_pairs
    write(iw,'(1x,a,i0,a,i0,a,i0)') '[MOZYME GPU fock plan] fallback_point_pairs basis_limit=', &
      fock_resident_basis_limit_point_pairs, ' direct_basis=', fock_resident_direct_basis_point_pairs, &
      ' other=', fock_resident_other_point_pairs
    write(iw,'(1x,a,l1,a,i0)') '[MOZYME GPU fock plan] resident_full_coverage_planned=', &
      fock_resident_full_coverage_planned, ' resident_executable_tasks=', fock_resident_executable_tasks
    write(iw,'(1x,a,i0,a,i0,a,i0)') '[MOZYME GPU fock plan] candidate_gpu_tasks=', fock_candidate_gpu_tasks, &
      ' production_gpu_tasks=', fock_production_gpu_tasks, ' one_center_terms=', fock_one_center_terms
    write(iw,'(1x,a,i0)') '[MOZYME GPU fock plan] two_center_terms=', fock_two_center_terms
    if (.not. (mozyme_fock_gpu .and. mozyme_f2_gpu)) then
      write(iw,'(1x,a,a)') '[MOZYME GPU plan] legacy_fock_gpu=off; ', &
        'production two-center GPU Fock uses resident sparse Fock when resident_fock_gpu=T'
    end if
    call flush(iw)

    if (disabled_no_work .and. preflight_stop) then
      call mopend('MOZYME_GPU preflight found no GPU production work for this system')
      error stop 'MOZYME_GPU preflight found no GPU production work'
    end if
#endif
  end subroutine report_mozyme_gpu_plan

  logical function mozyme_plan_env_enabled(var_name)
    implicit none
    character(len=*), intent(in) :: var_name
    character(len=16) :: env_value, value
    integer :: env_len, env_status

    env_value = ' '
    call get_environment_variable(var_name, env_value, length=env_len, status=env_status)
    mozyme_plan_env_enabled = .false.
    if (env_status /= 0 .or. env_len <= 0) return
    value = adjustl(env_value)
    if (len_trim(value) == 0) return
    select case (trim(value))
    case ('0', 'f', 'F', 'false', 'FALSE', 'False', &
          'n', 'N', 'no', 'NO', 'No', 'off', 'OFF', 'Off')
      mozyme_plan_env_enabled = .false.
    case default
      mozyme_plan_env_enabled = .true.
    end select
  end function mozyme_plan_env_enabled

  logical function mozyme_plan_resident_pair_supported(iab, jba)
    use MOZYME_C, only: direct
    implicit none
    integer, intent(in) :: iab, jba
    if (direct) then
      mozyme_plan_resident_pair_supported = mozyme_plan_resident_direct_basis_supported(iab) .and. &
        mozyme_plan_resident_direct_basis_supported(jba)
    else
      mozyme_plan_resident_pair_supported = mozyme_plan_resident_basis_supported(iab) .and. &
        mozyme_plan_resident_basis_supported(jba)
    end if
  end function mozyme_plan_resident_pair_supported

  logical function mozyme_plan_resident_pair_noop(iab, jba)
    implicit none
    integer, intent(in) :: iab, jba
    mozyme_plan_resident_pair_noop = iab == 0 .or. jba == 0
  end function mozyme_plan_resident_pair_noop

  logical function mozyme_plan_resident_point_supported(iab, jba, addr)
    use MOZYME_C, only: direct
    implicit none
    integer, intent(in) :: iab, jba, addr
    if (direct) then
      mozyme_plan_resident_point_supported = addr < 0 .and. &
        mozyme_plan_resident_direct_basis_supported(iab) .and. &
        mozyme_plan_resident_direct_basis_supported(jba)
    else
      mozyme_plan_resident_point_supported = addr < 0 .and. &
        mozyme_plan_resident_basis_supported(iab) .and. &
        mozyme_plan_resident_basis_supported(jba)
    end if
  end function mozyme_plan_resident_point_supported

  logical function mozyme_plan_resident_basis_supported(nbasis)
    implicit none
    integer, intent(in) :: nbasis
    mozyme_plan_resident_basis_supported = nbasis == 1 .or. nbasis == 4 .or. nbasis == 9
  end function mozyme_plan_resident_basis_supported

  logical function mozyme_plan_resident_direct_basis_supported(nbasis)
    implicit none
    integer, intent(in) :: nbasis
    mozyme_plan_resident_direct_basis_supported = nbasis == 1 .or. nbasis == 4 .or. nbasis == 9
  end function mozyme_plan_resident_direct_basis_supported

  logical function mozyme_plan_resident_basis_limit_fallback(iab, jba)
    implicit none
    integer, intent(in) :: iab, jba
    mozyme_plan_resident_basis_limit_fallback = &
      iab > max_resident_plan_basis .or. jba > max_resident_plan_basis
  end function mozyme_plan_resident_basis_limit_fallback

  logical function mozyme_plan_resident_direct_basis_fallback(iab, jba)
    use MOZYME_C, only: direct
    implicit none
    integer, intent(in) :: iab, jba
    mozyme_plan_resident_direct_basis_fallback = direct .and. &
      mozyme_plan_resident_basis_supported(iab) .and. &
      mozyme_plan_resident_basis_supported(jba) .and. &
      .not. (mozyme_plan_resident_direct_basis_supported(iab) .and. &
      mozyme_plan_resident_direct_basis_supported(jba))
  end function mozyme_plan_resident_direct_basis_fallback

end module mozyme_gpu_plan
