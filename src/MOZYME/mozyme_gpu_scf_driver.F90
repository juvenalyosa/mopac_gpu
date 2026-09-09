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

module mozyme_gpu_scf_driver
#ifdef GPU
  use iso_c_binding, only: c_int, c_double, c_size_t, c_ptr, c_null_ptr, &
    c_associated
  use gpu_mozyme_scf_interfaces, only: gpu_mozyme_scf_config, &
    gpu_mozyme_scf_state, gpu_mozyme_scf_status, &
    GPU_MOZYME_SCF_ABI_VERSION, GPU_MOZYME_SCF_SUCCESS, &
    GPU_MOZYME_SCF_NOT_READY, GPU_MOZYME_SCF_BAD_ARGUMENT, &
    GPU_MOZYME_SCF_UNSUPPORTED, GPU_MOZYME_SCF_CPU_BOUNDARY, &
    GPU_MOZYME_SCF_RESIDENT_DECISION_COMPLETE, &
    GPU_MOZYME_SCF_STAGE_FULL, GPU_MOZYME_SCF_STAGE_UPLOAD, &
    GPU_MOZYME_SCF_STAGE_EIMP, GPU_MOZYME_SCF_STAGE_DIAGG, &
    GPU_MOZYME_SCF_STAGE_DENSITY, GPU_MOZYME_SCF_STAGE_FOCK, &
    GPU_MOZYME_SCF_STAGE_CNVGZ, GPU_MOZYME_SCF_STAGE_HELECZ, &
    GPU_MOZYME_SCF_STAGE_ISITSC, GPU_MOZYME_SCF_STAGE_ADDHB, &
    GPU_MOZYME_SCF_STAGE_CHECK, &
    mopac_cuda_mozyme_scf_setup, mopac_cuda_mozyme_scf_register_state, &
    mopac_cuda_mozyme_scf_run, mopac_cuda_mozyme_scf_destroy, &
    mopac_cuda_mozyme_scf_status
#endif
  implicit none
  private

  logical, save :: request_checked = .false.
  logical, save :: request_present = .false.
  logical, save :: request_enabled = .false.
  logical, save :: no_fallback_required = .false.
  character(len=64), save :: request_reason = 'not_requested'
  logical, save :: profile_checked = .false.
  logical, save :: profile_enabled = .false.
  integer, save :: blocked_nscf = -1

#ifdef GPU
  type(c_ptr), save :: scf_context = c_null_ptr
#endif

  public :: mozyme_gpu_scf_requested
  public :: mozyme_gpu_scf_strict_resident
  public :: mozyme_gpu_scf_no_fallback_required
  public :: mozyme_gpu_scf_reset_request_state
  public :: mozyme_gpu_scf_force_final_reorth
  public :: mozyme_gpu_scf_early_probe
  public :: mozyme_gpu_scf_try

contains

  logical function mozyme_gpu_scf_requested()
    implicit none

    call ensure_request_state()
    mozyme_gpu_scf_requested = request_enabled
  end function mozyme_gpu_scf_requested

  logical function mozyme_gpu_scf_strict_resident()
    implicit none

    mozyme_gpu_scf_strict_resident = &
      env_is_one('MOPAC_MOZYME_SCF_STRICT_RESIDENT') .or. &
      env_is_one('MOPAC_MOZYME_SCF_GPU') .or. &
      env_is_one('MOPAC_MOZYME_GPU_STRICT') .or. &
      env_is_one('MOPAC_MOZYME_FULL_SCF_GPU')
  end function mozyme_gpu_scf_strict_resident

  logical function mozyme_gpu_scf_no_fallback_required()
    implicit none

    call ensure_request_state()
    mozyme_gpu_scf_no_fallback_required = no_fallback_required
  end function mozyme_gpu_scf_no_fallback_required

  subroutine mozyme_gpu_scf_reset_request_state()
    implicit none

    request_checked = .false.
    request_present = .false.
    request_enabled = .false.
    no_fallback_required = .false.
    request_reason = 'not_requested'
    blocked_nscf = -1
  end subroutine mozyme_gpu_scf_reset_request_state

  logical function mozyme_gpu_scf_force_final_reorth()
    implicit none

    mozyme_gpu_scf_force_final_reorth = &
      env_is_one('MOPAC_MOZYME_SCF_FORCE_FINAL_REORTH')
  end function mozyme_gpu_scf_force_final_reorth

  logical function mozyme_gpu_scf_early_probe(niter, nocc, nvir, itrmax, &
      selcon) result(handled)
    use chanel_C, only: iw
    implicit none
    integer, intent(in) :: niter, nocc, nvir, itrmax
    double precision, intent(in) :: selcon
    double precision :: start_time
    logical :: trace

    handled = .false.
    call ensure_request_state()
    if (.not. request_present) return
    if (.not. env_is_one('MOPAC_MOZYME_SCF_EARLY_PROBE')) return

    trace = mozyme_gpu_scf_profile_enabled()
    call cpu_time(start_time)
    call trace_begin(iw, niter, nocc, nvir, itrmax, selcon)
    if (.not. request_enabled) then
      call trace_end(iw, trace, .false., start_time, &
        scf_failure_message('reason='//trim(request_reason)), .true.)
    else
      call trace_end(iw, trace, .false., start_time, &
        scf_failure_message('reason=early_probe'), .true.)
    end if
    handled = .true.
  end function mozyme_gpu_scf_early_probe

  logical function mozyme_gpu_scf_try(ee, niter, nocc, nvir, itrmax, selcon, &
      fock_mode, idiagg, nhb, density_indi, scf_complete, previous_escf, &
      iemin, iemax, lstart, initial_setup, block_on_failure, &
      final_reorth, final_reorth_done, initial_tidy_done) result(handled)
    use chanel_C, only: iw
    use common_arrays_C, only: coord, ifact, nat, wj => w, wk
    use molkst_C, only: id, keywrd, norbs, numat, nscf, numcal, use_disk, iscf
    use cosmo_C, only: useps, lpka
    use MOZYME_C, only: icocc, icocc_dim, iorbs, kopt, ncf, nncf, &
      ovmax, tiny, sumt, sumb, ijc, pmax, shift, use_three_point_extrap
    use mozyme_diagg1_state, only: mozyme_diagg1_set_state
    use mozyme_diagg2_state, only: mozyme_diagg2_set_state
    use mozyme_isitsc_state, only: mozyme_isitsc_set_state
#ifdef GPU
    use mod_vars_cuda, only: lgpu, mozyme_gpu_requested, &
      mozyme_resident_fock_gpu
    use mozyme_gpu_int_utils, only: mozyme_c_int_checked
    use mozyme_resident_fock, only: mozyme_resident_fock_prepare_plan, &
      mozyme_resident_fock_plan_full_coverage, resident_fock_plan_full, &
      resident_fock_plan_partial
#endif
    implicit none
    double precision, intent(inout) :: ee
    integer, intent(inout) :: niter
    integer, intent(inout) :: idiagg, nhb
    integer, intent(inout) :: iemin, iemax, lstart
    integer, intent(in) :: nocc, nvir, itrmax, fock_mode
    integer, intent(in) :: density_indi
    double precision, intent(in) :: selcon
    double precision, intent(inout) :: previous_escf
    logical, intent(in), optional :: initial_setup, block_on_failure
    logical, intent(in), optional :: final_reorth
    logical, intent(in), optional :: initial_tidy_done
    logical, intent(out), optional :: final_reorth_done
    logical, intent(out) :: scf_complete

    double precision :: start_time
    logical :: trace, full_success, step_success
    logical :: initial_setup_requested, block_failures
    logical :: final_reorth_requested
    logical :: initial_tidy_completed
#ifdef GPU
    logical :: cosmo_state_supported
    integer(c_int) :: code
    integer :: resident_plan_ione
    integer :: resident_fock_plan_id
    integer(c_int) :: resident_fock_required_mask, resident_fock_covered_mask
    logical :: resident_fock_full_covered, resident_fock_partial_covered
    integer :: resident_max_iter
    character(len=64) :: missing_field
    logical :: resident_fock_plan_ready
    type(gpu_mozyme_scf_config) :: config
    type(gpu_mozyme_scf_state) :: state
    type(gpu_mozyme_scf_status) :: status
#endif

    handled = .false.
    scf_complete = .false.
    initial_setup_requested = .false.
    final_reorth_requested = .false.
    initial_tidy_completed = .false.
    block_failures = .true.
    if (present(initial_setup)) initial_setup_requested = initial_setup
    if (present(block_on_failure)) block_failures = block_on_failure
    if (present(final_reorth)) final_reorth_requested = final_reorth
    if (present(initial_tidy_done)) initial_tidy_completed = initial_tidy_done
    if (present(final_reorth_done)) final_reorth_done = .false.
    call ensure_request_state()
    if (.not. request_present) return
    if (blocked_nscf == nscf) return

    trace = mozyme_gpu_scf_profile_enabled()
    call cpu_time(start_time)
    if (trace) then
      call trace_begin(iw, niter, nocc, nvir, itrmax, selcon)
    end if
    if (.not. request_enabled) then
      if (block_failures) blocked_nscf = nscf
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason='//trim(request_reason)), .true.)
      return
    end if

#ifdef GPU
    if (lpka) then
      if (block_failures) blocked_nscf = nscf
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason=solvent_fock'), .true.)
      return
    end if

    if (.not. lgpu .and. .not. mozyme_resident_fock_gpu) then
      if (block_failures) blocked_nscf = nscf
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason=gpu_disabled'), .true.)
      return
    end if
    if (.not. mozyme_gpu_requested) then
      if (block_failures) blocked_nscf = nscf
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason=mozyme_gpu_not_requested'), .true.)
      return
    end if
    if (.not. mozyme_resident_fock_gpu) then
      if (block_failures) blocked_nscf = nscf
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason=resident_fock_disabled'), .true.)
      return
    end if

    if (itrmax <= niter) then
      if (block_failures) blocked_nscf = nscf
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason=iteration_budget_exhausted'), .true.)
      return
    end if

    resident_max_iter = mozyme_denout_resident_max_iter(use_disk, keywrd, &
      niter, itrmax)
    if (resident_max_iter <= niter) then
      if (block_failures) blocked_nscf = nscf
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason=denout_checkpoint'), .true.)
      return
    end if

    call destroy_scf_context()
    call init_config(config, nocc, nvir, resident_max_iter, niter, &
      fock_mode, idiagg, nhb, density_indi, selcon, previous_escf, &
      iemin, iemax, lstart, initial_setup_requested, final_reorth_requested)
    call init_status(status)
    call init_state(state, cosmo_state_supported)

    if (.not. cosmo_state_supported) then
      if (block_failures) blocked_nscf = nscf
      status%code = GPU_MOZYME_SCF_UNSUPPORTED
      call trace_status(iw, trace, status, scf_failure_status())
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason=solvent_fock detail=cosmo_prepare'), &
        .true.)
      return
    end if

    if (.not. state_has_required_pointers(state, missing_field)) then
      if (block_failures) blocked_nscf = nscf
      status%code = GPU_MOZYME_SCF_BAD_ARGUMENT
      call trace_status(iw, trace, status, scf_failure_status())
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason=state_incomplete missing='// &
        trim(missing_field)), .true.)
      return
    end if

    if (.not. state_layout_has_required_sizes(config, state, missing_field)) then
      if (block_failures) blocked_nscf = nscf
      status%code = GPU_MOZYME_SCF_BAD_ARGUMENT
      call trace_status(iw, trace, status, scf_failure_status())
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason=state_layout_invalid missing='// &
        trim(missing_field)), .true.)
      return
    end if

    if (state%use_nijbo /= 1_c_int) then
      if (block_failures) blocked_nscf = nscf
      status%code = GPU_MOZYME_SCF_BAD_ARGUMENT
      call trace_status(iw, trace, status, scf_failure_status())
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason=state_incomplete missing=nijbo'), .true.)
      return
    end if

    resident_plan_ione = 0
    if (id == 0) resident_plan_ione = 1
    resident_fock_plan_id = resident_fock_plan_full
    if (fock_mode /= 0) resident_fock_plan_id = resident_fock_plan_partial
    resident_fock_required_mask = 1_c_int
    if (fock_mode /= 0) resident_fock_required_mask = ior(resident_fock_required_mask, 2_c_int)
    resident_fock_covered_mask = 0_c_int
    resident_fock_full_covered = .false.
    resident_fock_partial_covered = .false.
    if (initial_setup_requested) then
      if (.not. mozyme_gpu_setupk_try(nocc, fock_mode)) then
        if (block_failures) blocked_nscf = nscf
        status%code = GPU_MOZYME_SCF_NOT_READY
        call trace_status(iw, trace, status, scf_failure_status())
        call trace_end(iw, trace, handled, start_time, &
          scf_failure_message('reason=backend_not_ready detail=setupk'), .true.)
        return
      end if
    end if
    if (id == 0) then
      resident_fock_plan_ready = mozyme_resident_fock_prepare_plan( &
        resident_fock_plan_full, iorbs, nat, ifact, wj, wj, 0, kopt, &
        resident_plan_ione, coord, .true.)
    else
      resident_fock_plan_ready = mozyme_resident_fock_prepare_plan( &
        resident_fock_plan_full, iorbs, nat, ifact, wj, wk, 0, kopt, &
        resident_plan_ione, coord, .true.)
    end if
    if (.not. resident_fock_plan_ready) then
      if (block_failures) blocked_nscf = nscf
      status%code = GPU_MOZYME_SCF_NOT_READY
      call trace_status(iw, trace, status, scf_failure_status())
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason=backend_not_ready detail=resident_fock_setup_full'), .true.)
      return
    end if
    resident_fock_full_covered = mozyme_resident_fock_plan_full_coverage(resident_fock_plan_full)
    if (.not. resident_fock_full_covered) then
      if (block_failures) blocked_nscf = nscf
      status%code = GPU_MOZYME_SCF_UNSUPPORTED
      call trace_status(iw, trace, status, scf_failure_status())
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason=resident_fock_partial_coverage detail=full'), .true.)
      return
    end if
    resident_fock_covered_mask = ior(resident_fock_covered_mask, 1_c_int)
    if (fock_mode /= 0) then
      if (id == 0) then
        resident_fock_plan_ready = mozyme_resident_fock_prepare_plan( &
          resident_fock_plan_partial, iorbs, nat, ifact, wj, wj, fock_mode, &
          kopt, resident_plan_ione, coord, .true.)
      else
        resident_fock_plan_ready = mozyme_resident_fock_prepare_plan( &
          resident_fock_plan_partial, iorbs, nat, ifact, wj, wk, fock_mode, &
          kopt, resident_plan_ione, coord, .true.)
      end if
      if (.not. resident_fock_plan_ready) then
        if (block_failures) blocked_nscf = nscf
        status%code = GPU_MOZYME_SCF_NOT_READY
        call trace_status(iw, trace, status, scf_failure_status())
        call trace_end(iw, trace, handled, start_time, &
          scf_failure_message('reason=backend_not_ready detail=resident_fock_setup_partial'), .true.)
        return
      end if
      resident_fock_partial_covered = &
        mozyme_resident_fock_plan_full_coverage(resident_fock_plan_partial)
      if (.not. resident_fock_partial_covered) then
        if (block_failures) blocked_nscf = nscf
        status%code = GPU_MOZYME_SCF_UNSUPPORTED
        call trace_status(iw, trace, status, scf_failure_status())
        call trace_end(iw, trace, handled, start_time, &
          scf_failure_message('reason=resident_fock_partial_coverage detail=partial'), .true.)
        return
      end if
      resident_fock_covered_mask = ior(resident_fock_covered_mask, 2_c_int)
    end if
    config%resident_fock_plan_id = mozyme_c_int_checked(resident_fock_plan_id)
    config%resident_fock_plan_full_coverage = merge(1_c_int, 0_c_int, resident_fock_full_covered)
    config%resident_fock_plan_partial_coverage = &
      merge(1_c_int, 0_c_int, resident_fock_partial_covered)
    config%resident_fock_plan_required_mask = resident_fock_required_mask
    config%resident_fock_plan_covered_mask = resident_fock_covered_mask
    write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', 'resident_fock_plan_id=', resident_fock_plan_id, &
      'resident_fock_plan_full_coverage=', int(config%resident_fock_plan_full_coverage), &
      'resident_fock_plan_partial_coverage=', int(config%resident_fock_plan_partial_coverage), &
      'resident_fock_plan_required_mask=', int(config%resident_fock_plan_required_mask), &
      'resident_fock_plan_covered_mask=', int(config%resident_fock_plan_covered_mask)

    code = mopac_cuda_mozyme_scf_setup(config, scf_context)

    if (code /= GPU_MOZYME_SCF_SUCCESS) then
      status%code = code
    else
      code = mopac_cuda_mozyme_scf_register_state(scf_context, state)
      if (code == GPU_MOZYME_SCF_SUCCESS) then
        code = mopac_cuda_mozyme_scf_status(scf_context, status)
      else
        status%code = code
      end if
    end if

    if (code /= GPU_MOZYME_SCF_SUCCESS .or. status%ready /= 1_c_int) then
      if (block_failures) blocked_nscf = nscf
      call destroy_scf_context()
      call trace_status(iw, trace, status, scf_failure_status())
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason='//trim(stage_code_reason('prepare', &
        status%code))//' stage=prepare code='//trim(int_text(code))// &
        ' status_code='//trim(int_text(status%code))), .true.)
      return
    end if

    call init_status(status)
    code = mopac_cuda_mozyme_scf_run(scf_context, status)
    call normalize_cnvgz_activity(status)
    full_success = backend_completed_scf(code, status, niter, &
      initial_setup_requested, initial_tidy_completed)
    step_success = backend_completed_step(code, status, niter)
    if (step_success .and. .not. full_success .and. &
        mozyme_gpu_scf_no_fallback_required()) then
      step_success = .false.
    end if
    handled = full_success .or. step_success
    scf_complete = full_success

    if (handled) then
      niter = int(status%iterations)
      ee = status%energy_total
      idiagg = int(status%idiagg)
      nhb = int(status%nhb)
      iemin = int(status%isitsc_iemin)
      iemax = int(status%isitsc_iemax)
      previous_escf = status%energy_scf - status%energy_delta
      shift = status%shift
      use_three_point_extrap = status%use_three_point /= 0_c_int
      lstart = int(status%lstart)
      if (present(final_reorth_done)) then
        final_reorth_done = status%final_reorth_applied /= 0_c_int
      end if
      if (status%isitsc_iscf > 0_c_int) iscf = int(status%isitsc_iscf)
      call mozyme_isitsc_set_state(status%isitsc_scf1 /= 0_c_int, &
        status%isitsc_escf0)
      tiny = status%next_tiny
      ovmax = status%diagg_tiny
      sumt = status%diagg_sumt
      sumb = status%diagg_sumb
      ijc = int(status%diagg_nij)
      pmax = status%density_max
      call mozyme_diagg1_set_state(int(numcal), int(status%diagg_nf), &
        status%diagg_fref, status%diagg_oldlim, status%diagg_safety)
      call mozyme_diagg2_set_state(int(status%diagg2_nrejct))
      if (full_success) then
        call trace_status(iw, trace, status, 'status=success')
      else
        call trace_status(iw, trace, status, 'status=resident_step')
      end if
      call destroy_scf_context()
      if (full_success) then
        call trace_end(iw, trace, handled, start_time, &
          'status=success reason=backend_complete code='// &
          trim(int_text(code))//' iterations='// &
          trim(int_text(status%iterations)), .true.)
      else
        call trace_end(iw, trace, handled, start_time, &
          'status=resident_step reason=backend_step_complete code='// &
          trim(int_text(code))//' iterations='// &
          trim(int_text(status%iterations))//' stage_missing='// &
          trim(int_text(status%stage_missing)), .true.)
      end if
    else
      if (block_failures) blocked_nscf = nscf
      call trace_status(iw, trace, status, scf_failure_status())
      call destroy_scf_context()
      call trace_end(iw, trace, handled, start_time, &
        scf_failure_message('reason='//trim(run_fallback_reason(code, &
        status, niter, initial_setup_requested, initial_tidy_completed))// &
        ' stage=run code='//trim(int_text(code))// &
        ' status_code='//trim(int_text(status%code))// &
        ' stage_missing='//trim(int_text(status%stage_missing))), .true.)
    end if
#else
    if (block_failures) blocked_nscf = nscf
      call trace_end(iw, trace, handled, start_time, &
      scf_failure_message('reason=not_gpu_build'), .true.)
#endif
  end function mozyme_gpu_scf_try

#ifdef GPU
  subroutine init_config(config, nocc, nvir, max_iter, current_iter, &
      fock_mode, idiagg, nhb, density_indi, selcon, previous_escf, iemin, &
      iemax, lstart, initial_setup, final_reorth)
    use common_arrays_C, only: nat
    use cosmo_C, only: solv_energy, useps
    use funcon_C, only: fpc_9
    use molkst_C, only: atheat, emin, enuclr, id, keywrd, mpack, norbs, &
      numat, numcal
    use MOZYME_C, only: mode, shift, thresh, tiny, use_three_point_extrap, &
      total_noccupied => noccupied, total_nvirtual => nvirtual
    use mozyme_diagg1_state, only: diagg1_icalcn, diagg1_nf, &
      diagg1_fref, diagg1_oldlim, diagg1_safety
    use mozyme_diagg2_state, only: diagg2_nrejct, mozyme_diagg2_retry
    use mozyme_gpu_int_utils, only: mozyme_c_int_checked, &
      mozyme_c_int_nonnegative_or_zero, mozyme_c_int_positive_or_zero
    use mozyme_isitsc_state, only: isitsc_scf1, isitsc_escf0
    use parameters_C, only: main_group
    implicit none
    type(gpu_mozyme_scf_config), intent(out) :: config
    integer, intent(in) :: nocc, nvir, max_iter, current_iter, fock_mode
    integer, intent(in) :: nhb
    integer, intent(in) :: idiagg, density_indi, iemin, iemax, lstart
    double precision, intent(in) :: selcon, previous_escf
    logical, intent(in) :: initial_setup, final_reorth
    double precision :: eps, eta
    double precision, external :: reada
    integer :: i
    external :: epseta

    config%version = GPU_MOZYME_SCF_ABI_VERSION
    config%natoms = mozyme_c_int_positive_or_zero(numat)
    config%norbs = mozyme_c_int_positive_or_zero(norbs)
    ! MOZYME stores sparse atom-pair blocks, so this is not norbs*(norbs+1)/2.
    config%mpack = mozyme_c_int_positive_or_zero(mpack)
    config%noccupied = mozyme_c_int_nonnegative_or_zero(nocc)
    config%nvirtual = mozyme_c_int_nonnegative_or_zero(nvir)
    config%total_occupied = mozyme_c_int_nonnegative_or_zero(total_noccupied)
    config%total_virtual = mozyme_c_int_nonnegative_or_zero(total_nvirtual)
    config%max_iter = mozyme_c_int_nonnegative_or_zero(max_iter)
    config%current_iter = mozyme_c_int_nonnegative_or_zero(current_iter)
    config%use_three_point = merge(1_c_int, 0_c_int, use_three_point_extrap)
    config%density_mode = mozyme_c_int_checked(mode)
    config%fock_mode = mozyme_c_int_checked(fock_mode)
    config%resident_fock_plan_id = 0_c_int
    config%resident_fock_plan_full_coverage = 0_c_int
    config%resident_fock_plan_partial_coverage = 0_c_int
    config%resident_fock_plan_required_mask = 0_c_int
    config%resident_fock_plan_covered_mask = 0_c_int
    config%flags = 0_c_int
    if (initial_setup) config%flags = ior(config%flags, 1_c_int)
    if (final_reorth) config%flags = ior(config%flags, 2_c_int)
    config%diagg_mode = mozyme_c_int_checked(idiagg)
    config%density_indi = mozyme_c_int_checked(density_indi)
    config%lstart = mozyme_c_int_nonnegative_or_zero(lstart)
    config%shift = shift
    config%thresh = thresh
    config%selcon = selcon
    i = index(keywrd, " DAMP")
    if (i /= 0) then
      config%diagg_rot_const = reada(keywrd, i+5)
    else if (id == 3) then
      config%diagg_rot_const = 0.5d0
    else
      config%diagg_rot_const = 1.0d0
      do i = 1, numat
        if (.not. main_group(nat(i))) config%diagg_rot_const = 0.5d0
      end do
    end if
    call epseta(eps, eta)
    config%diagg_bigeps = 50.0d0 * sqrt(eps)
    config%diagg_retry = merge(1_c_int, 0_c_int, mozyme_diagg2_retry())
    if (diagg1_icalcn /= numcal) then
      config%diagg_fref = 10.0d0
      if (index(keywrd, " OLDENS") /= 0) config%diagg_fref = 0.0d0
      config%diagg_oldlim = 0.0d0
      config%diagg_safety = 1.0d0
      config%diagg_nf = 0_c_int
    else
      config%diagg_fref = diagg1_fref
      config%diagg_oldlim = diagg1_oldlim
      config%diagg_safety = diagg1_safety
      config%diagg_nf = mozyme_c_int_nonnegative_or_zero(diagg1_nf)
    end if
    config%nhb = mozyme_c_int_nonnegative_or_zero(max(0, min(4, nhb)))
    config%addhb_due = merge(1_c_int, 0_c_int, &
      mod(current_iter + 1, 3) == 0 .and. nhb < 4)
    config%diagg2_nrejct(1) = &
      mozyme_c_int_nonnegative_or_zero(diagg2_nrejct(1))
    config%diagg2_nrejct(2) = &
      mozyme_c_int_nonnegative_or_zero(diagg2_nrejct(2))
    config%isitsc_iemin = mozyme_c_int_checked(iemin)
    config%isitsc_iemax = mozyme_c_int_checked(iemax)
    config%isitsc_scf1 = merge(1_c_int, 0_c_int, isitsc_scf1)
    config%energy_scale = fpc_9
    config%energy_offset = enuclr * fpc_9 + atheat
    if (useps) config%energy_offset = config%energy_offset + &
      solv_energy * fpc_9
    config%previous_escf = previous_escf
    config%emin = emin
    config%ovmax = tiny
    config%isitsc_escf0 = isitsc_escf0
  end subroutine init_config

  integer function mozyme_denout_resident_max_iter(use_disk, keywrd, &
      niter, itrmax) result(max_iter)
    implicit none
    logical, intent(in) :: use_disk
    character(len=*), intent(in) :: keywrd
    integer, intent(in) :: niter, itrmax

    integer :: i, idnout, next_checkpoint
    double precision, external :: reada

    max_iter = itrmax
    if (.not. use_disk) return

    i = index(keywrd, " DENOUT=")
    if (i == 0) return

    idnout = nint(reada(keywrd, i + 8))
    if (idnout <= 0) then
      max_iter = niter
      return
    end if

    next_checkpoint = ((max(0, niter) / idnout) + 1) * idnout
    if (next_checkpoint <= niter + 1) then
      max_iter = niter
    else if (next_checkpoint <= itrmax) then
      max_iter = next_checkpoint - 1
    end if
  end function mozyme_denout_resident_max_iter

#ifdef GPU
  logical function mozyme_gpu_setupk_try(nocc, fock_mode) result(ok)
    use chanel_C, only: iw
    use molkst_C, only: numat
    use MOZYME_C, only: icocc, icocc_dim, kopt, ncf, nncf
    use mozyme_gpu_int_utils, only: mozyme_c_int_positive_or_zero
    implicit none
    integer, intent(in) :: nocc
    integer, intent(in) :: fock_mode
    interface
      function mopac_cuda_mozyme_setupk(natoms_c, nocc_c, icocc_dim_c, &
          ncf_c, nncf_c, icocc_c, kopt_c, wall_ms_c) &
          bind(C,name='mopac_cuda_mozyme_setupk') result(code)
        use iso_c_binding, only: c_int, c_double
        integer(c_int), value :: natoms_c, nocc_c, icocc_dim_c
        integer(c_int), intent(in) :: ncf_c(*), nncf_c(*), icocc_c(*)
        integer(c_int) :: kopt_c(*)
        real(c_double) :: wall_ms_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_setupk
    end interface
    integer(c_int) :: code
    real(c_double) :: wall_ms
    real(c_double) :: printed_ms

    ok = .false.
    if (numat <= 0 .or. nocc <= 0 .or. icocc_dim <= 0) then
      write(iw,'(1x,a,1x,a," code=",i0,'// &
        '" initial_setup=1 fock_mode=",i0," all_initial_setup_paths=1")') &
        '[MOZYME GPU setupk]', trim(scf_failure_status()), &
        int(GPU_MOZYME_SCF_BAD_ARGUMENT), &
        fock_mode
      call flush(iw)
      return
    end if
    code = mopac_cuda_mozyme_setupk(mozyme_c_int_positive_or_zero(numat), &
      mozyme_c_int_positive_or_zero(nocc), &
      mozyme_c_int_positive_or_zero(icocc_dim), ncf, nncf, icocc, kopt, &
      wall_ms)
    ok = code == 0_c_int
    if (ok) then
      printed_ms = max(wall_ms, 0.001_c_double)
      write(iw,'(1x,a," success code=",i0," ms=",f10.3,'// &
        '" initial_setup=1 fock_mode=",i0," all_initial_setup_paths=1")') &
        '[MOZYME GPU setupk]', int(code), printed_ms, fock_mode
    else
      write(iw,'(1x,a,1x,a," code=",i0,'// &
        '" initial_setup=1 fock_mode=",i0," all_initial_setup_paths=1")') &
        '[MOZYME GPU setupk]', trim(scf_failure_status()), int(code), fock_mode
    end if
    call flush(iw)
  end function mozyme_gpu_setupk_try
#endif

  subroutine init_state(state, cosmo_state_supported)
    use iso_c_binding, only: c_loc, c_ptr
    use common_arrays_C, only: coord, eigs, f, h, nat, nfirst, nlast, p
    use cosmo_C, only: cosurf, disex2, ediel, fepsi, gden, iatsp, &
      ipiden, nps, phinet, qdenet, qscat, qscnet, solv_energy, srad, &
      useps
    use funcon_C, only: a0, ev
    use iter_C, only: pold
    use linear_cosmo, only: mozyme_cosmo_gpu_state, &
      mozyme_cosmo_prepare_gpu_state
    use MOZYME_C, only: cocc, cocc_dim, cvir, cvir_dim, fmo, icocc, &
      icocc_dim, icvir, icvir_dim, ifmo, idiag, iorbs, kopt, lijbo, &
      ncocc, ncvir, nce, ncf, nfmo, nijbo, nnce, nncf, p1, p2, p3, &
      partf, partp, fmo_dim
    use parameters_C, only: dd, qq, tore
    implicit none
    type(gpu_mozyme_scf_state), intent(out) :: state
    logical, intent(out) :: cosmo_state_supported
    integer(c_int) :: cosmo_npoints_dim, cosmo_a_diag_dim
    integer(c_int) :: cosmo_a_part_dim, cosmo_m_vec_dim
    integer(c_int) :: cosmo_iblock_pos_dim, cosmo_new_surface
    logical :: cosmo_prepare_ok
    type(c_ptr) :: cosmo_npoints_ptr, cosmo_a_diag_ptr
    type(c_ptr) :: cosmo_a_part_ptr, cosmo_a_part_i_ptr
    type(c_ptr) :: cosmo_a_part_j_ptr, cosmo_m_vec_ptr
    type(c_ptr) :: cosmo_iblock_pos_ptr

    cosmo_state_supported = .true.
    state = gpu_mozyme_scf_state()
    state%version = GPU_MOZYME_SCF_ABI_VERSION
    state%flags = 0_c_int
    state%use_nijbo = merge(1_c_int, 0_c_int, lijbo .and. allocated(nijbo))
    state%icocc_dim = c_int_or_zero(icocc_dim)
    state%cocc_dim = c_int_or_zero(cocc_dim)
    state%icvir_dim = c_int_or_zero(icvir_dim)
    state%cvir_dim = c_int_or_zero(cvir_dim)
    state%fmo_dim = c_int_or_zero(fmo_dim)
    state%partp_dim = c_int_or_zero(size_or_zero_real(partp))
    state%partf_dim = c_int_or_zero(size_or_zero_real(partf))
    state%nocc_slots = c_int_or_zero(min_size3_int(ncf, nncf, ncocc))
    state%nvir_slots = c_int_or_zero(min_size3_int(nce, nnce, ncvir))
    state%p_dim = c_int_or_zero(size_or_zero_real(p))
    state%f_dim = c_int_or_zero(size_or_zero_real(f))
    state%h_dim = c_int_or_zero(size_or_zero_real(h))
    state%pold_dim = c_int_or_zero(size_or_zero_real(pold))
    state%p1_dim = c_int_or_zero(size_or_zero_real(p1))
    state%p2_dim = c_int_or_zero(size_or_zero_real(p2))
    state%p3_dim = c_int_or_zero(size_or_zero_real(p3))
    state%idiag_dim = c_int_or_zero(size_or_zero_int(idiag))
    state%iorbs_dim = c_int_or_zero(size_or_zero_int(iorbs))
    state%kopt_dim = c_int_or_zero(size_or_zero_int(kopt))
    state%ncf_dim = c_int_or_zero(size_or_zero_int(ncf))
    state%nncf_dim = c_int_or_zero(size_or_zero_int(nncf))
    state%ncocc_dim = c_int_or_zero(size_or_zero_int(ncocc))
    state%nce_dim = c_int_or_zero(size_or_zero_int(nce))
    state%nnce_dim = c_int_or_zero(size_or_zero_int(nnce))
    state%ncvir_dim = c_int_or_zero(size_or_zero_int(ncvir))
    state%ifmo_rows = c_int_or_zero(size1_or_zero_int_2d(ifmo))
    state%ifmo_cols = c_int_or_zero(size2_or_zero_int_2d(ifmo))
    state%eigs_dim = c_int_or_zero(size_or_zero_real(eigs))
    state%nfmo_dim = c_int_or_zero(size_or_zero_int(nfmo))
    state%nfirst_dim = c_int_or_zero(size_or_zero_c_int(nfirst))
    state%nlast_dim = c_int_or_zero(size_or_zero_c_int(nlast))
    state%nijbo_rows = c_int_or_zero(size1_or_zero_int_2d(nijbo))
    state%nijbo_cols = c_int_or_zero(size2_or_zero_int_2d(nijbo))
    state%coord_rows = c_int_or_zero(size1_or_zero_real_2d(coord))
    state%coord_cols = c_int_or_zero(size2_or_zero_real_2d(coord))
    state%nat_dim = c_int_or_zero(size_or_zero_int(nat))
    state%p = ptr_or_null_real(p)
    state%f = ptr_or_null_real(f)
    state%h = ptr_or_null_real(h)
    state%partp = ptr_or_null_real(partp)
    state%partf = ptr_or_null_real(partf)
    state%pold = ptr_or_null_real(pold)
    state%p1 = ptr_or_null_real(p1)
    state%p2 = ptr_or_null_real(p2)
    state%p3 = ptr_or_null_real(p3)
    state%idiag = ptr_or_null_int(idiag)
    state%iorbs = ptr_or_null_int(iorbs)
    state%kopt = ptr_or_null_int(kopt)
    state%ncf = ptr_or_null_int(ncf)
    state%nncf = ptr_or_null_int(nncf)
    state%ncocc = ptr_or_null_int(ncocc)
    state%icocc = ptr_or_null_int(icocc)
    state%cocc = ptr_or_null_real(cocc)
    state%nce = ptr_or_null_int(nce)
    state%nnce = ptr_or_null_int(nnce)
    state%ncvir = ptr_or_null_int(ncvir)
    state%icvir = ptr_or_null_int(icvir)
    state%cvir = ptr_or_null_real(cvir)
    state%fmo = ptr_or_null_real(fmo)
    state%ifmo = ptr_or_null_int_2d(ifmo)
    state%eigs = ptr_or_null_real(eigs)
    state%nfmo = ptr_or_null_int(nfmo)
    state%nfirst = ptr_or_null_c_int(nfirst)
    state%nlast = ptr_or_null_c_int(nlast)
    state%nijbo = ptr_or_null_int_2d(nijbo)
    state%coord = ptr_or_null_real_2d(coord)
    state%nat = ptr_or_null_int(nat)
    state%param_dim = 107_c_int
    state%param_dd = c_loc(dd(1))
    state%param_qq = c_loc(qq(1))
    state%param_tore = c_loc(tore(1))
    if (useps) then
      call mozyme_cosmo_prepare_gpu_state(cosmo_prepare_ok)
      if (.not. cosmo_prepare_ok) then
        cosmo_state_supported = .false.
        state%cosmo_enabled = 0_c_int
        return
      end if
      call mozyme_cosmo_gpu_state(cosmo_npoints_dim, cosmo_a_diag_dim, &
        cosmo_a_part_dim, cosmo_m_vec_dim, cosmo_iblock_pos_dim, &
        cosmo_new_surface, cosmo_npoints_ptr, cosmo_a_diag_ptr, &
        cosmo_a_part_ptr, cosmo_a_part_i_ptr, cosmo_a_part_j_ptr, &
        cosmo_m_vec_ptr, cosmo_iblock_pos_ptr)
      state%cosmo_enabled = 1_c_int
      state%cosmo_nps = c_int_or_zero(nps)
      state%cosmo_lm61 = c_int_or_zero(size_or_zero_real(gden))
      state%cosmo_cosurf_rows = &
        c_int_or_zero(size1_or_zero_real_2d(cosurf))
      state%cosmo_cosurf_cols = &
        c_int_or_zero(size2_or_zero_real_2d(cosurf))
      state%cosmo_phinet_rows = &
        c_int_or_zero(size1_or_zero_real_2d(phinet))
      state%cosmo_phinet_cols = &
        c_int_or_zero(size2_or_zero_real_2d(phinet))
      state%cosmo_qscnet_rows = &
        c_int_or_zero(size1_or_zero_real_2d(qscnet))
      state%cosmo_qscnet_cols = &
        c_int_or_zero(size2_or_zero_real_2d(qscnet))
      state%cosmo_qdenet_rows = &
        c_int_or_zero(size1_or_zero_real_2d(qdenet))
      state%cosmo_qdenet_cols = &
        c_int_or_zero(size2_or_zero_real_2d(qdenet))
      state%cosmo_qscat_dim = c_int_or_zero(size_or_zero_real(qscat))
      state%cosmo_srad_dim = c_int_or_zero(size_or_zero_real(srad))
      state%cosmo_npoints_dim = cosmo_npoints_dim
      state%cosmo_a_diag_dim = cosmo_a_diag_dim
      state%cosmo_a_part_dim = cosmo_a_part_dim
      state%cosmo_m_vec_dim = cosmo_m_vec_dim
      state%cosmo_iblock_pos_dim = cosmo_iblock_pos_dim
      state%cosmo_new_surface = cosmo_new_surface
      if (state%cosmo_nps <= 0_c_int .or. &
          state%cosmo_lm61 <= 0_c_int .or. &
          state%cosmo_npoints_dim <= 0_c_int .or. &
          state%cosmo_a_diag_dim <= 0_c_int .or. &
          state%cosmo_m_vec_dim <= 0_c_int .or. &
          state%cosmo_iblock_pos_dim <= 0_c_int) then
        cosmo_state_supported = .false.
        state%cosmo_enabled = 0_c_int
        return
      end if
      state%cosmo_fepsi = fepsi
      state%cosmo_disex2 = disex2
      state%cosmo_solv_energy = solv_energy
      state%cosmo_ediel = ediel
      state%cosmo_a0 = a0
      state%cosmo_ev = ev
      state%cosmo_iatsp = ptr_or_null_int(iatsp)
      state%cosmo_ipiden = ptr_or_null_int(ipiden)
      state%cosmo_gden = ptr_or_null_real(gden)
      state%cosmo_qscat = ptr_or_null_real(qscat)
      state%cosmo_srad = ptr_or_null_real(srad)
      state%cosmo_cosurf = ptr_or_null_real_2d(cosurf)
      state%cosmo_phinet = ptr_or_null_real_2d(phinet)
      state%cosmo_qscnet = ptr_or_null_real_2d(qscnet)
      state%cosmo_qdenet = ptr_or_null_real_2d(qdenet)
      state%cosmo_npoints = cosmo_npoints_ptr
      state%cosmo_a_diag = cosmo_a_diag_ptr
      state%cosmo_a_part = cosmo_a_part_ptr
      state%cosmo_a_part_i = cosmo_a_part_i_ptr
      state%cosmo_a_part_j = cosmo_a_part_j_ptr
      state%cosmo_m_vec = cosmo_m_vec_ptr
      state%cosmo_iblock_pos = cosmo_iblock_pos_ptr
      state%cosmo_solv_energy_ptr = c_loc(solv_energy)
      state%cosmo_ediel_ptr = c_loc(ediel)
    end if
  end subroutine init_state

  integer(c_int) function c_int_or_zero(value) result(converted)
    use mozyme_gpu_int_utils, only: mozyme_c_int_positive_or_zero
    implicit none
    integer, intent(in) :: value

    converted = mozyme_c_int_positive_or_zero(value)
  end function c_int_or_zero

  integer function size_or_zero_real(values) result(count)
    implicit none
    double precision, allocatable, intent(in) :: values(:)

    count = 0
    if (allocated(values)) count = size(values)
  end function size_or_zero_real

  integer function size_or_zero_int(values) result(count)
    implicit none
    integer, allocatable, intent(in) :: values(:)

    count = 0
    if (allocated(values)) count = size(values)
  end function size_or_zero_int

  integer function size_or_zero_c_int(values) result(count)
    implicit none
    integer(c_int), allocatable, intent(in) :: values(:)

    count = 0
    if (allocated(values)) count = size(values)
  end function size_or_zero_c_int

  integer function size1_or_zero_int_2d(values) result(count)
    implicit none
    integer, allocatable, intent(in) :: values(:,:)

    count = 0
    if (allocated(values)) count = size(values, 1)
  end function size1_or_zero_int_2d

  integer function size2_or_zero_int_2d(values) result(count)
    implicit none
    integer, allocatable, intent(in) :: values(:,:)

    count = 0
    if (allocated(values)) count = size(values, 2)
  end function size2_or_zero_int_2d

  integer function size1_or_zero_real_2d(values) result(count)
    implicit none
    double precision, allocatable, intent(in) :: values(:,:)

    count = 0
    if (allocated(values)) count = size(values, 1)
  end function size1_or_zero_real_2d

  integer function size2_or_zero_real_2d(values) result(count)
    implicit none
    double precision, allocatable, intent(in) :: values(:,:)

    count = 0
    if (allocated(values)) count = size(values, 2)
  end function size2_or_zero_real_2d

  integer function min_size3_int(values1, values2, values3) result(count)
    implicit none
    integer, allocatable, intent(in) :: values1(:)
    integer, allocatable, intent(in) :: values2(:)
    integer, allocatable, intent(in) :: values3(:)

    count = 0
    if (allocated(values1) .and. allocated(values2) .and. &
        allocated(values3)) then
      count = min(size(values1), size(values2), size(values3))
    end if
  end function min_size3_int

  type(c_ptr) function ptr_or_null_real(values) result(ptr)
    use iso_c_binding, only: c_loc
    implicit none
    double precision, allocatable, target, intent(inout) :: values(:)

    ptr = c_null_ptr
    if (allocated(values)) then
      if (size(values) > 0) ptr = c_loc(values(1))
    end if
  end function ptr_or_null_real

  type(c_ptr) function ptr_or_null_int(values) result(ptr)
    use iso_c_binding, only: c_loc
    implicit none
    integer, allocatable, target, intent(inout) :: values(:)

    ptr = c_null_ptr
    if (allocated(values)) then
      if (size(values) > 0) ptr = c_loc(values(1))
    end if
  end function ptr_or_null_int

  type(c_ptr) function ptr_or_null_c_int(values) result(ptr)
    use iso_c_binding, only: c_loc
    implicit none
    integer(c_int), allocatable, target, intent(inout) :: values(:)

    ptr = c_null_ptr
    if (allocated(values)) then
      if (size(values) > 0) ptr = c_loc(values(1))
    end if
  end function ptr_or_null_c_int

  type(c_ptr) function ptr_or_null_int_2d(values) result(ptr)
    use iso_c_binding, only: c_loc
    implicit none
    integer, allocatable, target, intent(inout) :: values(:,:)

    ptr = c_null_ptr
    if (allocated(values)) then
      if (size(values) > 0) ptr = c_loc(values(1,1))
    end if
  end function ptr_or_null_int_2d

  type(c_ptr) function ptr_or_null_real_2d(values) result(ptr)
    use iso_c_binding, only: c_loc
    implicit none
    double precision, allocatable, target, intent(inout) :: values(:,:)

    ptr = c_null_ptr
    if (allocated(values)) then
      if (size(values) > 0) ptr = c_loc(values(1,1))
    end if
  end function ptr_or_null_real_2d

  logical function state_has_required_pointers(state, missing_field) &
      result(ready)
    implicit none
    type(gpu_mozyme_scf_state), intent(in) :: state
    character(len=*), intent(out) :: missing_field

    ready = .false.
    missing_field = 'none'

    if (.not. c_associated(state%p)) then
      missing_field = 'p'
    else if (.not. c_associated(state%f)) then
      missing_field = 'f'
    else if (.not. c_associated(state%h)) then
      missing_field = 'h'
    else if (.not. c_associated(state%partp)) then
      missing_field = 'partp'
    else if (state%partp_dim <= 0_c_int) then
      missing_field = 'partp_dim'
    else if (.not. c_associated(state%partf)) then
      missing_field = 'partf'
    else if (state%partf_dim <= 0_c_int) then
      missing_field = 'partf_dim'
    else if (.not. c_associated(state%pold)) then
      missing_field = 'pold'
    else if (.not. c_associated(state%p1)) then
      missing_field = 'p1'
    else if (.not. c_associated(state%p2)) then
      missing_field = 'p2'
    else if (.not. c_associated(state%p3)) then
      missing_field = 'p3'
    else if (.not. c_associated(state%idiag)) then
      missing_field = 'idiag'
    else if (.not. c_associated(state%iorbs)) then
      missing_field = 'iorbs'
    else if (.not. c_associated(state%kopt)) then
      missing_field = 'kopt'
    else if (.not. c_associated(state%ncf)) then
      missing_field = 'ncf'
    else if (.not. c_associated(state%nncf)) then
      missing_field = 'nncf'
    else if (.not. c_associated(state%ncocc)) then
      missing_field = 'ncocc'
    else if (state%nocc_slots <= 0_c_int) then
      missing_field = 'nocc_slots'
    else if (.not. c_associated(state%icocc)) then
      missing_field = 'icocc'
    else if (.not. c_associated(state%cocc)) then
      missing_field = 'cocc'
    else if (.not. c_associated(state%nce)) then
      missing_field = 'nce'
    else if (.not. c_associated(state%nnce)) then
      missing_field = 'nnce'
    else if (.not. c_associated(state%ncvir)) then
      missing_field = 'ncvir'
    else if (state%nvir_slots <= 0_c_int) then
      missing_field = 'nvir_slots'
    else if (.not. c_associated(state%icvir)) then
      missing_field = 'icvir'
    else if (.not. c_associated(state%cvir)) then
      missing_field = 'cvir'
    else if (state%fmo_dim <= 0_c_int) then
      missing_field = 'fmo_dim'
    else if (.not. c_associated(state%fmo)) then
      missing_field = 'fmo'
    else if (.not. c_associated(state%ifmo)) then
      missing_field = 'ifmo'
    else if (.not. c_associated(state%eigs)) then
      missing_field = 'eigs'
    else if (.not. c_associated(state%nfmo)) then
      missing_field = 'nfmo'
    else if (.not. c_associated(state%nfirst)) then
      missing_field = 'nfirst'
    else if (.not. c_associated(state%nlast)) then
      missing_field = 'nlast'
    else if (state%use_nijbo == 1_c_int .and. &
        .not. c_associated(state%nijbo)) then
      missing_field = 'nijbo'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%coord)) then
      missing_field = 'coord'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%nat)) then
      missing_field = 'nat'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%param_dd)) then
      missing_field = 'param_dd'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%param_qq)) then
      missing_field = 'param_qq'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%param_tore)) then
      missing_field = 'param_tore'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_iatsp)) then
      missing_field = 'cosmo_iatsp'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_ipiden)) then
      missing_field = 'cosmo_ipiden'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_gden)) then
      missing_field = 'cosmo_gden'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_qscat)) then
      missing_field = 'cosmo_qscat'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_srad)) then
      missing_field = 'cosmo_srad'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_cosurf)) then
      missing_field = 'cosmo_cosurf'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_phinet)) then
      missing_field = 'cosmo_phinet'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_qscnet)) then
      missing_field = 'cosmo_qscnet'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_qdenet)) then
      missing_field = 'cosmo_qdenet'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_npoints)) then
      missing_field = 'cosmo_npoints'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_a_diag)) then
      missing_field = 'cosmo_a_diag'
    else if (state%cosmo_enabled == 1_c_int .and. &
        state%cosmo_a_part_dim > 0_c_int .and. &
        .not. c_associated(state%cosmo_a_part)) then
      missing_field = 'cosmo_a_part'
    else if (state%cosmo_enabled == 1_c_int .and. &
        state%cosmo_a_part_dim > 0_c_int .and. &
        .not. c_associated(state%cosmo_a_part_i)) then
      missing_field = 'cosmo_a_part_i'
    else if (state%cosmo_enabled == 1_c_int .and. &
        state%cosmo_a_part_dim > 0_c_int .and. &
        .not. c_associated(state%cosmo_a_part_j)) then
      missing_field = 'cosmo_a_part_j'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_m_vec)) then
      missing_field = 'cosmo_m_vec'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_iblock_pos)) then
      missing_field = 'cosmo_iblock_pos'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_solv_energy_ptr)) then
      missing_field = 'cosmo_solv_energy'
    else if (state%cosmo_enabled == 1_c_int .and. &
        .not. c_associated(state%cosmo_ediel_ptr)) then
      missing_field = 'cosmo_ediel'
    else
      ready = .true.
    end if
  end function state_has_required_pointers

  logical function state_layout_has_required_sizes(config, state, &
      missing_field) result(ready)
    use common_arrays_C, only: coord, eigs, f, h, nat, nfirst, nlast, p
    use cosmo_C, only: cosurf, gden, iatsp, ipiden, phinet, qdenet, &
      qscat, qscnet, srad
    use iter_C, only: pold
    use molkst_C, only: mpack, norbs, numat
    use MOZYME_C, only: cocc, cvir, fmo, fmo_dim, icocc, icvir, ifmo, idiag, &
      iorbs, kopt, ncocc, ncvir, nce, ncf, nfmo, nijbo, nnce, nncf, &
      p1, p2, p3, partf, partp
    implicit none
    type(gpu_mozyme_scf_config), intent(in) :: config
    type(gpu_mozyme_scf_state), intent(in) :: state
    character(len=*), intent(out) :: missing_field

    ! resident_scf_state_layout_preflight keeps C pointers tied to allocations.
    ready = .false.
    missing_field = 'none'

    if (config%mpack /= c_int_or_zero(mpack)) then
      missing_field = 'config_mpack'
    else if (config%norbs /= c_int_or_zero(norbs)) then
      missing_field = 'config_norbs'
    else if (config%natoms /= c_int_or_zero(numat)) then
      missing_field = 'config_natoms'
    else if (.not. (storage_size(0) == storage_size(0_c_int))) then
      missing_field = 'integer_kind'
    else if (.not. (storage_size(0.0d0) == storage_size(0.0_c_double))) then
      missing_field = 'real_kind'
    else if (.not. allocated(p)) then
      missing_field = 'p'
    else if (.not. (size(p) >= mpack)) then
      missing_field = 'p_dim'
    else if (.not. allocated(f)) then
      missing_field = 'f'
    else if (.not. (size(f) >= mpack)) then
      missing_field = 'f_dim'
    else if (.not. allocated(h)) then
      missing_field = 'h'
    else if (.not. (size(h) >= mpack)) then
      missing_field = 'h_dim'
    else if (.not. allocated(pold)) then
      missing_field = 'pold'
    else if (.not. (size(pold) >= mpack)) then
      missing_field = 'pold_dim'
    else if (.not. allocated(partp)) then
      missing_field = 'partp'
    else if (size(partp) < 1) then
      missing_field = 'partp_dim'
    else if ((config%fock_mode /= 0_c_int .or. &
        config%density_indi /= 0_c_int) .and. &
        .not. (size(partp) >= mpack)) then
      missing_field = 'partp_dim'
    else if (.not. allocated(partf)) then
      missing_field = 'partf'
    else if (size(partf) < 1) then
      missing_field = 'partf_dim'
    else if ((config%fock_mode /= 0_c_int .or. &
        config%density_indi /= 0_c_int) .and. &
        .not. (size(partf) >= mpack)) then
      missing_field = 'partf_dim'
    else if (.not. allocated(p1)) then
      missing_field = 'p1'
    else if (.not. (size(p1) >= norbs)) then
      missing_field = 'p1_dim'
    else if (.not. allocated(p2)) then
      missing_field = 'p2'
    else if (.not. (size(p2) >= norbs)) then
      missing_field = 'p2_dim'
    else if (.not. allocated(p3)) then
      missing_field = 'p3'
    else if (.not. (size(p3) >= norbs)) then
      missing_field = 'p3_dim'
    else if (.not. allocated(idiag)) then
      missing_field = 'idiag'
    else if (.not. (size(idiag) >= norbs)) then
      missing_field = 'idiag_dim'
    else if (.not. allocated(eigs)) then
      missing_field = 'eigs'
    else if (.not. (size(eigs) >= norbs)) then
      missing_field = 'eigs_dim'
    else if (.not. allocated(nfmo)) then
      missing_field = 'nfmo'
    else if (.not. (size(nfmo) >= norbs)) then
      missing_field = 'nfmo_dim'
    else if (.not. allocated(iorbs)) then
      missing_field = 'iorbs'
    else if (.not. (size(iorbs) >= numat)) then
      missing_field = 'iorbs_dim'
    else if (.not. allocated(kopt)) then
      missing_field = 'kopt'
    else if (.not. (size(kopt) >= numat)) then
      missing_field = 'kopt_dim'
    else if (.not. allocated(nfirst)) then
      missing_field = 'nfirst'
    else if (.not. (size(nfirst) >= numat)) then
      missing_field = 'nfirst_dim'
    else if (.not. allocated(nlast)) then
      missing_field = 'nlast'
    else if (.not. (size(nlast) >= numat)) then
      missing_field = 'nlast_dim'
    else if (state%nocc_slots < config%total_occupied + 1_c_int) then
      missing_field = 'nocc_slots'
    else if (state%nvir_slots < config%total_virtual + 1_c_int) then
      missing_field = 'nvir_slots'
    else if (.not. allocated(ncf)) then
      missing_field = 'ncf'
    else if (size(ncf) < int(state%nocc_slots)) then
      missing_field = 'ncf_dim'
    else if (.not. allocated(nncf)) then
      missing_field = 'nncf'
    else if (size(nncf) < int(state%nocc_slots)) then
      missing_field = 'nncf_dim'
    else if (.not. allocated(ncocc)) then
      missing_field = 'ncocc'
    else if (size(ncocc) < int(state%nocc_slots)) then
      missing_field = 'ncocc_dim'
    else if (.not. allocated(nce)) then
      missing_field = 'nce'
    else if (size(nce) < int(state%nvir_slots)) then
      missing_field = 'nce_dim'
    else if (.not. allocated(nnce)) then
      missing_field = 'nnce'
    else if (size(nnce) < int(state%nvir_slots)) then
      missing_field = 'nnce_dim'
    else if (.not. allocated(ncvir)) then
      missing_field = 'ncvir'
    else if (size(ncvir) < int(state%nvir_slots)) then
      missing_field = 'ncvir_dim'
    else if (.not. allocated(icocc)) then
      missing_field = 'icocc'
    else if (state%icocc_dim <= 0_c_int .or. &
        size(icocc) < int(state%icocc_dim)) then
      missing_field = 'icocc_dim'
    else if (.not. allocated(cocc)) then
      missing_field = 'cocc'
    else if (state%cocc_dim <= 0_c_int .or. &
        size(cocc) < int(state%cocc_dim)) then
      missing_field = 'cocc_dim'
    else if (.not. allocated(icvir)) then
      missing_field = 'icvir'
    else if (state%icvir_dim <= 0_c_int .or. &
        size(icvir) < int(state%icvir_dim)) then
      missing_field = 'icvir_dim'
    else if (.not. allocated(cvir)) then
      missing_field = 'cvir'
    else if (state%cvir_dim <= 0_c_int .or. &
        size(cvir) < int(state%cvir_dim)) then
      missing_field = 'cvir_dim'
    else if (.not. allocated(fmo)) then
      missing_field = 'fmo'
    else if (state%fmo_dim <= 0_c_int .or. &
        size(fmo) < int(state%fmo_dim)) then
      missing_field = 'fmo_dim'
    else if (.not. allocated(ifmo)) then
      missing_field = 'ifmo'
    else if (.not. (size(ifmo, 1) == 2)) then
      missing_field = 'ifmo_rows'
    else if (.not. (size(ifmo, 2) >= fmo_dim)) then
      missing_field = 'ifmo_cols'
    else
      ready = .true.
    end if

    if (ready .and. state%use_nijbo == 1_c_int) then
      if (.not. allocated(nijbo)) then
        ready = .false.
        missing_field = 'nijbo'
      else if (.not. (size(nijbo, 1) == numat)) then
        ready = .false.
        missing_field = 'nijbo_rows'
      else if (.not. (size(nijbo, 2) >= numat)) then
        ready = .false.
        missing_field = 'nijbo_cols'
      end if
    end if
    if (ready .and. state%cosmo_enabled == 1_c_int) then
      if (.not. allocated(coord)) then
        ready = .false.
        missing_field = 'coord'
      else if (size(coord, 1) < 3 .or. size(coord, 2) < numat) then
        ready = .false.
        missing_field = 'coord_dim'
      else if (.not. allocated(nat)) then
        ready = .false.
        missing_field = 'nat'
      else if (size(nat) < numat) then
        ready = .false.
        missing_field = 'nat_dim'
      else if (.not. allocated(cosurf)) then
        ready = .false.
        missing_field = 'cosmo_cosurf'
      else if (size(cosurf, 1) < 4 .or. &
          size(cosurf, 2) < int(state%cosmo_nps)) then
        ready = .false.
        missing_field = 'cosmo_cosurf_dim'
      else if (.not. allocated(phinet)) then
        ready = .false.
        missing_field = 'cosmo_phinet'
      else if (size(phinet, 1) < int(state%cosmo_nps) .or. &
          size(phinet, 2) < 3) then
        ready = .false.
        missing_field = 'cosmo_phinet_dim'
      else if (.not. allocated(qscnet)) then
        ready = .false.
        missing_field = 'cosmo_qscnet'
      else if (size(qscnet, 1) < int(state%cosmo_nps) .or. &
          size(qscnet, 2) < 3) then
        ready = .false.
        missing_field = 'cosmo_qscnet_dim'
      else if (.not. allocated(qdenet)) then
        ready = .false.
        missing_field = 'cosmo_qdenet'
      else if (size(qdenet, 1) < int(state%cosmo_lm61) .or. &
          size(qdenet, 2) < 3) then
        ready = .false.
        missing_field = 'cosmo_qdenet_dim'
      else if (.not. allocated(iatsp) .or. &
          size(iatsp) < int(state%cosmo_nps)) then
        ready = .false.
        missing_field = 'cosmo_iatsp_dim'
      else if (.not. allocated(ipiden) .or. &
          size(ipiden) < int(state%cosmo_lm61)) then
        ready = .false.
        missing_field = 'cosmo_ipiden_dim'
      else if (.not. allocated(gden) .or. &
          size(gden) < int(state%cosmo_lm61)) then
        ready = .false.
        missing_field = 'cosmo_gden_dim'
      else if (.not. allocated(qscat) .or. size(qscat) < numat) then
        ready = .false.
        missing_field = 'cosmo_qscat_dim'
      else if (.not. allocated(srad) .or. size(srad) < numat) then
        ready = .false.
        missing_field = 'cosmo_srad_dim'
      else if (state%cosmo_npoints_dim < c_int_or_zero(numat + 1)) then
        ready = .false.
        missing_field = 'cosmo_npoints_dim'
      else if (state%cosmo_a_diag_dim < state%cosmo_nps) then
        ready = .false.
        missing_field = 'cosmo_a_diag_dim'
      else if (state%cosmo_m_vec_dim <= 0_c_int) then
        ready = .false.
        missing_field = 'cosmo_m_vec_dim'
      else if (state%cosmo_iblock_pos_dim < c_int_or_zero(numat)) then
        ready = .false.
        missing_field = 'cosmo_iblock_pos_dim'
      end if
    end if
  end function state_layout_has_required_sizes

  subroutine destroy_scf_context()
    implicit none
    integer(c_int) :: destroy_code

    if (c_associated(scf_context)) then
      destroy_code = mopac_cuda_mozyme_scf_destroy(scf_context)
      scf_context = c_null_ptr
    end if
  end subroutine destroy_scf_context

  subroutine init_status(status)
    implicit none
    type(gpu_mozyme_scf_status), intent(out) :: status

    status%version = GPU_MOZYME_SCF_ABI_VERSION
    status%code = GPU_MOZYME_SCF_NOT_READY
    status%ready = 0_c_int
    status%resident = 0_c_int
    status%device_id = -1_c_int
    status%natoms = 0_c_int
    status%norbs = 0_c_int
    status%mpack = 0_c_int
    status%iterations = 0_c_int
    status%stage_completed = 0_c_int
    status%stage_required = GPU_MOZYME_SCF_STAGE_FULL
    status%stage_missing = GPU_MOZYME_SCF_STAGE_FULL
    status%resident_decision = 0_c_int
    status%resident_fock_plan_id = 0_c_int
    status%resident_fock_plan_full_coverage = 0_c_int
    status%resident_fock_plan_partial_coverage = 0_c_int
    status%resident_fock_plan_required_mask = 0_c_int
    status%resident_fock_plan_covered_mask = 0_c_int
    status%final_publication_done = 0_c_int
    status%final_publication_arrays = 0_c_int
    status%final_publication_bytes = 0_c_size_t
    status%final_publication_cosmo = 0_c_int
    status%energy_total = 0.0d0
    status%energy_delta = 0.0d0
    status%density_max = 0.0d0
    status%density_rms = 0.0d0
    status%wall_ms = 0.0d0
    status%diagg_nij = 0_c_int
    status%diagg_nf = 0_c_int
    status%idiagg = 0_c_int
    status%nhb = 0_c_int
    status%addhb_due = 0_c_int
    status%addhb_applied = 0_c_int
    status%addhb_nij = 0_c_int
    status%diagg2_nrejct = 0_c_int
    status%diagg_tiny = 0.0d0
    status%next_tiny = 0.0d0
    status%diagg_fref = 0.0d0
    status%diagg_oldlim = 0.0d0
    status%diagg_safety = 0.0d0
    status%diagg_sumt = 0.0d0
    status%diagg_sumb = 0.0d0
    status%energy_scf = 0.0d0
    status%isitsc_okscf = 0_c_int
    status%isitsc_iscf = 0_c_int
    status%isitsc_iemin = 0_c_int
    status%isitsc_iemax = 0_c_int
    status%isitsc_scf1 = 0_c_int
    status%use_three_point = 0_c_int
    status%lstart = 0_c_int
    status%shift = 0.0d0
    status%isitsc_escf0 = 0.0d0
    status%resident_stage_calls = 0_c_int
    status%resident_stage_ms = 0.0d0
    status%final_reorth_applied = 0_c_int
    status%final_reorth_ms = 0.0d0
    status%final_reorth_sum = 0.0d0
    status%pls_supervisor_calls = 0_c_int
    status%pls_restart_required = 0_c_int
    status%pls_history_count = 0_c_int
    status%pls_ovmax_delta = 0.0d0
    status%pls_energy_delta = 0.0d0
    status%pls_restart_reset_device_calls = 0_c_int
    status%pls_restart_done = 0_c_int
    status%cosmo_enabled = 0_c_int
    status%cosmo_fock_calls = 0_c_int
    status%cosmo_matvec_calls = 0_c_int
    status%cosmo_cg_iterations = 0_c_int
    status%cosmo_nps = 0_c_int
    status%cosmo_lm61 = 0_c_int
    status%cosmo_pair_count = 0_c_int
    status%cosmo_solv_energy = 0.0d0
    status%cosmo_ediel = 0.0d0
    status%cosmo_last_residual = 0.0d0
    status%cosmo_cg_control_resident = 0_c_int
    status%cosmo_cg_converged = 0_c_int
    status%cosmo_cg_breakdown = 0_c_int
    status%cosmo_cg_host_syncs = 0_c_int
    status%cosmo_cg_target_tol = 0.0d0
    status%cnvgz_active_calls = 0_c_int
    status%cnvgz_noop_calls = 0_c_int
    status%strict_resident_host_syncs = 0_c_int
    status%strict_resident_control_polls = 0_c_int
  end subroutine init_status

  subroutine normalize_cnvgz_activity(status)
    implicit none
    type(gpu_mozyme_scf_status), intent(inout) :: status

    if (status%cnvgz_active_calls + status%cnvgz_noop_calls <= 0_c_int .and. &
        status%resident_stage_calls(6) > 0_c_int) then
      status%cnvgz_noop_calls = status%resident_stage_calls(6)
    end if
  end subroutine normalize_cnvgz_activity

  logical function backend_pls_resolved(status) result(resolved)
    implicit none
    type(gpu_mozyme_scf_status), intent(in) :: status

    resolved = status%pls_restart_required == 0_c_int .and. &
      (status%pls_restart_reset_device_calls == 0_c_int .or. &
       status%pls_restart_done == 1_c_int)
  end function backend_pls_resolved

  logical function backend_resident_fock_plans_complete(status) result(complete)
    implicit none
    type(gpu_mozyme_scf_status), intent(in) :: status

    complete = status%resident_fock_plan_required_mask /= 0_c_int .and. &
      status%resident_fock_plan_full_coverage == 1_c_int .and. &
      status%resident_fock_plan_covered_mask == &
      status%resident_fock_plan_required_mask .and. &
      (iand(status%resident_fock_plan_required_mask, 2_c_int) == 0_c_int .or. &
       status%resident_fock_plan_partial_coverage == 1_c_int)
  end function backend_resident_fock_plans_complete

  logical function backend_final_publication_complete(status) result(complete)
    implicit none
    type(gpu_mozyme_scf_status), intent(in) :: status

    if (.not. mozyme_gpu_scf_strict_resident()) then
      complete = .true.
      return
    end if
    complete = status%final_publication_done == 1_c_int .and. &
      status%final_publication_arrays > 0_c_int .and. &
      status%final_publication_bytes > 0_c_size_t .and. &
      status%final_publication_cosmo == &
      merge(1_c_int, 0_c_int, status%cosmo_enabled /= 0_c_int)
  end function backend_final_publication_complete

  logical function backend_strict_host_control_clean(status) result(clean)
    implicit none
    type(gpu_mozyme_scf_status), intent(in) :: status

    if (.not. mozyme_gpu_scf_strict_resident()) then
      clean = .true.
      return
    end if
    clean = status%strict_resident_host_syncs == 0_c_int .and. &
      status%strict_resident_control_polls == 0_c_int
  end function backend_strict_host_control_clean

  logical function backend_cnvgz_work_complete(status) result(complete)
    implicit none
    type(gpu_mozyme_scf_status), intent(in) :: status

    complete = status%cnvgz_active_calls >= 0_c_int .and. &
      status%cnvgz_noop_calls >= 0_c_int .and. &
      status%cnvgz_active_calls + status%cnvgz_noop_calls > 0_c_int
  end function backend_cnvgz_work_complete

  logical function backend_resident_tidy_complete(initial_setup_requested, &
      initial_tidy_done) &
      result(complete)
    implicit none
    logical, intent(in) :: initial_setup_requested
    logical, intent(in) :: initial_tidy_done

    complete = .not. (mozyme_gpu_scf_strict_resident() .and. &
      initial_setup_requested .and. .not. initial_tidy_done)
  end function backend_resident_tidy_complete

  logical function backend_completed_scf(code, status, previous_iter, &
      initial_setup_requested, initial_tidy_done) &
      result(completed)
    implicit none
    integer(c_int), intent(in) :: code
    type(gpu_mozyme_scf_status), intent(in) :: status
    integer, intent(in) :: previous_iter
    logical, intent(in) :: initial_setup_requested
    logical, intent(in) :: initial_tidy_done

    completed = code == GPU_MOZYME_SCF_SUCCESS .and. &
      status%version == GPU_MOZYME_SCF_ABI_VERSION .and. &
      status%code == GPU_MOZYME_SCF_SUCCESS .and. &
      status%ready == 1_c_int .and. status%resident == 1_c_int .and. &
      status%device_id >= 0_c_int .and. &
      status%stage_required == GPU_MOZYME_SCF_STAGE_FULL .and. &
      status%stage_completed == GPU_MOZYME_SCF_STAGE_FULL .and. &
      status%stage_missing == 0_c_int .and. &
      backend_stage_calls_complete(status, previous_iter) .and. &
      status%resident_decision == &
      GPU_MOZYME_SCF_RESIDENT_DECISION_COMPLETE .and. &
      backend_resident_fock_plans_complete(status) .and. &
      backend_strict_host_control_clean(status) .and. &
      backend_final_publication_complete(status) .and. &
      backend_cnvgz_work_complete(status) .and. &
      backend_resident_tidy_complete(initial_setup_requested, &
      initial_tidy_done) .and. &
      status%isitsc_okscf == 1_c_int .and. &
      backend_pls_resolved(status) .and. &
      (status%cosmo_enabled == 0_c_int .or. &
       (status%cosmo_cg_control_resident == 1_c_int .and. &
        status%cosmo_cg_converged == 1_c_int .and. &
        status%cosmo_cg_breakdown == 0_c_int .and. &
        status%cosmo_cg_host_syncs == 0_c_int)) .and. &
      int(status%iterations) > previous_iter .and. &
      usable_real(status%energy_total)
  end function backend_completed_scf

  logical function backend_completed_step(code, status, previous_iter) &
      result(completed)
    use gpu_mozyme_scf_interfaces, only: GPU_MOZYME_SCF_STAGE_UPLOAD, &
      GPU_MOZYME_SCF_STAGE_EIMP, GPU_MOZYME_SCF_STAGE_DIAGG, &
      GPU_MOZYME_SCF_STAGE_DENSITY, GPU_MOZYME_SCF_STAGE_FOCK, &
      GPU_MOZYME_SCF_STAGE_CNVGZ, GPU_MOZYME_SCF_STAGE_HELECZ, &
      GPU_MOZYME_SCF_STAGE_ISITSC, GPU_MOZYME_SCF_STAGE_ADDHB, &
      GPU_MOZYME_SCF_STAGE_CHECK
    implicit none
    integer(c_int), intent(in) :: code
    type(gpu_mozyme_scf_status), intent(in) :: status
    integer, intent(in) :: previous_iter
    integer(c_int) :: step_stages

    step_stages = GPU_MOZYME_SCF_STAGE_UPLOAD + &
      GPU_MOZYME_SCF_STAGE_EIMP + GPU_MOZYME_SCF_STAGE_DIAGG + &
      GPU_MOZYME_SCF_STAGE_DENSITY + GPU_MOZYME_SCF_STAGE_ADDHB + &
      GPU_MOZYME_SCF_STAGE_FOCK + GPU_MOZYME_SCF_STAGE_CNVGZ + &
      GPU_MOZYME_SCF_STAGE_HELECZ + GPU_MOZYME_SCF_STAGE_ISITSC + &
      GPU_MOZYME_SCF_STAGE_CHECK
    completed = code == GPU_MOZYME_SCF_SUCCESS .and. &
      status%version == GPU_MOZYME_SCF_ABI_VERSION .and. &
      status%code == GPU_MOZYME_SCF_SUCCESS .and. &
      status%ready == 1_c_int .and. status%resident == 1_c_int .and. &
      status%device_id >= 0_c_int .and. &
      status%stage_missing == 0_c_int .and. &
      iand(status%stage_completed, step_stages) == step_stages .and. &
      backend_pls_resolved(status) .and. &
      int(status%iterations) > previous_iter .and. &
      usable_real(status%energy_total)
  end function backend_completed_step

  character(len=64) function run_fallback_reason(code, status, previous_iter, &
      initial_setup_requested, initial_tidy_done) &
      result(reason)
    implicit none
    integer(c_int), intent(in) :: code
    type(gpu_mozyme_scf_status), intent(in) :: status
    integer, intent(in) :: previous_iter
    logical, intent(in) :: initial_setup_requested
    logical, intent(in) :: initial_tidy_done

    if (.not. backend_pls_resolved(status)) then
      reason = 'backend_pls_restart_required'
    else if (code == GPU_MOZYME_SCF_CPU_BOUNDARY .or. &
        status%code == GPU_MOZYME_SCF_CPU_BOUNDARY) then
      reason = 'backend_cpu_boundary'
    else if (status%version /= GPU_MOZYME_SCF_ABI_VERSION) then
      reason = 'backend_bad_abi'
    else if (status%code /= GPU_MOZYME_SCF_SUCCESS .and. &
        status%code /= GPU_MOZYME_SCF_UNSUPPORTED .and. &
        status%code /= GPU_MOZYME_SCF_NOT_READY) then
      reason = stage_code_reason('run_status', status%code)
    else if (status%resident /= 1_c_int) then
      reason = 'backend_not_resident'
    else if (status%stage_required == 0_c_int .or. status%stage_missing /= 0_c_int) then
      reason = 'backend_missing_stages'
    else if (.not. backend_stage_calls_complete(status, previous_iter)) then
      reason = 'backend_stage_call_counts'
    else if (status%resident_decision /= &
        GPU_MOZYME_SCF_RESIDENT_DECISION_COMPLETE) then
      reason = 'backend_resident_decision'
    else if (.not. backend_resident_fock_plans_complete(status)) then
      reason = 'backend_resident_fock_partial_coverage'
    else if (.not. backend_strict_host_control_clean(status)) then
      reason = 'backend_strict_host_control_polling'
    else if (.not. backend_final_publication_complete(status)) then
      reason = 'backend_final_publication'
    else if (.not. backend_cnvgz_work_complete(status)) then
      reason = 'backend_cnvgz_no_device_work'
    else if (.not. backend_resident_tidy_complete(initial_setup_requested, &
        initial_tidy_done)) then
      reason = 'backend_resident_tidy_missing'
    else if (status%isitsc_okscf /= 1_c_int) then
      reason = 'backend_not_converged'
    else if (int(status%iterations) <= previous_iter) then
      reason = 'backend_incomplete_iterations'
    else if (.not. usable_real(status%energy_total)) then
      reason = 'backend_invalid_energy'
    else if (status%ready /= 1_c_int) then
      reason = 'backend_not_ready'
    else if (code == GPU_MOZYME_SCF_NOT_READY .or. &
        status%code == GPU_MOZYME_SCF_NOT_READY) then
      reason = 'backend_not_ready'
    else if (code /= GPU_MOZYME_SCF_SUCCESS) then
      reason = stage_code_reason('run', code)
    else if (status%code /= GPU_MOZYME_SCF_SUCCESS) then
      reason = stage_code_reason('run_status', status%code)
    else
      reason = 'backend_incomplete_success'
    end if
  end function run_fallback_reason

  logical function backend_stage_calls_complete(status, previous_iter) &
      result(complete)
    implicit none
    type(gpu_mozyme_scf_status), intent(in) :: status
    integer, intent(in) :: previous_iter
    integer :: idx
    integer :: min_calls

    min_calls = max(1, int(status%iterations) - previous_iter)
    complete = .true.
    if (int(status%resident_stage_calls(1)) < 1) then
      complete = .false.
      return
    end if
    do idx = 2, 10
      if (int(status%resident_stage_calls(idx)) < min_calls) then
        complete = .false.
        return
      end if
    end do
  end function backend_stage_calls_complete

  character(len=64) function stage_code_reason(stage, code) result(reason)
    implicit none
    character(len=*), intent(in) :: stage
    integer(c_int), intent(in) :: code

    select case (code)
    case (GPU_MOZYME_SCF_SUCCESS)
      reason = trim(stage)//'_status_not_ready'
    case (GPU_MOZYME_SCF_NOT_READY)
      reason = 'backend_not_ready'
    case (GPU_MOZYME_SCF_BAD_ARGUMENT)
      reason = trim(stage)//'_bad_argument'
    case (GPU_MOZYME_SCF_UNSUPPORTED)
      reason = 'backend_unsupported'
    case (GPU_MOZYME_SCF_CPU_BOUNDARY)
      reason = 'backend_cpu_boundary'
    case default
      reason = trim(stage)//'_failed'
    end select
  end function stage_code_reason

  logical function usable_real(value)
    implicit none
    double precision, intent(in) :: value

    usable_real = (value == value .and. abs(value) < huge(value))
  end function usable_real

  character(len=24) function int_text(value) result(text)
    implicit none
    integer(c_int), intent(in) :: value

    write(text, '(i0)') int(value)
  end function int_text

  character(len=160) function stage_mask_names(mask) result(names)
    implicit none
    integer(c_int), intent(in) :: mask

    names = ''
    call append_stage_name(names, mask, GPU_MOZYME_SCF_STAGE_UPLOAD, 'upload')
    call append_stage_name(names, mask, GPU_MOZYME_SCF_STAGE_EIMP, 'eimp')
    call append_stage_name(names, mask, GPU_MOZYME_SCF_STAGE_DIAGG, 'diagg')
    call append_stage_name(names, mask, GPU_MOZYME_SCF_STAGE_DENSITY, 'density')
    call append_stage_name(names, mask, GPU_MOZYME_SCF_STAGE_FOCK, 'fock')
    call append_stage_name(names, mask, GPU_MOZYME_SCF_STAGE_CNVGZ, 'cnvgz')
    call append_stage_name(names, mask, GPU_MOZYME_SCF_STAGE_HELECZ, 'helecz')
    call append_stage_name(names, mask, GPU_MOZYME_SCF_STAGE_ISITSC, 'isitsc')
    call append_stage_name(names, mask, GPU_MOZYME_SCF_STAGE_ADDHB, 'addhb')
    call append_stage_name(names, mask, GPU_MOZYME_SCF_STAGE_CHECK, 'check')
    if (len_trim(names) == 0) names = 'none'
  end function stage_mask_names

  subroutine append_stage_name(names, mask, bit, name)
    implicit none
    character(len=*), intent(inout) :: names
    integer(c_int), intent(in) :: mask, bit
    character(len=*), intent(in) :: name

    if (iand(mask, bit) == 0_c_int) return
    if (len_trim(names) == 0) then
      names = trim(name)
    else
      names = trim(names)//'+'//trim(name)
    end if
  end subroutine append_stage_name

  subroutine trace_status(iw, trace, status, prefix)
    implicit none
    integer, intent(in) :: iw
    logical, intent(in) :: trace
    type(gpu_mozyme_scf_status), intent(in) :: status
    character(len=*), intent(in) :: prefix
    integer :: strict_resident_value
    integer :: no_fallback_value
    integer :: full_stage_value
    integer :: resident_decision_complete_value

    write(iw,'(1x,a,1x,a,1x,a,1x,i0,1x,a,1x,a,1x,a,1x,i0,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', trim(prefix), 'code=', int(status%code), &
      'code_name=', trim(status_code_name(status%code)), &
      'ready=', int(status%ready), 'resident=', int(status%resident)
    write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', 'stage_completed=', int(status%stage_completed), &
      'stage_required=', int(status%stage_required), &
      'stage_missing=', int(status%stage_missing)
    write(iw,'(1x,a,1x,a,1x,a,1x,a,1x,a)') &
      '[MOZYME GPU SCF]', 'stage_completed_names=', &
      trim(stage_mask_names(status%stage_completed)), &
      'stage_missing_names=', trim(stage_mask_names(status%stage_missing))
    write(iw,'(1x,a,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', 'resident_decision=', int(status%resident_decision)
    strict_resident_value = merge(1, 0, mozyme_gpu_scf_strict_resident())
    no_fallback_value = merge(1, 0, mozyme_gpu_scf_no_fallback_required())
    full_stage_value = merge(1, 0, &
      status%stage_required == GPU_MOZYME_SCF_STAGE_FULL .and. &
      status%stage_completed == GPU_MOZYME_SCF_STAGE_FULL .and. &
      status%stage_missing == 0_c_int)
    resident_decision_complete_value = merge(1, 0, &
      status%resident_decision == GPU_MOZYME_SCF_RESIDENT_DECISION_COMPLETE)
    write(iw,'(1x,a,1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,'// &
        '1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', 'strict_proof', &
      'strict_resident=', strict_resident_value, &
      'no_fallback_required=', no_fallback_value, &
      'full_stage_mask=', full_stage_value, &
      'resident_decision_complete=', resident_decision_complete_value, &
      'strict_host_syncs=', int(status%strict_resident_host_syncs), &
      'strict_control_polls=', int(status%strict_resident_control_polls)
    write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', 'final_publication_done=', &
      int(status%final_publication_done), 'arrays=', &
      int(status%final_publication_arrays), 'bytes=', &
      status%final_publication_bytes, 'cosmo=', &
      int(status%final_publication_cosmo)
    if (index(prefix, 'status=success') > 0) then
      write(iw,'(1x,a,1x,a)') &
        '[MOZYME GPU SCF]', 'final_density=current_resident'
    end if
    write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', 'resident_fock_plan_id=', &
      int(status%resident_fock_plan_id), &
      'resident_fock_plan_full_coverage=', &
      int(status%resident_fock_plan_full_coverage), &
      'resident_fock_plan_partial_coverage=', &
      int(status%resident_fock_plan_partial_coverage), &
      'resident_fock_plan_required_mask=', &
      int(status%resident_fock_plan_required_mask), &
      'resident_fock_plan_covered_mask=', &
      int(status%resident_fock_plan_covered_mask)
    write(iw,'(1x,a,1x,a,10(1x,a,1x,i0))') &
      '[MOZYME GPU SCF]', 'resident_stage_calls', &
      'upload=', int(status%resident_stage_calls(1)), &
      'eimp=', int(status%resident_stage_calls(2)), &
      'diagg=', int(status%resident_stage_calls(3)), &
      'density=', int(status%resident_stage_calls(4)), &
      'fock=', int(status%resident_stage_calls(5)), &
      'cnvgz=', int(status%resident_stage_calls(6)), &
      'helecz=', int(status%resident_stage_calls(7)), &
      'isitsc=', int(status%resident_stage_calls(8)), &
      'addhb=', int(status%resident_stage_calls(9)), &
      'check=', int(status%resident_stage_calls(10))
    write(iw,'(1x,a,1x,a,10(1x,a,1x,es12.5))') &
      '[MOZYME GPU SCF]', 'resident_stage_ms', &
      'upload=', status%resident_stage_ms(1), &
      'eimp=', status%resident_stage_ms(2), &
      'diagg=', status%resident_stage_ms(3), &
      'density=', status%resident_stage_ms(4), &
      'fock=', status%resident_stage_ms(5), &
      'cnvgz=', status%resident_stage_ms(6), &
      'helecz=', status%resident_stage_ms(7), &
      'isitsc=', status%resident_stage_ms(8), &
      'addhb=', status%resident_stage_ms(9), &
      'check=', status%resident_stage_ms(10)
    if (status%final_reorth_applied /= 0_c_int .and. &
        index(prefix, 'status=success') > 0) then
      write(iw,'(1x,a," status=success resident=1 ms=",f10.3," sum=",es12.5)') &
        '[MOZYME GPU reorth]', status%final_reorth_ms, &
        status%final_reorth_sum
    end if
    write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', 'idiagg=', int(status%idiagg), &
      'nhb=', int(status%nhb), 'addhb_due=', int(status%addhb_due), &
      'addhb_applied=', int(status%addhb_applied), &
      'addhb_nij=', int(status%addhb_nij)
    write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', 'isitsc_okscf=', int(status%isitsc_okscf), &
      'isitsc_iscf=', int(status%isitsc_iscf), &
      'isitsc_iemin=', int(status%isitsc_iemin), &
      'isitsc_iemax=', int(status%isitsc_iemax), &
      'isitsc_scf1=', int(status%isitsc_scf1)
    if (status%pls_supervisor_calls /= 0_c_int .or. &
        status%pls_restart_required /= 0_c_int .or. &
        status%pls_restart_reset_device_calls /= 0_c_int .or. &
        status%pls_restart_done /= 0_c_int) then
      write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
        '[MOZYME GPU SCF]', 'pls_supervisor_calls=', &
        int(status%pls_supervisor_calls), 'pls_restart_required=', &
        int(status%pls_restart_required), 'pls_history_count=', &
        int(status%pls_history_count)
      write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0)') &
        '[MOZYME GPU SCF]', 'pls_restart_reset_device_calls=', &
        int(status%pls_restart_reset_device_calls), 'pls_restart_done=', &
        int(status%pls_restart_done)
      write(iw,'(1x,a,1x,a,1x,es12.5,1x,a,1x,es12.5)') &
        '[MOZYME GPU SCF]', 'pls_ovmax_delta=', status%pls_ovmax_delta, &
        'pls_energy_delta=', status%pls_energy_delta
    end if
    if (status%cosmo_enabled /= 0_c_int) then
      write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,'// &
          '1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
        '[MOZYME GPU SCF]', 'cosmo_enabled=', int(status%cosmo_enabled), &
        'cosmo_fock_calls=', int(status%cosmo_fock_calls), &
        'cosmo_matvec_calls=', int(status%cosmo_matvec_calls), &
        'cosmo_cg_iterations=', int(status%cosmo_cg_iterations), &
        'cosmo_nps=', int(status%cosmo_nps), &
        'cosmo_lm61=', int(status%cosmo_lm61), &
        'cosmo_pair_count=', int(status%cosmo_pair_count)
      write(iw,'(1x,a,1x,a,1x,es12.5,1x,a,1x,es12.5,1x,a,1x,es12.5)') &
        '[MOZYME GPU SCF]', 'cosmo_solv_energy=', &
        status%cosmo_solv_energy, 'cosmo_ediel=', status%cosmo_ediel, &
        'cosmo_last_residual=', status%cosmo_last_residual
      write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,'// &
          '1x,a,1x,i0,1x,a,1x,es12.5)') &
        '[MOZYME GPU SCF]', 'cosmo_cg_control_resident=', &
        int(status%cosmo_cg_control_resident), &
        'cosmo_cg_converged=', int(status%cosmo_cg_converged), &
        'cosmo_cg_breakdown=', int(status%cosmo_cg_breakdown), &
        'cosmo_cg_host_syncs=', int(status%cosmo_cg_host_syncs), &
        'cosmo_cg_target_tol=', status%cosmo_cg_target_tol
    end if
    write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', 'device_id=', int(status%device_id), &
      'natoms=', int(status%natoms), 'norbs=', int(status%norbs), &
      'iterations=', int(status%iterations)
    write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', 'cnvgz_active_calls=', &
      int(status%cnvgz_active_calls), 'cnvgz_noop_calls=', &
      int(status%cnvgz_noop_calls)
    if (.not. trace) return
    write(iw,'(1x,a,1x,a,1x,f12.3,1x,a,1x,es12.5,1x,a,1x,es12.5)') &
      '[MOZYME GPU SCF]', 'wall_ms=', status%wall_ms, &
      'density_max=', status%density_max, 'density_rms=', status%density_rms
    write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,es12.5)') &
      '[MOZYME GPU SCF]', 'diagg_nij=', int(status%diagg_nij), &
      'diagg_nf=', int(status%diagg_nf), 'diagg_tiny=', status%diagg_tiny
    write(iw,'(1x,a,1x,a,1x,es12.5,1x,a,1x,es12.5)') &
      '[MOZYME GPU SCF]', 'diagg_sumt=', status%diagg_sumt, &
      'diagg_sumb=', status%diagg_sumb
    write(iw,'(1x,a,1x,a,1x,es12.5,1x,a,1x,i0,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', 'next_tiny=', status%next_tiny, &
      'diagg2_nrejct1=', int(status%diagg2_nrejct(1)), &
      'diagg2_nrejct2=', int(status%diagg2_nrejct(2))
    write(iw,'(1x,a,1x,a,1x,es20.10)') &
      '[MOZYME GPU SCF]', 'energy_total=', status%energy_total
    write(iw,'(1x,a,1x,a,1x,es20.10,1x,a,1x,es20.10)') &
      '[MOZYME GPU SCF]', 'energy_scf=', status%energy_scf, &
      'energy_delta=', status%energy_delta
    write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,es12.5)') &
      '[MOZYME GPU SCF]', 'use_three_point=', int(status%use_three_point), &
      'lstart=', int(status%lstart), 'shift=', status%shift
  end subroutine trace_status

  character(len=24) function status_code_name(code) result(name)
    implicit none
    integer(c_int), intent(in) :: code

    select case (code)
    case (GPU_MOZYME_SCF_SUCCESS)
      name = 'SUCCESS'
    case (GPU_MOZYME_SCF_NOT_READY)
      name = 'NOT_READY'
    case (GPU_MOZYME_SCF_BAD_ARGUMENT)
      name = 'BAD_ARGUMENT'
    case (GPU_MOZYME_SCF_UNSUPPORTED)
      name = 'UNSUPPORTED'
    case (GPU_MOZYME_SCF_CPU_BOUNDARY)
      name = 'CPU_BOUNDARY'
    case default
      name = 'UNKNOWN'
    end select
  end function status_code_name
#endif

  subroutine trace_begin(iw, niter, nocc, nvir, itrmax, selcon)
    use molkst_C, only: norbs, numat, nscf
    implicit none
    integer, intent(in) :: iw, niter, nocc, nvir, itrmax
    double precision, intent(in) :: selcon

    write(iw,'(1x,a)') '[PROFILE] MOZYME_GPU_SCF begin'
    write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', 'numat=', numat, 'norbs=', norbs, &
      'nscf=', nscf
    write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
      '[MOZYME GPU SCF]', 'niter=', niter, 'nocc=', nocc, 'nvir=', nvir
    write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,es12.5)') &
      '[MOZYME GPU SCF]', 'itrmax=', itrmax, 'selcon=', selcon
    call flush(iw)
  end subroutine trace_begin

  subroutine trace_end(iw, trace, success, start_time, message, force_message)
    implicit none
    integer, intent(in) :: iw
    logical, intent(in) :: trace, success
    double precision, intent(in) :: start_time
    character(len=*), intent(in) :: message
    logical, intent(in), optional :: force_message

    double precision :: elapsed_ms
    logical :: emit_message

    emit_message = trace
    if (present(force_message)) emit_message = emit_message .or. force_message
    if (emit_message) then
      write(iw,'(1x,a,1x,a)') '[MOZYME GPU SCF]', trim(message)
    end if
    if (trace) then
      call cpu_time(elapsed_ms)
      elapsed_ms = max(0.d0, elapsed_ms - start_time) * 1000.d0
      write(iw,'(1x,a,1x,l1,1x,a,1x,f12.3)') &
        '[PROFILE] MOZYME_GPU_SCF end success=', success, 'ms=', elapsed_ms
    end if
    if (emit_message .or. trace) call flush(iw)
  end subroutine trace_end

  character(len=32) function scf_failure_status()
    implicit none

    if (mozyme_gpu_scf_no_fallback_required()) then
      scf_failure_status = 'status=strict_abort'
    else
      scf_failure_status = 'status=fallback_cpu'
    end if
  end function scf_failure_status

  character(len=256) function scf_failure_message(detail)
    implicit none
    character(len=*), intent(in) :: detail

    if (mozyme_gpu_scf_no_fallback_required()) then
      scf_failure_message = 'status=strict_abort '//trim(detail)
    else
      scf_failure_message = 'status=fallback_cpu '//trim(detail)
    end if
  end function scf_failure_message

  logical function mozyme_gpu_scf_profile_enabled()
    implicit none

    if (.not. profile_checked) then
      profile_enabled = env_enabled('MOPAC_GPU_PROFILE') .or. &
        env_enabled('MOPAC_MOZYME_SECTION_PROFILE')
      profile_checked = .true.
    end if
    mozyme_gpu_scf_profile_enabled = profile_enabled
  end function mozyme_gpu_scf_profile_enabled

  subroutine ensure_request_state()
    implicit none
    logical :: strict_requested, full_scf_requested, resident_requested

    if (request_checked) return

    strict_requested = env_is_one('MOPAC_MOZYME_SCF_STRICT_RESIDENT')
    strict_requested = strict_requested .or. &
      env_is_one('MOPAC_MOZYME_GPU_STRICT')
    full_scf_requested = env_is_one('MOPAC_MOZYME_SCF_GPU')
    full_scf_requested = full_scf_requested .or. &
      env_is_one('MOPAC_MOZYME_FULL_SCF_GPU')
    resident_requested = env_is_one('MOPAC_MOZYME_RESIDENT_SCF')
    request_present = resident_requested .or. full_scf_requested .or. &
      strict_requested
    ! Only the explicit strict/proof variables forbid CPU setup and bookend
    ! work; MOPAC_MOZYME_SCF_GPU runs the resident SCF loop on the device but
    ! lets per-geometry host routines (hcore, add_more_interactions, tidy,
    ! OLD_SCF warm starts) proceed, which optimization and MD steps need.
    no_fallback_required = strict_requested
    request_enabled = .false.
    request_reason = 'not_requested'

    if (.not. request_present) then
      request_reason = 'not_requested'
    else if (.not. (strict_requested .or. full_scf_requested) .and. &
        .not. env_is_one('MOPAC_MOZYME_SCF_EXPERIMENTAL')) then
      request_reason = 'experimental_gate_closed'
    else if (env_enabled('MOPAC_NOGPU')) then
      request_reason = 'disabled_by_MOPAC_NOGPU'
    else if (env_enabled('MOZYME_GPU_OFF')) then
      request_reason = 'disabled_by_MOZYME_GPU_OFF'
    else if (env_is_cpu_task('MOPAC_GPU_SCFTASK')) then
      request_reason = 'disabled_by_MOPAC_GPU_SCFTASK'
    else
      request_enabled = .true.
      request_reason = 'requested'
    end if
    request_checked = .true.
  end subroutine ensure_request_state

  logical function env_is_one(var_name)
    implicit none
    character(len=*), intent(in) :: var_name

    env_is_one = env_enabled(var_name)
  end function env_is_one

  logical function env_enabled(var_name)
    implicit none
    character(len=*), intent(in) :: var_name

    integer :: env_len, env_status
    character(len=32) :: env_value
    character(len=32) :: value

    env_enabled = .false.
    env_value = ' '
    call get_environment_variable(var_name, env_value, length=env_len, &
      status=env_status)
    if (env_status /= 0 .or. env_len <= 0) return

    value = adjustl(env_value)
    select case (trim(value))
    case ('0', 'f', 'F', 'false', 'FALSE', 'False', &
          'n', 'N', 'no', 'NO', 'No', 'off', 'OFF', 'Off')
      env_enabled = .false.
    case default
      env_enabled = .true.
    end select
  end function env_enabled

  logical function env_is_cpu_task(var_name)
    implicit none
    character(len=*), intent(in) :: var_name

    integer :: env_len, env_status
    character(len=32) :: env_value

    env_value = ' '
    call get_environment_variable(var_name, env_value, length=env_len, &
      status=env_status)
    if (env_status /= 0 .or. env_len <= 0) then
      env_is_cpu_task = .false.
      return
    end if

    select case (trim(adjustl(env_value)))
    case ('cpu', 'CPU', 'Cpu')
      env_is_cpu_task = .true.
    case default
      env_is_cpu_task = .false.
    end select
  end function env_is_cpu_task

end module mozyme_gpu_scf_driver
