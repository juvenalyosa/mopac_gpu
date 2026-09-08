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

module mozyme_gpu_tidy
  use iso_c_binding, only: c_int, c_double, c_ptr, c_loc, c_null_ptr
  implicit none
  private
  public :: mozyme_gpu_tidy_try

contains

  logical function mozyme_gpu_tidy_try(mode, ln, mn, use_selmos, error_code) &
      result(done)
    use chanel_C, only: iw
#ifdef GPU
    use molkst_C, only: norbs, numat
    use MOZYME_C, only: cocc, cocc_dim, cvir, cvir_dim, icocc, &
      icocc_dim, icvir, icvir_dim, iorbs, ncocc, ncvir, nce, ncf, &
      noccupied, nvirtual, nnce, nncf, thresh, jopt, numred, nelred, &
      norred
    use mod_vars_cuda, only: lgpu, mozyme_gpu_requested, &
      mozyme_resident_fock_gpu
    use mozyme_gpu_int_utils, only: mozyme_c_int_nonnegative_or_zero, &
      mozyme_c_int_positive_or_zero
#endif
    implicit none
    integer, intent(in) :: mode
    integer, intent(out) :: ln, mn
    logical, intent(in), optional :: use_selmos
    integer, intent(out), optional :: error_code
#ifdef GPU
    interface
      function mopac_cuda_mozyme_tidy(nmos_c, natoms_c, norbs_c, ic_dim_c, &
          c_dim_c, thresh_c, use_selmos_c, numred_c, mode_c, nc_c, ic_c, c_c, &
          nnc_c, ncmo_c, iorbs_c, jopt_c, ln_c, mn_c, selected_c, &
          wall_ms_c) bind(C,name='mopac_cuda_mozyme_tidy') &
          result(code)
        use iso_c_binding, only: c_int, c_double, c_ptr
        integer(c_int), value :: nmos_c, natoms_c, norbs_c
        integer(c_int), value :: ic_dim_c, c_dim_c
        integer(c_int), value :: use_selmos_c, numred_c, mode_c
        real(c_double), value :: thresh_c
        integer(c_int) :: nc_c(*), ic_c(*), nnc_c(*), ncmo_c(*)
        integer(c_int) :: iorbs_c(*), ln_c, mn_c, selected_c
        type(c_ptr), value :: jopt_c
        real(c_double) :: c_c(*), wall_ms_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_tidy
    end interface
    integer(c_int) :: code
    integer(c_int) :: ln_c, mn_c, selected_c
    integer(c_int) :: use_selmos_c
    type(c_ptr) :: jopt_ptr
    real(c_double) :: wall_ms
#endif
    character(len=8) :: mode_name
    logical :: trace_requested
    logical :: select_lmos

    done = .false.
    ln = 0
    mn = 0
    if (present(error_code)) error_code = -999999
    mode_name = 'unknown'
    if (mode == 1) mode_name = 'occupied'
    if (mode == 2) mode_name = 'virtual'
    select_lmos = .false.
    if (present(use_selmos)) select_lmos = use_selmos
    trace_requested = env_is_one('MOPAC_MOZYME_SCF_STRICT_RESIDENT') .or. &
      env_is_one('MOPAC_MOZYME_SCF_GPU') .or. &
      env_is_one('MOPAC_MOZYME_GPU_STRICT') .or. &
      env_is_one('MOPAC_MOZYME_FULL_SCF_GPU') .or. &
      env_is_one('MOPAC_MOZYME_RESIDENT_SCF') .or. &
      env_is_one('MOPAC_MOZYME_TIDY_GPU')
#ifdef GPU
    if (.not. trace_requested) return
    if (mode /= 1 .and. mode /= 2) then
      call tidy_trace_fallback(iw, mode_name, 'bad_mode')
      return
    end if
    if (storage_size(0) /= storage_size(0_c_int) .or. &
        storage_size(0.0d0) /= storage_size(0.0_c_double)) then
      call tidy_trace_fallback(iw, mode_name, 'kind_mismatch')
      return
    end if
    if (.not. (lgpu .or. mozyme_resident_fock_gpu)) then
      call tidy_trace_fallback(iw, mode_name, 'gpu_disabled')
      return
    end if
    if (.not. mozyme_gpu_requested) then
      call tidy_trace_fallback(iw, mode_name, 'not_requested')
      return
    end if
    if (select_lmos .and. numred > 0 .and. .not. allocated(jopt)) then
      call tidy_trace_fallback(iw, mode_name, 'jopt_missing')
      return
    end if
    if (select_lmos .and. (numred < 0 .or. numred > numat)) then
      call tidy_trace_fallback(iw, mode_name, 'numred_out_of_range')
      return
    end if

    ln_c = 0_c_int
    mn_c = 0_c_int
    selected_c = -1_c_int
    use_selmos_c = merge(1_c_int, 0_c_int, select_lmos)
    jopt_ptr = c_null_ptr
    if (select_lmos .and. numred > 0) jopt_ptr = c_loc(jopt(1))
    wall_ms = 0.0_c_double
    if (mode == 1) then
      code = mopac_cuda_mozyme_tidy( &
        mozyme_c_int_nonnegative_or_zero(noccupied), &
        mozyme_c_int_positive_or_zero(numat), &
        mozyme_c_int_positive_or_zero(norbs), &
        mozyme_c_int_positive_or_zero(icocc_dim), &
        mozyme_c_int_positive_or_zero(cocc_dim), real(thresh, c_double), &
        use_selmos_c, mozyme_c_int_nonnegative_or_zero(numred), &
        mozyme_c_int_positive_or_zero(mode), &
        ncf, icocc, cocc, nncf, ncocc, iorbs, jopt_ptr, ln_c, mn_c, &
        selected_c, wall_ms)
    else
      code = mopac_cuda_mozyme_tidy( &
        mozyme_c_int_nonnegative_or_zero(nvirtual), &
        mozyme_c_int_positive_or_zero(numat), &
        mozyme_c_int_positive_or_zero(norbs), &
        mozyme_c_int_positive_or_zero(icvir_dim), &
        mozyme_c_int_positive_or_zero(cvir_dim), real(thresh, c_double), &
        use_selmos_c, mozyme_c_int_nonnegative_or_zero(numred), &
        mozyme_c_int_positive_or_zero(mode), &
        nce, icvir, cvir, nnce, ncvir, iorbs, jopt_ptr, ln_c, mn_c, &
        selected_c, wall_ms)
    end if

    done = code == 0_c_int
    if (present(error_code)) error_code = int(code)
    ln = int(ln_c)
    mn = int(mn_c)
    if (done .and. selected_c >= 0_c_int) then
      if (mode == 1) then
        nelred = 2 * int(selected_c)
      else
        norred = int(selected_c) + nelred / 2
      end if
    end if
    if (done) then
      write(iw,'(1x,a," status=success mode=",a," selmos=",i0,' // &
        '" selected=",i0," ln=",i0," mn=",i0," wall_ms=",f10.3)') &
        '[MOZYME GPU tidy]', trim(mode_name), int(use_selmos_c), &
        int(selected_c), ln, mn, &
        max(wall_ms, 0.001_c_double)
    else
      write(iw,'(1x,a," status=fallback_cpu mode=",a," code=",i0)') &
        '[MOZYME GPU tidy]', trim(mode_name), int(code)
    end if
    call flush(iw)
#else
    if (present(error_code)) error_code = -3
    if (trace_requested) call tidy_trace_fallback(iw, mode_name, 'not_gpu_build')
#endif
  end function mozyme_gpu_tidy_try

  subroutine tidy_trace_fallback(iw, mode_name, reason)
    implicit none
    integer, intent(in) :: iw
    character(len=*), intent(in) :: mode_name
    character(len=*), intent(in) :: reason

    write(iw,'(1x,a," status=fallback_cpu mode=",a," reason=",a)') &
      '[MOZYME GPU tidy]', trim(mode_name), trim(reason)
    call flush(iw)
  end subroutine tidy_trace_fallback

  logical function env_is_one(var_name)
    implicit none
    character(len=*), intent(in) :: var_name

    integer :: env_len, env_status
    character(len=32) :: env_value
    character(len=32) :: value

    env_value = ' '
    call get_environment_variable(var_name, env_value, length=env_len, &
      status=env_status)
    env_is_one = .false.
    if (env_status /= 0 .or. env_len <= 0) return
    value = adjustl(env_value)
    call upcase(value, len_trim(value))
    select case (trim(value))
    case ('', '0', 'FALSE', 'F', 'NO', 'N', 'OFF')
      env_is_one = .false.
    case default
      env_is_one = .true.
    end select
  end function env_is_one

end module mozyme_gpu_tidy
