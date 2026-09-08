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

module mozyme_gpu_relocalize
  use iso_c_binding, only: c_int, c_double
  implicit none
  private
  public :: mozyme_gpu_relocalize_try

contains

  logical function mozyme_gpu_relocalize_try(lmo_type) result(done)
    use chanel_C, only: iw
#ifdef GPU
    use common_arrays_C, only: eigs, f, nfirst, nlast, p
    use molkst_C, only: mpack, nelecs, norbs, numat
    use MOZYME_C, only: cocc, cocc_dim, cvir, cvir_dim, icocc, &
      icocc_dim, icvir, icvir_dim, ijc, iorbs, lijbo, nce, ncf, &
      ncocc, ncvir, nijbo, nnce, nncf
    use mozyme_gpu_int_utils, only: mozyme_c_int_nonnegative_or_zero, &
      mozyme_c_int_positive_or_zero
    use mod_vars_cuda, only: lgpu, mozyme_gpu_requested, &
      mozyme_resident_fock_gpu
#endif
    implicit none
    character(len=*), intent(in) :: lmo_type
#ifdef GPU
    interface
      function mopac_cuda_mozyme_relocalize(kind_c, natoms_c, norbs_c, &
          mpack_c, nmos_c, c_dim_c, ic_dim_c, use_nijbo_c, c_c, ic_c, &
          nc_c, ncstrt_c, nnc_c, iorbs_c, nfirst_c, nlast_c, nijbo_c, &
          p_c, f_c, eigs_c, iterations_c, total_c, wall_ms_c) &
          bind(C,name='mopac_cuda_mozyme_relocalize') result(code)
        use iso_c_binding, only: c_int, c_double
        integer(c_int), value :: kind_c, natoms_c, norbs_c, mpack_c
        integer(c_int), value :: nmos_c, c_dim_c, ic_dim_c, use_nijbo_c
        real(c_double) :: c_c(*), p_c(*), f_c(*), eigs_c(*)
        integer(c_int) :: ic_c(*), nc_c(*), ncstrt_c(*), nnc_c(*)
        integer(c_int) :: iorbs_c(*), nfirst_c(*), nlast_c(*), nijbo_c(*)
        integer(c_int) :: iterations_c
        real(c_double) :: total_c, wall_ms_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_relocalize
    end interface
    integer(c_int) :: code
    integer(c_int) :: iterations
    integer(c_int) :: use_nijbo
    integer :: nmos
    real(c_double) :: total
    real(c_double) :: wall_ms
#endif
    logical :: trace_requested

    done = .false.
    trace_requested = env_is_one('MOPAC_MOZYME_SCF_STRICT_RESIDENT') .or. &
      env_is_one('MOPAC_MOZYME_SCF_GPU') .or. &
      env_is_one('MOPAC_MOZYME_GPU_STRICT') .or. &
      env_is_one('MOPAC_MOZYME_FULL_SCF_GPU') .or. &
      env_is_one('MOPAC_MOZYME_RESIDENT_SCF') .or. &
      env_is_one('MOPAC_MOZYME_RELOCAL_GPU')
#ifdef GPU
    if (.not. trace_requested) return
    if (storage_size(0) /= storage_size(0_c_int) .or. &
        storage_size(0.0d0) /= storage_size(0.0_c_double)) then
      call relocal_trace_fallback(iw, lmo_type, 'kind_mismatch')
      return
    end if
    if (.not. (lgpu .or. mozyme_resident_fock_gpu)) then
      call relocal_trace_fallback(iw, lmo_type, 'gpu_disabled')
      return
    end if
    if (.not. mozyme_gpu_requested) then
      call relocal_trace_fallback(iw, lmo_type, 'not_requested')
      return
    end if
    if (.not. lijbo .or. .not. allocated(nijbo)) then
      call relocal_trace_fallback(iw, lmo_type, 'no_nijbo')
      return
    end if

    use_nijbo = 1_c_int
    iterations = 0_c_int
    total = 0.0_c_double
    wall_ms = 0.0_c_double
    select case (trim(lmo_type))
    case ('OCCUPIED')
      nmos = nelecs / 2
      code = mopac_cuda_mozyme_relocalize(1_c_int, &
        mozyme_c_int_positive_or_zero(numat), &
        mozyme_c_int_positive_or_zero(norbs), &
        mozyme_c_int_positive_or_zero(mpack), &
        mozyme_c_int_nonnegative_or_zero(nmos), &
        mozyme_c_int_positive_or_zero(cocc_dim), &
        mozyme_c_int_positive_or_zero(icocc_dim), use_nijbo, cocc, &
        icocc, ncf, ncocc, nncf, iorbs, nfirst, nlast, nijbo, p, f, eigs, &
        iterations, total, wall_ms)
      if (code == 0_c_int) ijc = 0
    case ('VIRTUAL')
      nmos = norbs - nelecs / 2
      code = mopac_cuda_mozyme_relocalize(2_c_int, &
        mozyme_c_int_positive_or_zero(numat), &
        mozyme_c_int_positive_or_zero(norbs), &
        mozyme_c_int_positive_or_zero(mpack), &
        mozyme_c_int_nonnegative_or_zero(nmos), &
        mozyme_c_int_positive_or_zero(cvir_dim), &
        mozyme_c_int_positive_or_zero(icvir_dim), use_nijbo, cvir, &
        icvir, nce, ncvir, nnce, iorbs, nfirst, nlast, nijbo, p, f, eigs, &
        iterations, total, wall_ms)
    case default
      call relocal_trace_fallback(iw, lmo_type, 'bad_type')
      return
    end select

    done = code == 0_c_int
    if (done) then
      write(iw,'(1x,a," status=success kind=",a," iterations=",i0,'// &
        '" total=",es13.5," wall_ms=",f10.3)') &
        '[MOZYME GPU relocal]', trim(lmo_type), int(iterations), total, &
        max(wall_ms, 0.001_c_double)
      if (trim(lmo_type) == 'OCCUPIED') then
        write(iw, "(10x,'NUMBER OF ITERATIONS =',i4,/,10x,'LOCALIZATION VALUE =',f14.9,/)") &
          int(iterations), total
      end if
    else
      write(iw,'(1x,a," status=fallback_cpu kind=",a," code=",i0)') &
        '[MOZYME GPU relocal]', trim(lmo_type), int(code)
    end if
    call flush(iw)
#else
    if (trace_requested) then
      call relocal_trace_fallback(iw, lmo_type, 'not_gpu_build')
    end if
#endif
  end function mozyme_gpu_relocalize_try

  subroutine relocal_trace_fallback(iw, lmo_type, reason)
    implicit none
    integer, intent(in) :: iw
    character(len=*), intent(in) :: lmo_type, reason

    write(iw,'(1x,a," status=fallback_cpu kind=",a," reason=",a)') &
      '[MOZYME GPU relocal]', trim(lmo_type), trim(reason)
    call flush(iw)
  end subroutine relocal_trace_fallback

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

end module mozyme_gpu_relocalize
