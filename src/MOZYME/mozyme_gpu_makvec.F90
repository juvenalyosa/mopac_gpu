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

module mozyme_gpu_makvec
  use iso_c_binding, only: c_int, c_double
  implicit none
  private
  public :: mozyme_gpu_makvec_try

contains

  logical function mozyme_gpu_makvec_try() result(done)
    use chanel_C, only: iw
#ifdef GPU
    use common_arrays_C, only: h, p, f, pdiag, nfirst, nlast, nat, &
      nbonds, ibonds, ifact, w, wk, coord
    use molkst_C, only: numat, norbs, mpack, id
    use MOZYME_C, only: iorbs, icocc, icvir, nce, ncf, morb, Lewis_elem, &
      ncocc, ncvir, nnce, nncf, Lewis_tot, noccupied, nvirtual, cocc, &
      cvir, cocc_dim, cvir_dim, icocc_dim, icvir_dim, ipad2, ipad4, &
      lijbo, nijbo, kopt
    use mozyme_gpu_int_utils, only: mozyme_c_int_nonnegative_or_zero, &
      mozyme_c_int_positive_or_zero
    use mod_vars_cuda, only: lgpu, mozyme_gpu_requested, mozyme_resident_fock_gpu
    use mozyme_resident_fock, only: mozyme_resident_fock_prepare_plan, &
      resident_fock_plan_full, mozyme_resident_fock_plan_full_coverage
#endif
    implicit none
#ifdef GPU
    interface
      function mopac_cuda_mozyme_makvec(natoms_c, norbs_c, mpack_c, morb_c, &
          lewis_tot_c, noccupied_c, nvirtual_c, ipad2_c, ipad4_c, &
          icocc_dim_c, cocc_dim_c, icvir_dim_c, cvir_dim_c, ibonds_rows_c, &
          pdiag_c, h_c, p_c, f_c, iorbs_c, nfirst_c, nlast_c, nijbo_c, &
          nbonds_c, ibonds_c, lewis_elem_c, ncf_c, nncf_c, ncocc_c, &
          icocc_c, cocc_c, nce_c, nnce_c, ncvir_c, icvir_c, cvir_c, &
          wall_ms_c) bind(C,name='mopac_cuda_mozyme_makvec') result(code)
        use iso_c_binding, only: c_int, c_double
        integer(c_int), value :: natoms_c, norbs_c, mpack_c, morb_c
        integer(c_int), value :: lewis_tot_c, noccupied_c, nvirtual_c
        integer(c_int), value :: ipad2_c, ipad4_c
        integer(c_int), value :: icocc_dim_c, cocc_dim_c, icvir_dim_c
        integer(c_int), value :: cvir_dim_c, ibonds_rows_c
        real(c_double) :: pdiag_c(*), h_c(*), p_c(*), f_c(*)
        integer(c_int) :: iorbs_c(*), nfirst_c(*), nlast_c(*), nijbo_c(*)
        integer(c_int) :: nbonds_c(*), ibonds_c(*), lewis_elem_c(*)
        integer(c_int) :: ncf_c(*), nncf_c(*), ncocc_c(*), icocc_c(*)
        real(c_double) :: cocc_c(*)
        integer(c_int) :: nce_c(*), nnce_c(*), ncvir_c(*), icvir_c(*)
        real(c_double) :: cvir_c(*), wall_ms_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_makvec
    end interface
    integer(c_int) :: code
    integer :: resident_plan_ione
    double precision :: wall_ms
#endif
    logical :: trace_requested

    done = .false.
    trace_requested = env_is_one('MOPAC_MOZYME_SCF_STRICT_RESIDENT') .or. &
      env_is_one('MOPAC_MOZYME_SCF_GPU') .or. &
      env_is_one('MOPAC_MOZYME_GPU_STRICT') .or. &
      env_is_one('MOPAC_MOZYME_FULL_SCF_GPU') .or. &
      env_is_one('MOPAC_MOZYME_RESIDENT_SCF') .or. &
      env_is_one('MOPAC_MOZYME_MAKVEC_GPU')
#ifdef GPU
    if (.not. trace_requested) return
    if (.not. (lgpu .or. mozyme_resident_fock_gpu)) then
      write(iw,'(1x,a)') '[MOZYME GPU makvec] status=fallback_cpu reason=gpu_disabled'
      call flush(iw)
      return
    end if
    if (.not. mozyme_gpu_requested) then
      write(iw,'(1x,a)') '[MOZYME GPU makvec] status=fallback_cpu reason=not_requested'
      call flush(iw)
      return
    end if
    if (.not. mozyme_resident_fock_gpu) then
      write(iw,'(1x,a)') '[MOZYME GPU makvec] status=fallback_cpu reason=resident_fock_disabled'
      call flush(iw)
      return
    end if
    if (.not. lijbo) then
      write(iw,'(1x,a)') '[MOZYME GPU makvec] status=fallback_cpu reason=no_nijbo'
      call flush(iw)
      return
    end if
    if (.not. allocated(nijbo)) then
      write(iw,'(1x,a)') '[MOZYME GPU makvec] status=fallback_cpu reason=no_nijbo'
      call flush(iw)
      return
    end if
    if (Lewis_tot <= 0 .or. noccupied <= 0 .or. nvirtual < 0) then
      write(iw,'(1x,a)') '[MOZYME GPU makvec] status=fallback_cpu reason=state_incomplete'
      call flush(iw)
      return
    end if

    resident_plan_ione = 0
    if (id == 0) resident_plan_ione = 1
    if (id == 0) then
      done = mozyme_resident_fock_prepare_plan(resident_fock_plan_full, &
        iorbs, nat, ifact, w, w, 0, kopt, resident_plan_ione, coord, .true.)
    else
      done = mozyme_resident_fock_prepare_plan(resident_fock_plan_full, &
        iorbs, nat, ifact, w, wk, 0, kopt, resident_plan_ione, coord, .true.)
    end if
    if (.not. done) then
      write(iw,'(1x,a)') '[MOZYME GPU makvec] status=fallback_cpu reason=resident_fock_setup'
      call flush(iw)
      return
    end if
    if (.not. mozyme_resident_fock_plan_full_coverage(resident_fock_plan_full)) then
      write(iw,'(1x,a)') '[MOZYME GPU makvec] status=fallback_cpu reason=resident_fock_partial_coverage'
      call flush(iw)
      done = .false.
      return
    end if

    wall_ms = 0.0d0
    code = mopac_cuda_mozyme_makvec( &
      mozyme_c_int_positive_or_zero(numat), &
      mozyme_c_int_positive_or_zero(norbs), &
      mozyme_c_int_positive_or_zero(mpack), &
      mozyme_c_int_positive_or_zero(morb), &
      mozyme_c_int_positive_or_zero(Lewis_tot), &
      mozyme_c_int_nonnegative_or_zero(noccupied), &
      mozyme_c_int_nonnegative_or_zero(nvirtual), &
      mozyme_c_int_nonnegative_or_zero(ipad2), &
      mozyme_c_int_nonnegative_or_zero(ipad4), &
      mozyme_c_int_positive_or_zero(icocc_dim), &
      mozyme_c_int_positive_or_zero(cocc_dim), &
      mozyme_c_int_positive_or_zero(icvir_dim), &
      mozyme_c_int_positive_or_zero(cvir_dim), &
      mozyme_c_int_positive_or_zero(size(ibonds, 1)), pdiag, h, p, f, &
      iorbs, nfirst, nlast, nijbo, nbonds, ibonds, &
      Lewis_elem, ncf, nncf, ncocc, icocc, cocc, nce, nnce, ncvir, &
      icvir, cvir, wall_ms)
    done = (code == 0_c_int)
    if (done) then
      write(iw,'(1x,a,1x,f0.3)') '[MOZYME GPU makvec] status=success wall_ms=', wall_ms
    else
      write(iw,'(1x,a,1x,i0)') '[MOZYME GPU makvec] status=fallback_cpu code=', int(code)
    end if
    call flush(iw)
#else
    if (trace_requested) then
      write(iw,'(1x,a)') '[MOZYME GPU makvec] status=fallback_cpu reason=not_gpu_build'
      call flush(iw)
    end if
#endif
  end function mozyme_gpu_makvec_try

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

end module mozyme_gpu_makvec
