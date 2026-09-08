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

subroutine cnvgz (pnew, p, p1, p2, p3, niter, idiag)
    use molkst_C, only: norbs, mpack
    use MOZYME_C, only : use_three_point_extrap, pmax
#ifdef GPU
    use iso_c_binding, only: c_int, c_double
    use chanel_C, only: iw
    use mozyme_gpu_int_utils, only: mozyme_c_int_nonnegative_or_zero, &
      mozyme_c_int_positive_or_zero
    use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required
    use mod_vars_cuda, only: lgpu, mozyme_gpu
#endif
    implicit none
    integer, intent (in) :: niter
    integer, dimension (norbs), intent (in) :: idiag ! Pointers to diagonal elements
    double precision, dimension (norbs), intent (inout) :: p1, p2, p3
    double precision, dimension (mpack), intent (inout) :: p, pnew
    integer :: i, j
    double precision :: damp, fac, faca, facb, sa
#ifdef GPU
    integer(c_int) :: gpu_code
    real(c_double) :: gpu_pmax, gpu_rms, gpu_wall_ms

    interface
      function mopac_cuda_mozyme_cnvgz(mpack_c, norbs_c, use_three_point_c, &
          niter_c, idiag_c, pnew_c, pold_c, p1_c, p2_c, p3_c, pmax_c, &
          rms_c, wall_ms_c) bind(C,name='mopac_cuda_mozyme_cnvgz') result(code)
        import :: c_int, c_double
        integer(c_int), value :: mpack_c, norbs_c, use_three_point_c, niter_c
        integer(c_int), intent(in) :: idiag_c(*)
        real(c_double) :: pnew_c(*), pold_c(*), p1_c(*), p2_c(*), p3_c(*)
        real(c_double) :: pmax_c, rms_c, wall_ms_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_cnvgz
    end interface
#endif
    intrinsic Abs, Max, Min, Mod, Sign, Sqrt
#ifdef GPU
    if (mozyme_cnvgz_gpu_enabled()) then
      gpu_code = mopac_cuda_mozyme_cnvgz( &
        mozyme_c_int_positive_or_zero(mpack), &
        mozyme_c_int_positive_or_zero(norbs), &
        merge(1_c_int, 0_c_int, use_three_point_extrap), &
        mozyme_c_int_nonnegative_or_zero(niter), &
        idiag, pnew, p, p1, p2, p3, gpu_pmax, gpu_rms, gpu_wall_ms)
      if (gpu_code == 0_c_int) then
        pmax = gpu_pmax
        if (mozyme_cnvgz_trace()) then
          write(iw,'(1x,a," success code=",i0," pmax=",es12.5," rms=",es12.5," ms=",f10.3)') &
            '[MOZYME GPU cnvgz]', int(gpu_code), gpu_pmax, gpu_rms, gpu_wall_ms
          call flush(iw)
        end if
        return
      else if (mozyme_cnvgz_trace()) then
        write(iw,'(1x,a," fallback_cpu code=",i0)') '[MOZYME GPU cnvgz]', int(gpu_code)
        call flush(iw)
      end if
    end if
    if (mozyme_gpu_scf_no_fallback_required()) then
      write(iw,'(1x,a)') &
        '[MOZYME GPU SCF] status=strict_abort reason=strict_cnvgz_cpu_fallback'
      call flush(iw)
      error stop 'MOZYME GPU strict cnvgz abort'
    end if
#endif
   !
   ! Save the diagonal of the current and previous density for later use
   !
    do i = 1, norbs
      j = idiag(i)
      p3(i) = pnew(j)
      p2(i) = p(j)
    end do
!
!   Calculate the maximum and RMS change in the density matrix
!
    pmax = 0.0d0
    do i = 1, mpack
      sa = Abs (pnew(i) - p(i))
      pmax = Max (pmax, sa)
    end do
    if (use_three_point_extrap) then
!
!   Three-point extrapolation in use. Extrapolation is used on
!   every third SCF cycle
!
      if (Mod (niter, 3) == 0) then
        faca = 0.0d0
        facb = 0.0d0
        do i = 1, norbs
          sa = Abs (p3(i) - p2(i))
          faca = faca + sa ** 2
          facb = facb + (p3(i)-2.0d0*p2(i)+p1(i)) ** 2
        end do
!
        if (facb > 0.d0 .and. faca < (100.d0*facb)) then
          fac = Sqrt (faca/facb)
          pnew = pnew + fac * (pnew-p)
        end if
!
      end if
!
!   From iteration 4 on, the change in the diagonal elements is
!   limited to 'damp'
!
      if (niter > 3) then
        damp = 0.05d0
        if (pmax > damp) then
          do i = 1, norbs
            j = idiag(i)
            if (Abs (p3(i)-p2(i)) > damp) then
              pnew(j) = p2(i) + Sign (damp, p3(i)-p2(i))
              pnew(j) = Min (2.0d0, Max (pnew(j), 0.0d0))
            end if
          end do
        end if
      end if
    end if ! three-point extrapolation
!
!  Save the density for use in the next iteration
!
    p1 = p2
    p = pnew
#ifdef GPU
contains
  logical function mozyme_cnvgz_gpu_enabled()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_cnvgz_gpu_enabled = lgpu .and. mozyme_gpu
    if (.not. mozyme_cnvgz_gpu_enabled) return
    env_value = ' '
    call get_environment_variable('MOPAC_MOZYME_CNVGZ_GPU', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0) then
      select case (trim(env_value))
      case ('0', 'off', 'OFF', 'false', 'FALSE', 'no', 'NO')
        mozyme_cnvgz_gpu_enabled = .false.
      case default
        mozyme_cnvgz_gpu_enabled = .true.
      end select
    end if
  end function mozyme_cnvgz_gpu_enabled

  logical function mozyme_cnvgz_trace()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_cnvgz_trace = .false.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_PROFILE', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') &
      mozyme_cnvgz_trace = .true.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_VERBOSE', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') &
      mozyme_cnvgz_trace = .true.
  end function mozyme_cnvgz_trace
#endif
end subroutine cnvgz
