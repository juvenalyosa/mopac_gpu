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

subroutine isitsc (escf, selcon, emin, iemin, iemax, okscf, niter, itrmax)
    use molkst_C, only: iscf
    use MOZYME_C, only : ovmax, energy_diff
    use mozyme_isitsc_state, only: scf1 => isitsc_scf1, &
      escf0 => isitsc_escf0
#ifdef GPU
    use iso_c_binding, only: c_int, c_double
    use chanel_C, only: iw
    use mozyme_gpu_int_utils, only: mozyme_c_int_checked, &
      mozyme_c_int_nonnegative_or_zero, mozyme_c_int_positive_or_zero
    use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required
    use mod_vars_cuda, only: lgpu, mozyme_gpu
#endif
    implicit none
    logical, intent (out) :: okscf
    integer, intent (in) :: itrmax, niter
    integer, intent (inout) :: iemax, iemin
    double precision, intent (in) :: emin, escf, selcon
    integer :: i, iemax1, iemin1
    double precision :: energy_test, fmo_test
#ifdef GPU
    integer(c_int) :: gpu_code, gpu_iemin, gpu_iemax, gpu_scf1
    integer(c_int) :: gpu_okscf, gpu_iscf
    real(c_double) :: gpu_wall_ms

    interface
      function mopac_cuda_mozyme_isitsc(escf_c, selcon_c, emin_c, ovmax_c, &
          energy_diff_c, niter_c, itrmax_c, iemin_c, iemax_c, scf1_c, &
          escf0_c, okscf_c, iscf_c, wall_ms_c) &
          bind(C,name='mopac_cuda_mozyme_isitsc') result(code)
        import :: c_int, c_double
        real(c_double), value :: escf_c, selcon_c, emin_c, ovmax_c
        real(c_double), value :: energy_diff_c
        integer(c_int), value :: niter_c, itrmax_c
        integer(c_int) :: iemin_c, iemax_c, scf1_c, okscf_c, iscf_c
        real(c_double) :: escf0_c(*), wall_ms_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_isitsc
    end interface
#endif
   !
   ! Test the change in energy on successive iterations and the maximum
   ! element of the occ-vir block of the Fock matrix in the LMO basis
   !
    energy_test = selcon
    fmo_test = selcon * 5.0d0

#ifdef GPU
    if (mozyme_isitsc_gpu_enabled()) then
      gpu_iemin = mozyme_c_int_checked(iemin)
      gpu_iemax = mozyme_c_int_checked(iemax)
      gpu_scf1 = merge(1_c_int, 0_c_int, scf1)
      gpu_okscf = 0_c_int
      gpu_iscf = 0_c_int
      gpu_wall_ms = 0.0_c_double
      gpu_code = mopac_cuda_mozyme_isitsc(escf, selcon, emin, ovmax, &
        energy_diff, mozyme_c_int_nonnegative_or_zero(niter), &
        mozyme_c_int_positive_or_zero(itrmax), gpu_iemin, &
        gpu_iemax, gpu_scf1, escf0, gpu_okscf, gpu_iscf, gpu_wall_ms)
      if (gpu_code == 0_c_int) then
        iemin = int(gpu_iemin)
        iemax = int(gpu_iemax)
        scf1 = (gpu_scf1 /= 0_c_int)
        okscf = (gpu_okscf /= 0_c_int)
        if (okscf) iscf = int(gpu_iscf)
        if (mozyme_isitsc_trace()) then
          write(iw,'(1x,a," success code=",i0," okscf=",l1," ms=",f10.3)') &
            '[MOZYME GPU isitsc]', int(gpu_code), okscf, gpu_wall_ms
          call flush(iw)
        end if
        return
      else if (mozyme_isitsc_trace()) then
        write(iw,'(1x,a," fallback_cpu code=",i0)') &
          '[MOZYME GPU isitsc]', int(gpu_code)
        call flush(iw)
      end if
    end if
    if (mozyme_gpu_scf_no_fallback_required()) then
      write(iw,'(1x,a)') &
        '[MOZYME GPU SCF] status=strict_abort reason=strict_isitsc_cpu_fallback'
      call flush(iw)
      error stop 'MOZYME GPU strict isitsc abort'
    end if
#endif

    if (ovmax < fmo_test .and. Abs (energy_diff) < energy_test &
         & .and. scf1 .or. niter > itrmax) then
      okscf = .true.
      iscf = 2
      if (scf1) iscf = 1
    else
      scf1 = (ovmax < fmo_test .and. Abs (energy_diff) < energy_test)
      if (emin /= 0.d0) then
        !*****************************************************************
        !
        !  THE FOLLOWING TESTS ARE INTENDED TO ALLOW A FAST EXIT FROM
        !  ITER IF THE RESULT IS 'GOOD ENOUGH' FOR THE CURRENT STEP IN
        !  THE GEOMETRY OPTIMIZATION
        !
        if (escf < emin) then
          !
          !  THE ENERGY IS LOWER THAN THE PREVIOUS MINIMUM.
          !  NOW CHECK THAT IT IS CONSISTENTLY LOWER.
          !
          iemax = 0
          iemin1 = iemin
          iemin = Min (5, iemin+1)
          if (iemin1 == 5) then
            do i = 2, 5
              escf0(i-1) = escf0(i)
            end do
          end if
          escf0(iemin) = escf
          !
          !  IS THE DIFFERENCE IN ENERGY BETWEEN TWO ITERATIONS LESS THAN 10%
          !  OF THE ENERGY GAIN FOR THIS GEOMETRY RELATIVE TO THE PREVIOUS
          !  MINIMUM.
          !
          if (iemin > 3) then
            do i = 2, iemin
              if (Abs (escf0(i)-escf0(i-1)) > 0.1d0*(emin-escf)) go to 1000
            end do
               !
               ! IS GOOD ENOUGH -- RAPID EXIT
               !
            okscf = .true.
            iscf = 1
            return
          end if
        else
            !
            !  THE ENERGY HAS RISEN ABOVE THAT OF THE PREVIOUS MINIMUM.
            !  WE NEED TO CHECK WHETHER THIS IS A FLUKE OR IS THIS REALLY
            !  A BAD GEOMETRY.
            !
          iemin = 0
          iemax1 = iemax
          iemax = Min (5, iemax+1)
          if (iemax1 == 5) then
            do i = 2, 5
              escf0(i-1) = escf0(i)
            end do
          end if
          escf0(iemax) = escf
          !
          !  IS THE DIFFERENCE IN ENERGY BETWEEN TWO ITERATIONS LESS THAN 10%
          !  OF THE ENERGY LOST FOR THIS GEOMETRY RELATIVE TO THE PREVIOUS
          !  MINIMUM.
          !
          if (iemax > 3) then
            do i = 2, iemax
              if (Abs (escf0(i)-escf0(i-1)) > 0.1d0*(escf-emin)) go to 1000
            end do
               !
               ! IS GOOD ENOUGH -- RAPID EXIT
               !
            okscf = .true.
            iscf = 1
            return
          end if
        end if
      end if
1000  okscf = .false.
    end if
#ifdef GPU
contains
  logical function mozyme_isitsc_gpu_enabled()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_isitsc_gpu_enabled = .false.
    if (.not. (lgpu .and. mozyme_gpu)) return
    env_value = ' '
    call get_environment_variable('MOPAC_MOZYME_ISITSC_GPU', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0) then
      select case (trim(env_value))
      case ('1', 'on', 'ON', 'true', 'TRUE', 'yes', 'YES')
        mozyme_isitsc_gpu_enabled = .true.
      end select
    end if
  end function mozyme_isitsc_gpu_enabled

  logical function mozyme_isitsc_trace()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_isitsc_trace = .false.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_PROFILE', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') &
      mozyme_isitsc_trace = .true.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_VERBOSE', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') &
      mozyme_isitsc_trace = .true.
  end function mozyme_isitsc_trace
#endif
  end subroutine isitsc
  logical function PLS_faulty()
!
!  When some systems are run using MOZYME, the DIAGG1 - DIAGG2 combination fails to converge,
!  and the ovmax converges to a non-zero minimum.  If the job is stopped and a <file>.den
!  is generated, then on restarting the same job, the fault is automatically corrected.
!
!  PLS_faulty detects the conditions of the failure, at run time, and silently writes out
!  the <file>.den, then after reading in the same file, it re-runs the SCF calculation.
!  This corrects the fault.
!
    use MOZYME_C, only : ovmax
    use molkst_C, only : escf
    implicit none
    integer :: loop = -1, j = 0
    integer, parameter :: loop_lim = 6
    double precision :: ovmax_old = 0.d0, array_ovmax(loop_lim), escf_old = 0.d0, array_escf(loop_lim)
    save
    if (loop == -1) then
      array_ovmax = 10.d0
      array_escf = 10.d0
      loop = 0
    else
      loop = loop + 1
      if (loop == loop_lim + 1) then
        do loop = 2, loop_lim
          array_ovmax(loop - 1) = array_ovmax(loop)
          array_escf(loop - 1) = array_escf(loop)
        end do
        loop = loop_lim
      end if
      array_ovmax(loop) = abs(ovmax - ovmax_old)
      ovmax_old = ovmax
      array_escf(loop) = abs(escf - escf_old)
      escf_old = escf
      do j = 1, loop
        if (array_ovmax(j) > 0.01d0) exit
      end do
      if (j <= loop) then
        do j = 1, loop
          if (array_escf(j) > 0.1d0) exit
        end do
      end if
    end if
!
!  Check to see if (a) ovmax (PLS) has converged, and (b) that ovmax is not near zero.
!
    PLS_faulty = (j > loop .and. ovmax > 0.1d0)
    if (j > loop  .and. ovmax > 0.1d0) then
!
!  Deliberately overwrite array_ovmax with different numbers.
!  This prevents the appearance of convergance, for loop_lim iterations.
!
      do j = 1, loop_lim
        array_ovmax(j) = j
        array_escf(j) = j
      end do
    end if
    return
  end function PLS_faulty
