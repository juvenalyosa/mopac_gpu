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

subroutine eimp ()
!
!   Store in the s-s location of array p the value of the Fock terms relating
!   atom A and atom B, for all pairs of atoms.  This quantity will be used by
!   the diagonalizer
!
  use molkst_C, only: numat, mpack
  use common_arrays_C, only : p, f
  use MOZYME_C, only : iorbs, lijbo, nijbo
#ifdef GPU
  use iso_c_binding, only: c_int, c_double
  use chanel_C, only: iw
  use mozyme_gpu_int_utils, only: mozyme_c_int_positive_or_zero
  use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required
  use mod_vars_cuda, only: lgpu, mozyme_gpu
#endif
  implicit none
  integer :: i, j, k, l, m
  double precision :: sum
  integer, external :: ijbo
#ifdef GPU
  integer(c_int) :: gpu_code, gpu_updates
  real(c_double) :: gpu_wall_ms

  interface
    function mopac_cuda_mozyme_eimp(mpack_c, numat_c, iorbs_c, nijbo_c, &
        f_c, p_c, updated_pairs_c, wall_ms_c) bind(C,name='mopac_cuda_mozyme_eimp') result(code)
      import :: c_int, c_double
      integer(c_int), value :: mpack_c, numat_c
      integer(c_int), intent(in) :: iorbs_c(*), nijbo_c(*)
      real(c_double), intent(in) :: f_c(*)
      real(c_double) :: p_c(*), wall_ms_c
      integer(c_int) :: updated_pairs_c
      integer(c_int) :: code
    end function mopac_cuda_mozyme_eimp
  end interface

  if (mozyme_eimp_gpu_enabled()) then
    gpu_updates = 0_c_int
    gpu_code = mopac_cuda_mozyme_eimp( &
      mozyme_c_int_positive_or_zero(mpack), &
      mozyme_c_int_positive_or_zero(numat), iorbs, nijbo, f, p, &
      gpu_updates, gpu_wall_ms)
    if (gpu_code == 0_c_int) then
      if (mozyme_eimp_trace()) then
        write(iw,'(1x,a," success code=",i0," pairs=",i0," ms=",f10.3)') &
          '[MOZYME GPU eimp]', int(gpu_code), int(gpu_updates), gpu_wall_ms
        call flush(iw)
      end if
      return
    else if (mozyme_eimp_trace()) then
      write(iw,'(1x,a," fallback_cpu code=",i0)') '[MOZYME GPU eimp]', int(gpu_code)
      call flush(iw)
    end if
  end if
  if (mozyme_gpu_scf_no_fallback_required()) then
    write(iw,'(1x,a)') &
      '[MOZYME GPU SCF] status=strict_abort reason=strict_eimp_cpu_fallback'
    call flush(iw)
    error stop 'MOZYME GPU strict eimp abort'
  end if
#endif
  do i = 1, numat
    do j = 1, i - 1
      k = ijbo (i, j)
      if (k >= 0) then
        l = iorbs(i) * iorbs(j)
        if (l /= 0) then
          l = k + l
          sum = 0.d0
          do m = k + 1, l
            sum = sum + f(m) ** 2
          end do
          p(k+1) = sum
        end if
      end if
    end do
  end do
#ifdef GPU
contains
  logical function mozyme_eimp_gpu_enabled()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_eimp_gpu_enabled = lgpu .and. mozyme_gpu .and. lijbo .and. allocated(nijbo)
    if (.not. mozyme_eimp_gpu_enabled) return
    env_value = ' '
    call get_environment_variable('MOPAC_MOZYME_EIMP_GPU', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0) then
      select case (trim(env_value))
      case ('0', 'off', 'OFF', 'false', 'FALSE', 'no', 'NO')
        mozyme_eimp_gpu_enabled = .false.
      case default
        mozyme_eimp_gpu_enabled = .true.
      end select
    end if
  end function mozyme_eimp_gpu_enabled

  logical function mozyme_eimp_trace()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_eimp_trace = .false.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_PROFILE', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') &
      mozyme_eimp_trace = .true.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_VERBOSE', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') &
      mozyme_eimp_trace = .true.
  end function mozyme_eimp_trace
#endif
end subroutine eimp
