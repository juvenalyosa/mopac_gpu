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

double precision function helecz ()
!
!   Return the total electronic energy,
!   from e = 0.5P(H + F)
!
    use molkst_C, only: mpack, numat
    use MOZYME_C, only : iorbs, lijbo, nijbo
    use common_arrays_C, only : p, h, f
#ifdef GPU
    use iso_c_binding, only: c_int, c_double
    use chanel_C, only: iw
    use mozyme_gpu_int_utils, only: mozyme_c_int_positive_or_zero
    use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required
    use mod_vars_cuda, only: lgpu, mozyme_gpu
#endif
    implicit none
    integer :: i, j, k, l, l1, l2, m
    double precision :: ed, ee
    integer, external :: ijbo
#ifdef GPU
    integer(c_int) :: gpu_code
    real(c_double) :: gpu_energy, gpu_wall_ms

    interface
      function mopac_cuda_mozyme_helecz(mpack_c, numat_c, iorbs_c, nijbo_c, &
          p_c, h_c, f_c, energy_c, wall_ms_c) bind(C,name='mopac_cuda_mozyme_helecz') result(code)
        import :: c_int, c_double
        integer(c_int), value :: mpack_c, numat_c
        integer(c_int), intent(in) :: iorbs_c(*), nijbo_c(*)
        real(c_double), intent(in) :: p_c(*), h_c(*), f_c(*)
        real(c_double) :: energy_c, wall_ms_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_helecz
    end interface

    if (mozyme_helecz_gpu_enabled()) then
      gpu_code = mopac_cuda_mozyme_helecz( &
        mozyme_c_int_positive_or_zero(mpack), &
        mozyme_c_int_positive_or_zero(numat), iorbs, nijbo, p, h, f, &
        gpu_energy, gpu_wall_ms)
      if (gpu_code == 0_c_int) then
        helecz = gpu_energy
        if (mozyme_helecz_trace()) then
          write(iw,'(1x,a," success code=",i0," energy=",es20.10," ms=",f10.3)') &
            '[MOZYME GPU helecz]', int(gpu_code), gpu_energy, gpu_wall_ms
          call flush(iw)
        end if
        return
      else if (mozyme_helecz_trace()) then
        write(iw,'(1x,a," fallback_cpu code=",i0)') '[MOZYME GPU helecz]', int(gpu_code)
        call flush(iw)
      end if
    end if
    if (mozyme_gpu_scf_no_fallback_required()) then
      write(iw,'(1x,a)') &
        '[MOZYME GPU SCF] status=strict_abort reason=strict_helecz_cpu_fallback'
      call flush(iw)
      error stop 'MOZYME GPU strict helecz abort'
    end if
#endif
    ed = 0.0d00
    ee = 0.0d00
    do i = 1, numat
      do j = 1, i - 1
          k = ijbo (i, j)
          if (k >= 0) then
            l = k + iorbs(i) * iorbs(j)
            do m = k + 1, l
              ee = ee + p(m) * (h(m)+f(m))
            end do
          end if
      end do
        k = ijbo (i, i)
        do l1 = 1, iorbs(i)
          do l2 = 1, l1 - 1
            k = k + 1
            ee = ee + p(k) * (h(k)+f(k))
          end do
          k = k + 1
          ed = ed + p(k) * (h(k)+f(k))
        end do
    end do
    ee = ee + .5d00 * ed
    helecz = ee
#ifdef GPU
contains
  logical function mozyme_helecz_gpu_enabled()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_helecz_gpu_enabled = lgpu .and. mozyme_gpu .and. lijbo .and. allocated(nijbo)
    if (.not. mozyme_helecz_gpu_enabled) return
    env_value = ' '
    call get_environment_variable('MOPAC_MOZYME_HELECZ_GPU', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0) then
      select case (trim(env_value))
      case ('0', 'off', 'OFF', 'false', 'FALSE', 'no', 'NO')
        mozyme_helecz_gpu_enabled = .false.
      case default
        mozyme_helecz_gpu_enabled = .true.
      end select
    end if
  end function mozyme_helecz_gpu_enabled

  logical function mozyme_helecz_trace()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_helecz_trace = .false.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_PROFILE', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') &
      mozyme_helecz_trace = .true.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_VERBOSE', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') &
      mozyme_helecz_trace = .true.
  end function mozyme_helecz_trace
#endif
end function helecz
