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

subroutine fock1_for_MOZYME (f, ptot, w, kr, iab, ilim)
    use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required
#ifdef GPU
    use mod_vars_cuda, only: lgpu, mozyme_gpu, mozyme_fock_gpu
    use gpu_fock_interfaces, only: mopac_cuda_mozyme_fock1
    use chanel_C, only: iw
#endif
    implicit none
    integer, intent (in) :: iab, ilim
    integer, intent (inout) :: kr
    double precision, dimension ((iab*(iab+1))/2), intent (in) :: ptot
    double precision, dimension ((iab*(iab+1))/2), intent (inout) :: f
    double precision, dimension (ilim, ilim), intent (in) :: w
    external :: mozyme_gpu_strict_abort
!
    integer :: i, ij, ijp, ijw, ikw, im, ip, j, jlw, jm, jp, k, klw, l
#ifdef GPU
    integer :: gpu_info
    integer :: env_len, env_status
    logical :: trace_gpu
    logical, save :: printed_gpu_success = .false.
    logical, save :: printed_gpu_fallback = .false.
    character(len=16) :: gpu_profile_env, gpu_verbose_env
#endif
    double precision :: sum
   ! *********************************************************************
   !
   ! *** COMPUTE THE REMAINING CONTRIBUTIONS TO THE ONE-CENTER ELEMENTS.
   !
   ! *********************************************************************
   !
!   One-center coulomb and exchange terms for atom II.
   !
   !  F(i,j)=F(i,j)+sum(k,l)((PA(k,l)+PB(k,l))*<i,j|k,l>
!                        -(PA(k,l)        )*<i,k|j,l>), k,l on atom II.
    !
    if (mozyme_gpu_scf_no_fallback_required()) then
      call mozyme_gpu_strict_abort('strict_fock1_cpu_fallback', &
        'MOZYME GPU strict resident SCF does not support CPU one-center Fock construction')
      return
    end if
#ifdef GPU
    if (lgpu .and. mozyme_gpu .and. mozyme_fock_gpu) then
      if (.not. printed_gpu_success .and. .not. printed_gpu_fallback) then
        trace_gpu = .false.
        gpu_profile_env = ' '
        gpu_verbose_env = ' '
        call get_environment_variable('MOPAC_GPU_PROFILE', gpu_profile_env, length=env_len, status=env_status)
        if (env_status == 0 .and. env_len > 0 .and. trim(gpu_profile_env) /= '0') trace_gpu = .true.
        call get_environment_variable('MOPAC_GPU_VERBOSE', gpu_verbose_env, length=env_len, status=env_status)
        if (env_status == 0 .and. env_len > 0 .and. trim(gpu_verbose_env) /= '0') trace_gpu = .true.
        if (trace_gpu) then
          write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0)') &
            '[MOZYME GPU fock1]', 'attempt iab=', iab, 'ilim=', ilim
          call flush(iw)
        end if
      end if
      gpu_info = mopac_cuda_mozyme_fock1(iab, ilim, ptot, f, w)
      if (gpu_info == 0) then
        if (.not. printed_gpu_success) then
          trace_gpu = .false.
          gpu_profile_env = ' '
          gpu_verbose_env = ' '
          call get_environment_variable('MOPAC_GPU_PROFILE', gpu_profile_env, length=env_len, status=env_status)
          if (env_status == 0 .and. env_len > 0 .and. trim(gpu_profile_env) /= '0') trace_gpu = .true.
          call get_environment_variable('MOPAC_GPU_VERBOSE', gpu_verbose_env, length=env_len, status=env_status)
          if (env_status == 0 .and. env_len > 0 .and. trim(gpu_verbose_env) /= '0') trace_gpu = .true.
          if (trace_gpu) then
            write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0)') &
              '[MOZYME GPU fock1]', 'success iab=', iab, 'ilim=', ilim
          end if
          printed_gpu_success = .true.
        end if
        kr = kr + ilim ** 2
        return
      else if (.not. printed_gpu_fallback) then
        trace_gpu = .false.
        gpu_profile_env = ' '
        gpu_verbose_env = ' '
        call get_environment_variable('MOPAC_GPU_PROFILE', gpu_profile_env, length=env_len, status=env_status)
        if (env_status == 0 .and. env_len > 0 .and. trim(gpu_profile_env) /= '0') trace_gpu = .true.
        call get_environment_variable('MOPAC_GPU_VERBOSE', gpu_verbose_env, length=env_len, status=env_status)
        if (env_status == 0 .and. env_len > 0 .and. trim(gpu_verbose_env) /= '0') trace_gpu = .true.
        if (trace_gpu) then
          write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
            '[MOZYME GPU fock1]', 'fallback code=', gpu_info, 'iab=', iab, 'ilim=', ilim
        end if
        printed_gpu_fallback = .true.
      end if
    end if
#endif

    do i = 1, iab
      do j = 1, i
         !
         !    Address in 'F'
         !
        ij = (i*(i-1)) / 2 + j
         !
         !    'J' Address IJ in W
         !
        ijw = (i*(i-1)) / 2 + j
        sum = 0.d0
        do k = 1, iab
          do l = 1, iab
            ip = Max (k, l)
            jp = Min (k, l)
               !
               !    Address in 'P'
               !
            ijp = (ip*(ip-1)) / 2 + jp
               !
               !    'J' Address KL in W
               !
            im = Max (k, l)
            jm = Min (k, l)
            klw = (im*(im-1)) / 2 + jm
               !
               !    'K' Address IK in W
               !
            im = Max (k, j)
            jm = Min (k, j)
            ikw = (im*(im-1)) / 2 + jm
               !
               !    'K' Address JL in W
               !
            im = Max (l, i)
            jm = Min (l, i)
            jlw = (im*(im-1)) / 2 + jm
               !
               !   The term itself
               !
            sum = sum + ptot(ijp) * w(ijw, klw) - 0.5d0 * ptot(ijp) * w &
           & (ikw, jlw)
          end do
        end do
        f(ij) = f(ij) + sum
      end do
    end do
    kr = kr + ilim ** 2
end subroutine fock1_for_MOZYME
