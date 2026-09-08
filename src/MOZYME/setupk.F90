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

subroutine setupk (nocc1)
#ifdef GPU
    use iso_c_binding, only: c_int, c_double
    use chanel_C, only: iw
    use mozyme_gpu_int_utils, only: mozyme_c_int_positive_or_zero
    use mod_vars_cuda, only: lgpu, mozyme_gpu, mozyme_gpu_requested
#endif
    use MOZYME_C, only : icocc, icocc_dim, ncf, nncf, kopt
    use molkst_C, only : numat
    use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required
    implicit none
    integer, intent (in) :: nocc1
    integer :: i, j, k, l
    external :: mozyme_gpu_strict_abort
#ifdef GPU
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
    integer(c_int) :: gpu_code
    real(c_double) :: gpu_wall_ms
    logical :: trace_gpu_setupk
    integer :: env_len, env_status
    character(len=16) :: env_value
#endif
   !********************************************************************
   !
   !   SETUPK DETERMINES WHICH ATOMS NEED TO BE CONSIDERED IN
   !   RE-CONSTRUCTING THE FOCK MATRIX.
   !
   !   THE SET OF ATOMS TO BE USED IS PUT INTO KOPT.  ALL ATOMS USED IN
   !   THE LMOs IN THE SCF CALCULATION ARE PUT INTO KOPT.
   !
   !   (Note:  Perhaps a better test would be 'All atoms that make a
   !   significant contribution to the LMOs in the SCF calculation?'
   !
   !********************************************************************
#ifdef GPU
    if (lgpu .and. (mozyme_gpu_requested .or. mozyme_gpu) .and. &
        nocc1 > 0 .and. nocc1 <= size(ncf) .and. &
        nocc1 <= size(nncf) .and. icocc_dim > 0 .and. &
        icocc_dim <= size(icocc) .and. numat > 0 .and. &
        numat <= size(kopt)) then
      trace_gpu_setupk = .false.
      env_value = ' '
      call get_environment_variable('MOPAC_GPU_PROFILE', env_value, &
        length=env_len, status=env_status)
      if (env_status == 0 .and. env_len > 0 .and. &
          trim(env_value) /= '0') trace_gpu_setupk = .true.
      env_value = ' '
      call get_environment_variable('MOPAC_GPU_VERBOSE', env_value, &
        length=env_len, status=env_status)
      if (env_status == 0 .and. env_len > 0 .and. &
          trim(env_value) /= '0') trace_gpu_setupk = .true.

      gpu_wall_ms = 0.0_c_double
      gpu_code = mopac_cuda_mozyme_setupk( &
        mozyme_c_int_positive_or_zero(numat), &
        mozyme_c_int_positive_or_zero(nocc1), &
        mozyme_c_int_positive_or_zero(icocc_dim), ncf, nncf, icocc, &
        kopt, gpu_wall_ms)
      if (gpu_code == 0_c_int) then
        if (trace_gpu_setupk) then
          write(iw,'(1x,a," success code=",i0," ms=",f10.3)') &
            '[MOZYME GPU setupk]', int(gpu_code), gpu_wall_ms
        end if
        return
      else if (trace_gpu_setupk) then
        write(iw,'(1x,a," fallback_cpu code=",i0)') &
          '[MOZYME GPU setupk]', int(gpu_code)
      end if
    end if
#endif
    if (mozyme_gpu_scf_no_fallback_required()) then
      call mozyme_gpu_strict_abort('strict_setupk_cpu_fallback', &
        'MOZYME GPU strict resident SCF does not support CPU setupk')
      return
    end if
    kopt = 0
    do i = 1, nocc1
      j = nncf(i)
      do k = 1, ncf(i)
        kopt(icocc(k+j)) = 1
      end do
    end do
!
!   If atom i is to be used in constructing the Fock matrix, then kopt(i) = 1
!   otherwise kopt(i) = 0
!
!   Now compress kopt, so that all the atoms used for the Fock matrix are in
!   order.
!
    l = 0
    do i = 1, numat
      if (kopt(i) == 1) then
        l = l + 1
        kopt(l) = i
      end if
    end do
!
!  Force a zero in after the last atom.  This can be used in finding the end of the atom list.
!
    if (l /= numat) kopt(l+1) = 0
end subroutine setupk
