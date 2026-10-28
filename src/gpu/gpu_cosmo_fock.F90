! Molecular Orbital PACkage (MOPAC)
! Copyright 2025-2026 Juvenal Yosa, PhD (juvenal.yosa@gmail.com)
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

! COSMO terms of the Fock matrix (addfck, src/solvation/cosmo.F90) with the dense algebra on the
! device (src/gpu/cosmo_fock.cu): phi = B^T qd, the Cholesky solve of coscl2 and B qsc.  The rest
! of addfck (qdenet, phinet, qscnet, qscat, ediel and f) is done here as addfck does it.
!
! On with the GPU in use for at least MOPAC_GPU_COSMO_MIN surface segments (default 500; smaller
! cavities are faster on the CPU); MOPAC_GPU_COSMO=0 turns it off.  MOPAC_GPU_COSMO_CHECK=1 lets
! addfck do it on the CPU as well and reports the differences (CPU result kept).
module gpu_cosmo_fock
  use iso_c_binding, only : c_int, c_long_long, c_double
  implicit none
  private
  public :: gpu_cosmo_addfck, gpu_cosmo_check_finish

  double precision, allocatable, save :: phi_dev(:), qsc_dev(:), fim_dev(:)
  logical, save :: check_pending = .false.
  integer, save :: check_reports = 0

  interface
    function mopac_cuda_cosmo_fock(lm61_c, nps_c, gen_c, bmat, amat, qd, fepsi_c, phi, qsc, fim) &
        bind(C, name='mopac_cuda_cosmo_fock') result(rc)
      import :: c_int, c_long_long, c_double
      integer(c_int), value :: lm61_c, nps_c
      integer(c_long_long), value :: gen_c
      real(c_double) :: bmat(*), amat(*), qd(*), phi(*), qsc(*), fim(*)
      real(c_double), value :: fepsi_c
      integer(c_int) :: rc
    end function mopac_cuda_cosmo_fock
  end interface

contains

  logical function env_flag(name, default)
    character(len=*), intent(in) :: name
    logical, intent(in) :: default
    character(len=8) :: v
    integer :: st
    v = ' '
    env_flag = default
    call get_environment_variable(name, v, status=st)
    if (st == 0 .and. len_trim(v) > 0) env_flag = (v(1:1) /= '0')
  end function env_flag

  ! addfck(f, p) with the device: .true. when f and the COSMO arrays are done, .false. when addfck
  ! must do it (path off, small cavity, check mode, device failure).
  logical function gpu_cosmo_addfck(f, p) result(done)
    use molkst_C, only : lm61, numat, mpack
    use funcon_C, only : a0, ev
    use cosmo_C, only : ediel, fepsi, nps, amat, bmat, iatsp, phinet, qscnet, qdenet, ipiden, gden, qscat, &
      cosmo_mat_gen
    use mod_vars_cuda, only : lgpu
    double precision, intent(in) :: p(mpack)
    double precision, intent(inout) :: f(mpack)
    integer :: i, iat, im, st, nmin
    character(len=16) :: v
    double precision :: fcon, qsc3
    logical :: check
    done = .false.
    check_pending = .false.
    if (.not. lgpu .or. lm61 < 1 .or. nps < 1) return
    if (env_flag('MOPAC_NOGPU', .false.) .or. .not. env_flag('MOPAC_GPU_COSMO', .true.)) return
    nmin = 500
    v = ' '
    call get_environment_variable('MOPAC_GPU_COSMO_MIN', v, status=st)
    if (st == 0 .and. len_trim(v) > 0) then
      read(v, *, iostat=st) nmin
      if (st /= 0) nmin = 500
    end if
    if (nps < nmin) return
    if (.not. allocated(bmat) .or. .not. allocated(amat)) return
    check = env_flag('MOPAC_GPU_COSMO_CHECK', .false.)
    if (allocated(phi_dev)) then
      if (size(phi_dev) /= nps .or. size(fim_dev) /= lm61) deallocate(phi_dev, qsc_dev, fim_dev)
    end if
    if (.not. allocated(phi_dev)) allocate(phi_dev(nps), qsc_dev(nps), fim_dev(lm61))
    do i = 1, lm61
      qdenet(i, 2) = gden(i) * p(ipiden(i))
      qdenet(i, 3) = qdenet(i, 2) + qdenet(i, 1)
    end do
    if (mopac_cuda_cosmo_fock(int(lm61, c_int), int(nps, c_int), int(cosmo_mat_gen, c_long_long), bmat, &
        amat, qdenet(1, 2), real(fepsi, c_double), phi_dev, qsc_dev, fim_dev) /= 1) return
    if (check) then
      check_pending = .true.
      return
    end if
    fcon = a0 * ev
    do i = 1, numat
      qscat(i) = 0.d0
    end do
    do i = 1, nps
      phinet(i, 2) = phi_dev(i)
      phinet(i, 3) = phinet(i, 1) + phi_dev(i)
    end do
    ediel = 0.d0
    do i = 1, nps
      iat = iatsp(i)
      qscnet(i, 2) = qsc_dev(i)
      qsc3 = qscnet(i, 1) + qscnet(i, 2)
      qscnet(i, 3) = qsc3
      ediel = ediel + qsc3 * phinet(i, 3)
      qscat(iat) = qscat(iat) + qsc3
    end do
    ediel = ediel * fcon / 2
    do i = 1, lm61
      im = ipiden(i)
      f(im) = f(im) - fcon * fim_dev(i)
    end do
    done = .true.
  end function gpu_cosmo_addfck

  ! End of addfck in check mode: the CPU result against the device one.
  subroutine gpu_cosmo_check_finish()
    use cosmo_C, only : nps, phinet, qscnet
    use chanel_C, only : iw
    if (.not. check_pending) return
    check_pending = .false.
    if (check_reports >= 5) return
    check_reports = check_reports + 1
    write(iw, '(1x,a,i0,a,es12.4,a,es12.4,a,es12.4)') '[GPU COSMO] check nps=', nps, &
      ' max|phi_gpu - phi_cpu|=', maxval(abs(phi_dev(1:nps) - phinet(1:nps, 2))), &
      ' max|qsc_gpu - qsc_cpu|=', maxval(abs(qsc_dev(1:nps) - qscnet(1:nps, 2))), &
      ' max|qsc_cpu|=', maxval(abs(qscnet(1:nps, 2)))
  end subroutine gpu_cosmo_check_finish
end module gpu_cosmo_fock
