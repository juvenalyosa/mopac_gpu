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

! Fock matrix of a solid (id /= 0, conventional SCF, fock2 mode 2) with the
! two-centre terms on the device (src/gpu/periodic_fock.cu) and the
! one-centre terms (fock1) on the host.  hcore bumps conv_w_generation when it
! rebuilds w / wk, so the device copies are uploaded once per geometry.
!
! MOPAC_GPU_CONV_FOCK=1 enables it.  MOPAC_GPU_FOCK_CHECK=1 builds the device
! Fock matrix, lets fock2 build the CPU one, and reports the difference in
! gpu_conv_fock2_check_finish (the CPU result is kept).
module gpu_conv_fock
  use iso_c_binding, only : c_int, c_long_long, c_double
  implicit none
  private
  public :: gpu_conv_fock2, gpu_conv_fock2_check_finish, conv_w_generation

  integer, save :: conv_w_generation = 0

  integer(c_int), allocatable, save :: pia(:), pib(:), pja(:), pjb(:), sia(:), sib(:)
  integer(c_long_long), allocatable, save :: poff(:), soff(:)
  integer, allocatable, save :: one_off(:)
  integer, save :: tab_numcal = -1, tab_numat = -1, npairs = 0, nself = 0
  double precision, allocatable, save :: f_dev(:)
  logical, save :: check_pending = .false.
  integer, save :: check_reports = 0

  interface
    function mopac_cuda_conv_fock2(mpack_c, n2elec_c, npairs_c, pair_ia, pair_ib, pair_ja, pair_jb, &
        pair_off, nself_c, self_ia, self_ib, self_off, w, wk, gen, ptot, p, f, ms) &
        bind(C, name='mopac_cuda_conv_fock2') result(rc)
      import :: c_int, c_long_long, c_double
      integer(c_long_long), value :: mpack_c, n2elec_c, gen
      integer(c_int), value :: npairs_c, nself_c
      integer(c_int) :: pair_ia(*), pair_ib(*), pair_ja(*), pair_jb(*), self_ia(*), self_ib(*)
      integer(c_long_long) :: pair_off(*), self_off(*)
      real(c_double) :: w(*), wk(*), ptot(*), p(*), f(*)
      real(c_double) :: ms
      integer(c_int) :: rc
    end function mopac_cuda_conv_fock2
  end interface

contains

  logical function env_on(name)
    character(len=*), intent(in) :: name
    character(len=8) :: v
    integer :: st
    v = ' '
    call get_environment_variable(name, v, status=st)
    env_on = (st == 0 .and. len_trim(v) > 0 .and. v(1:1) /= '0')
  end function env_on

  ! Pair table in fock2's order (ii, jj <= ii; w offsets as hcore lays them out).
  subroutine build_tables(numat, nfirst, nlast)
    use molkst_C, only : numcal
    integer, intent(in) :: numat, nfirst(numat), nlast(numat)
    integer :: ii, jj, ni, nj, kk, ns, np
    if (tab_numcal == numcal .and. tab_numat == numat .and. allocated(pia)) return
    if (allocated(pia)) deallocate(pia, pib, pja, pjb, sia, sib, poff, soff, one_off)
    np = (numat*(numat - 1))/2
    allocate(pia(max(1, np)), pib(max(1, np)), pja(max(1, np)), pjb(max(1, np)), poff(max(1, np)))
    allocate(sia(numat), sib(numat), soff(numat), one_off(numat))
    kk = 0
    np = 0
    ns = 0
    do ii = 1, numat
      ni = nlast(ii) - nfirst(ii) + 1
      ni = (ni*(ni + 1))/2
      do jj = 1, ii
        nj = nlast(jj) - nfirst(jj) + 1
        nj = (nj*(nj + 1))/2
        if (jj < ii) then
          np = np + 1
          pia(np) = int(nfirst(ii), c_int)
          pib(np) = int(nlast(ii), c_int)
          pja(np) = int(nfirst(jj), c_int)
          pjb(np) = int(nlast(jj), c_int)
          poff(np) = int(kk, c_long_long)
        else
          ns = ns + 1
          sia(ns) = int(nfirst(ii), c_int)
          sib(ns) = int(nlast(ii), c_int)
          soff(ns) = int(kk, c_long_long)
        end if
        kk = kk + ni*nj
      end do
      one_off(ii) = kk
      kk = kk + ni*ni
    end do
    npairs = np
    nself = ns
    tab_numcal = numcal
    tab_numat = numat
  end subroutine build_tables

  ! Builds f (mode 2: two-centre and one-centre terms, COSMO) for a solid.
  ! Returns .false. when fock2 must do it (path off, check mode, device failure).
  logical function gpu_conv_fock2(f, ptot, p, w, wk, numat, nfirst, nlast) result(done)
    use molkst_C, only : mpack, n2elec, id, mozyme
    use mod_vars_cuda, only : lgpu, gpu_trace
    use cosmo_C, only : useps
    use chanel_C, only : iw
    integer, intent(in) :: numat, nfirst(numat), nlast(numat)
    double precision, intent(inout) :: f(mpack)
    double precision, intent(in) :: ptot(mpack), p(mpack), w(n2elec), wk(n2elec)
    double precision :: ms
    integer :: ii, ia, ib, ilim, kk
    integer(c_int) :: rc
    logical :: check
    done = .false.
    if (.not. lgpu .or. id == 0 .or. mozyme .or. numat < 1) return
    if (.not. env_on('MOPAC_GPU_CONV_FOCK') .or. env_on('MOPAC_NOGPU')) return
    check = env_on('MOPAC_GPU_FOCK_CHECK')
    call build_tables(numat, nfirst, nlast)
    if (check) then
      if (allocated(f_dev)) deallocate(f_dev)
      allocate(f_dev(mpack))
      f_dev = f
    end if
    ms = 0.d0
    if (check) then
      rc = mopac_cuda_conv_fock2(int(mpack, c_long_long), int(n2elec, c_long_long), int(npairs, c_int), &
        pia, pib, pja, pjb, poff, int(nself, c_int), sia, sib, soff, w, wk, &
        int(conv_w_generation, c_long_long), ptot, p, f_dev, ms)
    else
      rc = mopac_cuda_conv_fock2(int(mpack, c_long_long), int(n2elec, c_long_long), int(npairs, c_int), &
        pia, pib, pja, pjb, poff, int(nself, c_int), sia, sib, soff, w, wk, &
        int(conv_w_generation, c_long_long), ptot, p, f, ms)
    end if
    if (rc /= 0) then
      if (gpu_trace) write(iw, '(1x,a,i0)') '[GPU Fock] device failure, CPU used, code=', rc
      check_pending = .false.
      return
    end if
    do ii = 1, numat
      ia = nfirst(ii)
      ib = nlast(ii)
      ilim = ((ib - ia + 1)*(ib - ia + 2))/2
      kk = one_off(ii)
      if (check) then
        call fock1(f_dev, ptot, p, mpack, w(kk + 1), kk, ia, ib, ilim)
      else
        call fock1(f, ptot, p, mpack, w(kk + 1), kk, ia, ib, ilim)
      end if
    end do
    if (check) then
      if (useps) call addfck(f_dev, ptot)
      check_pending = .true.
      return
    end if
    if (useps) call addfck(f, ptot)
    if (gpu_trace) write(iw, '(1x,a,i0,a,i0,a,f10.2)') '[GPU Fock] pairs=', npairs, ' self=', nself, ' ms=', ms
    done = .true.
  end function gpu_conv_fock2

  ! Check mode: f is fock2's CPU result; report its difference from the device one.
  subroutine gpu_conv_fock2_check_finish(f)
    use molkst_C, only : mpack
    use chanel_C, only : iw
    double precision, intent(in) :: f(mpack)
    if (.not. check_pending) return
    check_pending = .false.
    if (check_reports >= 5) return
    check_reports = check_reports + 1
    write(iw, '(1x,a,es12.4,a,es12.4,a)') '[GPU Fock] check max|F_gpu - F_cpu|=', maxval(abs(f_dev - f)), &
      ' max|F|=', maxval(abs(f)), ' eV (CPU result kept)'
  end subroutine gpu_conv_fock2_check_finish

end module gpu_conv_fock
