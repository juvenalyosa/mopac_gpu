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

!
!  QM/MM electrostatic embedding for MOZYME (qmmm_C).
!
!  An MM point charge q at x(J) acts on the electrons of QM atom A through the semiempirical
!  two-centre integrals (mu nu|J) of an atom with no additive term (reppd, heavy atom - hydrogen case,
!  with the "hydrogen" a point): in the local frame
!     (ss|J) = ri1,  (s p_sigma|J) = ri2,  (p_sigma p_sigma|J) = ri3,  (p_pi p_pi|J) = ri4,
!  so they keep the Klopman-Ohno damping of atom A and tend to the multipoles of COSMO (get_bvec)
!  at long range.  In the molecular frame, u = (x(J) - x(A))/R:
!     v(ss) = ri1,  v(s p_k) = -ri2 u_k,  v(p_k p_l) = ri4 delta_kl + (ri3 - ri4) u_k u_l,
!     v(d d) = ri1 on the diagonal (as COSMO does for d orbitals),
!  and H(mu nu) on A gets -q v(mu nu).  The core of A sees q through the same (ss|J) as its s electrons,
!  so that a neutral atom has no field at short range (with the gamma of the core-core term, whose
!  additive term po(9) differs from that of the electrons, a water O collapsed onto a QM H in MD:
!  thousands of kcal/mol released).  With mm_roff > 0 every pair is switched off between mm_ron and
!  mm_roff by S(x) = 1 - 10x**3 + 15x**4 - 6x**5.
!  The API can also set qm_pot, a potential at each QM atom that acts on its Mulliken charge (the MM driver
!  uses it for the long-range part of the QM-MM electrostatics, PME).
!
subroutine qmmm_read
!
!  Keyword MM_CHARGES="file": one charge per line, x y z q (Angstroms, electrons).
!
    use qmmm_C, only : n_mm, mm_xyz, mm_q, mm_grad, mm_numcal, qmmm_session
    use molkst_C, only : keywrd_quoted, numcal, line
    use chanel_C, only : iw
    implicit none
    integer :: i, j, k, io, n
    double precision :: x(4)
    character(len=400) :: fname
    i = Index (keywrd_quoted, " MM_CHARGES=")
    if (i == 0) then
      if (.not. qmmm_session) n_mm = 0     ! charges of an earlier job must not leak into this one
      return
    end if
    if (mm_numcal == numcal) return
    mm_numcal = numcal
    j = i + 13                                      ! after MM_CHARGES="
    k = Index (keywrd_quoted(j:), '"') + j - 1
    fname = keywrd_quoted(j:k - 1)
    open (unit=97, file=trim(fname), status="OLD", iostat=io)
    if (io /= 0) then
      call mopend ("MM_CHARGES file '"//trim(fname)//"' not found")
      return
    end if
    n = 0
    do
      read (97, *, iostat=io) x
      if (io /= 0) exit
      n = n + 1
    end do
    rewind (97)
    if (allocated(mm_xyz)) deallocate (mm_xyz, mm_q, mm_grad)
    allocate (mm_xyz(3, n), mm_q(n), mm_grad(3, n))
    do i = 1, n
      read (97, *) x
      mm_xyz(:, i) = x(1:3)
      mm_q(i) = x(4)
    end do
    close (97)
    n_mm = n
    mm_grad = 0.d0
    write (line, '(a,i8,a,f12.4,a)') " QM/MM: ", n, " MM point charges, total charge", Sum (mm_q), &
      " from '"//trim(fname)//"'"
    write (iw, '(a)') trim(line)
end subroutine qmmm_read
subroutine qmmm_pair (ni, d, nv, v, dv, gam, dgam, sw, dsw, want_grad)
!
!  One QM atom (atomic number ni) and one unit point charge at d = x(J) - x(A) (Angstroms):
!  v(k), k = 1..nv (packed lower triangle of the atom's orbitals, eV), dv(:, k) = dv(k)/dd (eV/A),
!  gam (eV) and dgam = dgam/dd for the core, sw / dsw = switching function and its gradient.
!  want_grad = .false.: dv, dgam and dsw are not computed (qmmm_hcore).  Only the elements of v (and dv)
!  that the atom has are set; for a d atom the d off-diagonal elements are zero.
!
    use parameters_C, only : natorb, dd, qq, am, ad, aq
    use funcon_C, only : ev, a0
    use qmmm_C, only : mm_ron, mm_roff
    implicit none
    integer, intent (in) :: ni
    double precision, intent (in) :: d(3)
    logical, intent (in) :: want_grad
    integer, intent (out) :: nv
    double precision, intent (out) :: v(45), dv(3, 45), gam, dgam(3), sw, dsw(3)
    integer :: k, l, kl, nao, c, e, idx(3)
    double precision :: rr, r, u(3), aee, ade, aqe, da, qa, ri1, ri2, ri3, ri4, d1, d2, d3, d4, &
      s1, s2, s3, s4, s5, s6, s7, t, x, ds
    data idx / 2, 3, 4 /
    rr = Sqrt (d(1)**2 + d(2)**2 + d(3)**2)
    u = d / rr
    r = rr / a0
    nao = natorb(ni)
    aee = (0.5d0/am(ni))**2
    s1 = Sqrt (r*r + aee)
    ri1 = ev / s1
    d1 = -ev * r / s1**3 / a0
    !
    !  core: the same (ss|J) as the s electrons
    !
    gam = ri1
    if (want_grad) dgam = d1 * u
    if (nao < 4) then
      nv = 1
      v(1) = ri1
      if (want_grad) dv(:, 1) = d1 * u
    else
      da = dd(ni)
      qa = 2.d0 * qq(ni)
      ade = (0.5d0/ad(ni))**2
      aqe = (0.5d0/aq(ni))**2
      s2 = Sqrt ((r + da)**2 + ade)
      s3 = Sqrt ((r - da)**2 + ade)
      s4 = Sqrt ((r + qa)**2 + aqe)
      s5 = Sqrt ((r - qa)**2 + aqe)
      s6 = Sqrt (r*r + aqe)
      s7 = Sqrt (r*r + aqe + qa*qa)
      ri2 = 0.5d0*ev/s2 - 0.5d0*ev/s3
      ri3 = ri1 + 0.25d0*ev/s4 + 0.25d0*ev/s5 - 0.5d0*ev/s6
      ri4 = ri1 + 0.5d0*ev/s7 - 0.5d0*ev/s6
      nv = 10
      v(1) = ri1
      do c = 1, 3
        k = idx(c)
        v((k*(k - 1))/2 + 1) = -ri2 * u(c)                  ! (p_c, s)
        do e = 1, c
          l = idx(e)
          t = 0.d0
          if (c == e) t = 1.d0
          v((k*(k - 1))/2 + l) = ri4 * t + (ri3 - ri4) * u(c) * u(e)   ! (p_c, p_e)
        end do
      end do
      if (want_grad) then
        d2 = (-0.5d0*ev*(r + da)/s2**3 + 0.5d0*ev*(r - da)/s3**3) / a0
        d3 = d1 + (-0.25d0*ev*(r + qa)/s4**3 - 0.25d0*ev*(r - qa)/s5**3 + 0.5d0*ev*r/s6**3) / a0
        d4 = d1 + (-0.5d0*ev*r/s7**3 + 0.5d0*ev*r/s6**3) / a0
        dv(:, 1) = d1 * u
        do c = 1, 3
          k = idx(c)
          kl = (k*(k - 1))/2 + 1
          dv(:, kl) = -d2 * u(c) * u + ri2 * u(c) * u / rr
          dv(c, kl) = dv(c, kl) - ri2 / rr
          do e = 1, c
            l = idx(e)
            kl = (k*(k - 1))/2 + l
            t = 0.d0
            if (c == e) t = 1.d0
            !  d(u_c u_e)/dd = ((e_c - u_c u) u_e + u_c (e_e - u_e u))/R
            dv(:, kl) = (d4 * t + (d3 - d4) * u(c) * u(e)) * u - (ri3 - ri4) * 2.d0 * u(c) * u(e) * u / rr
            dv(c, kl) = dv(c, kl) + (ri3 - ri4) * u(e) / rr
            dv(e, kl) = dv(e, kl) + (ri3 - ri4) * u(c) / rr
          end do
        end do
      end if
      if (nao == 9) then
        nv = 45
        v(11:45) = 0.d0
        if (want_grad) dv(:, 11:45) = 0.d0
        do k = 5, 9
          kl = (k*(k + 1))/2
          v(kl) = ri1
          if (want_grad) dv(:, kl) = d1 * u
        end do
      end if
    end if
    !
    !  switching
    !
    sw = 1.d0
    dsw = 0.d0
    if (mm_roff > 0.d0) then
      if (rr >= mm_roff) then
        sw = 0.d0
      else if (rr > mm_ron) then
        x = (rr - mm_ron) / (mm_roff - mm_ron)
        sw = 1.d0 - 10.d0*x**3 + 15.d0*x**4 - 6.d0*x**5
        ds = (-30.d0*x**2 + 60.d0*x**3 - 30.d0*x**4) / (mm_roff - mm_ron)
        dsw = ds * u
      end if
    end if
end subroutine qmmm_pair
subroutine qmmm_hcore (h, enuclr)
!
!  MOZYME: add the MM charges to the one-electron matrix (packed with ijbo) and to the core energy (eV)
!
    use qmmm_C, only : n_mm, qm_pot
    use molkst_C, only : numat
    use common_arrays_C, only : nat
    use parameters_C, only : tore, natorb
    use funcon_C, only : fpc_9
    use mozyme_section_timers, only : mozyme_section_timer_begin, mozyme_section_timer_end
    implicit none
    double precision, intent (inout) :: h(*), enuclr
    integer :: i, k, ii
    double precision :: en, timer
    logical :: done
    integer, external :: ijbo
    call qmmm_read
    !
    !  External potential at the QM atoms (kcal/(mol e)): -qm_pot on every diagonal element of the atom, its
    !  core charge in enuclr; the energy is Sum_A (Mulliken charge of A) qm_pot(A)
    !
    if (allocated(qm_pot)) then
      if (size(qm_pot) >= numat) then
        do i = 1, numat
          ii = ijbo (i, i)
          do k = 1, natorb(nat(i))
            h(ii + (k*(k + 1))/2) = h(ii + (k*(k + 1))/2) - qm_pot(i) / fpc_9
          end do
          enuclr = enuclr + tore(nat(i)) * qm_pot(i) / fpc_9
        end do
      end if
    end if
    if (n_mm == 0) return
    call mozyme_section_timer_begin('qmmm_hcore', timer)
    en = 0.d0
    done = .false.
#ifdef GPU
    call qmmm_hcore_gpu (h, en, done)
#endif
    if (.not. done) call qmmm_hcore_pairs (h, en)
    enuclr = enuclr + en
    call mozyme_section_timer_end('qmmm_hcore', timer)
end subroutine qmmm_hcore
subroutine qmmm_hcore_pairs (h, en)
!
!  The MM charges' terms of qmmm_hcore on the CPU: added to h, the core energy (eV) to en
!
    use qmmm_C, only : n_mm, mm_xyz, mm_q
    use molkst_C, only : numat
    use common_arrays_C, only : coord, nat
    use parameters_C, only : tore
    implicit none
    double precision, intent (inout) :: h(*), en
    integer :: i, j, k, ii, nv
    double precision :: v(45), dv(3, 45), gam, dgam(3), sw, dsw(3), d(3)
    integer, external :: ijbo
!   each QM atom owns its diagonal block of h: atoms in parallel, the core energy by reduction
!$omp parallel do schedule(dynamic, 1) default(shared) reduction(+:en) &
!$omp   private(i, j, k, ii, nv, v, dv, gam, dgam, sw, dsw, d)
    do i = 1, numat
      ii = ijbo (i, i)
      do j = 1, n_mm
        d = mm_xyz(:, j) - coord(:, i)
        call qmmm_pair (nat(i), d, nv, v, dv, gam, dgam, sw, dsw, .false.)
        if (sw == 0.d0) cycle
        do k = 1, nv
          h(ii + k) = h(ii + k) - mm_q(j) * sw * v(k)
        end do
        en = en + tore(nat(i)) * mm_q(j) * sw * gam
      end do
    end do
!$omp end parallel do
end subroutine qmmm_hcore_pairs
subroutine qmmm_grad (dxyz)
!
!  MOZYME: gradient of the QM-MM energy (kcal/(mol A)) on the QM atoms (added to dxyz) and on the
!  MM charges (mm_grad).  E(A,J) = q S (Z_A gam - Sum_k g_k P_k v_k), g = 1 on the diagonal, 2 off it.
!
    use qmmm_C, only : n_mm, mm_grad, mm_pot, e_qmmm, qmmm_session
    use molkst_C, only : numat
    use funcon_C, only : fpc_9
    use molkst_C, only : keywrd
    use chanel_C, only : iw
    use mozyme_section_timers, only : mozyme_section_timer_begin, mozyme_section_timer_end
    implicit none
    double precision, intent (inout) :: dxyz(3, numat)
    integer :: j
    double precision :: e, timer
    logical :: done
    if (n_mm == 0) return
    call mozyme_section_timer_begin('qmmm_grad', timer)
    if (allocated(mm_pot)) then
      if (size(mm_pot) < n_mm) deallocate (mm_pot)
    end if
    if (.not. allocated(mm_pot)) allocate (mm_pot(n_mm))
    mm_grad = 0.d0
    mm_pot(:n_mm) = 0.d0
    e = 0.d0
    done = .false.
#ifdef GPU
    call qmmm_grad_gpu (dxyz, e, done)
#endif
    if (.not. done) call qmmm_grad_pairs (dxyz, e)
    e_qmmm = e * fpc_9
    call mozyme_section_timer_end('qmmm_grad', timer)
    if (Index (keywrd, " GRADIENTS") /= 0 .and. n_mm <= 2000 .and. .not. qmmm_session) then
      write (iw, '(/10x,a,f14.6,a)') "QM/MM ELECTROSTATIC ENERGY =", e_qmmm, " KCAL/MOL"
      write (iw, '(10x,a)') "GRADIENT ON THE MM CHARGES (KCAL/MOL/ANGSTROM)"
      do j = 1, n_mm
        write (iw, '(a,i8,3f14.6)') " MM_GRAD", j, mm_grad(:, j)
      end do
    end if
end subroutine qmmm_grad
subroutine qmmm_grad_pairs (dxyz, e)
!
!  qmmm_grad on the CPU: mm_grad and mm_pot set, the QM atoms' gradient added to dxyz, the energy (eV) to e
!
    use qmmm_C, only : n_mm, mm_xyz, mm_q, mm_grad, mm_pot
    use molkst_C, only : numat
    use common_arrays_C, only : coord, nat, p
    use parameters_C, only : tore
    use funcon_C, only : fpc_9
    implicit none
    double precision, intent (inout) :: dxyz(3, numat), e
    integer :: i, j, k, ii, nv
    double precision :: v(45), dv(3, 45), gam, dgam(3), sw, dsw(3), d(3), pv, dpv(3), g(3), wk(45), pk, &
      ej, potj, gj(3)
    integer, allocatable :: iia(:)
    integer, external :: ijbo
    call qmmm_wk (wk)
    allocate (iia(numat))
    do i = 1, numat
      iia(i) = ijbo (i, i)
    end do
!   MM charges in parallel (each owns mm_grad(:, j) and mm_pot(j)); the QM gradient and the energy by
!   reduction
!$omp parallel do schedule(static) default(shared) reduction(+:e, dxyz) &
!$omp   private(i, j, k, ii, nv, v, dv, gam, dgam, sw, dsw, d, pv, dpv, pk, g, ej, potj, gj)
    do j = 1, n_mm
      ej = 0.d0
      potj = 0.d0
      gj = 0.d0
      do i = 1, numat
        ii = iia(i)
        d = mm_xyz(:, j) - coord(:, i)
        call qmmm_pair (nat(i), d, nv, v, dv, gam, dgam, sw, dsw, .true.)
        if (sw == 0.d0) cycle
        pv = 0.d0
        dpv = 0.d0
        do k = 1, nv
          pk = wk(k) * p(ii + k)
          pv = pv + pk * v(k)
          dpv = dpv + pk * dv(:, k)
        end do
        ej = ej + mm_q(j) * sw * (tore(nat(i))*gam - pv)
        potj = potj + sw * (tore(nat(i))*gam - pv) * fpc_9
        !  dE/dd, d = x(J) - x(A)
        g = mm_q(j) * (dsw * (tore(nat(i))*gam - pv) + sw * (tore(nat(i))*dgam - dpv)) * fpc_9
        gj = gj + g
        dxyz(:, i) = dxyz(:, i) - g
      end do
      e = e + ej
      mm_pot(j) = potj
      mm_grad(:, j) = gj
    end do
!$omp end parallel do
    deallocate (iia)
end subroutine qmmm_grad_pairs
subroutine qmmm_wk (wk)
!
!  weights g_k of the packed diagonal block: 1 on the diagonal, 2 off it
!
    implicit none
    double precision, intent (out) :: wk(45)
    integer :: k, m, n
    wk = 0.d0
    n = 0
    do k = 1, 9
      do m = 1, k
        n = n + 1
        if (n > 45) exit
        wk(n) = 2.d0
        if (m == k) wk(n) = 1.d0
      end do
    end do
end subroutine qmmm_wk
#ifdef GPU
subroutine qmmm_atom_data (nao, par)
!
!  Per QM atom, for src/gpu/qmmm_point_charges.cu: orbitals, and am, dd, qq, ad, aq, tore of its element
!
    use iso_c_binding, only : c_int
    use molkst_C, only : numat
    use common_arrays_C, only : nat
    use parameters_C, only : tore, natorb, dd, qq, am, ad, aq
    implicit none
    integer(c_int), intent (out) :: nao(numat)
    double precision, intent (out) :: par(6, numat)
    integer :: i, ni
    do i = 1, numat
      ni = nat(i)
      nao(i) = int(natorb(ni), c_int)
      par(:, i) = (/ am(ni), dd(ni), qq(ni), ad(ni), aq(ni), tore(ni) /)
    end do
end subroutine qmmm_atom_data
subroutine qmmm_hcore_gpu (h, en, done)
!
!  qmmm_hcore_pairs on the device (MOPAC_QMMM_GPU, set by the MOZYME GPU defaults).  The first call also runs
!  the CPU loops on a scratch matrix and keeps the device for later calls only if they agree; done = .false.
!  (CPU loops to be run by the caller) when the device is not used.
!
    use iso_c_binding, only : c_int, c_double
    use qmmm_C, only : n_mm, mm_xyz, mm_q, mm_ron, mm_roff
    use molkst_C, only : numat
    use common_arrays_C, only : coord
    use funcon_C, only : ev, a0
    use chanel_C, only : iw
    use mozyme_gpu_gradient, only : mozyme_gpu_qmmm_enabled
    implicit none
    double precision, intent (inout) :: h(*), en
    logical, intent (out) :: done
    interface
      function mopac_cuda_qmmm_hcore(numat, n_mm, nao, par, coord, mm_xyz, mm_q, ron, roff, ev, a0, hv, en) &
          bind(C, name='mopac_cuda_qmmm_hcore') result(rc)
        import :: c_int, c_double
        integer(c_int), value :: numat, n_mm
        integer(c_int), intent(in) :: nao(*)
        real(c_double), intent(in) :: par(*), coord(*), mm_xyz(*), mm_q(*)
        real(c_double), value :: ron, roff, ev, a0
        real(c_double), intent(out) :: hv(*), en(*)
        integer(c_int) :: rc
      end function mopac_cuda_qmmm_hcore
    end interface
    integer, parameter :: slot_pos(15) = (/ 0, 1, 3, 6, 2, 4, 5, 7, 8, 9, 14, 20, 27, 35, 44 /)
    logical, save :: checked = .false., off = .false.
    integer(c_int), allocatable :: nao(:)
    double precision, allocatable :: par(:,:), hv(:,:), en_atom(:), hs(:)
    double precision :: en_cpu, d, dmax, ref
    integer :: i, k, ii, ns, top
    integer(c_int) :: rc
    integer, external :: ijbo
    done = .false.
    if (off .or. .not. mozyme_gpu_qmmm_enabled()) return
    allocate (nao(numat), par(6, numat), hv(15, numat), en_atom(numat))
    call qmmm_atom_data (nao, par)
    rc = mopac_cuda_qmmm_hcore(int(numat, c_int), int(n_mm, c_int), nao, par, coord, mm_xyz, mm_q, mm_ron, &
      mm_roff, ev, a0, hv, en_atom)
    if (rc /= 0) then
      off = .true.
      write (iw, '(1x,a,i0,a)') '[MOZYME GPU qmmm] hcore failed (code ', rc, '); CPU loops from now on'
      return
    end if
    if (.not. checked) then
      !  the CPU loops on a zero matrix holding the QM atoms' diagonal blocks
      checked = .true.
      top = 0
      do i = 1, numat
        top = max(top, ijbo(i, i) + 45)
      end do
      allocate (hs(top))
      hs = 0.d0
      en_cpu = 0.d0
      call qmmm_hcore_pairs (hs, en_cpu)
      dmax = abs(Sum(en_atom) - en_cpu)
      ref = abs(en_cpu)
      do i = 1, numat
        ii = ijbo(i, i)
        ns = 1
        if (nao(i) >= 4) ns = 10
        if (nao(i) >= 9) ns = 15
        do k = 1, ns
          d = abs(-hv(k, i) - hs(ii + slot_pos(k) + 1))
          dmax = max(dmax, d)
          ref = max(ref, abs(hs(ii + slot_pos(k) + 1)))
        end do
      end do
      deallocate (hs)
      if (dmax > 1.d-9 * max(1.d0, ref)) then
        off = .true.
        write (iw, '(1x,a,es10.2,a)') '[MOZYME GPU qmmm] hcore differs from the CPU loops by ', dmax, &
          ' eV; CPU loops from now on'
        deallocate (nao, par, hv, en_atom)
        return
      end if
      write (iw, '(1x,a,es10.2,a)') '[MOZYME GPU qmmm] hcore as the CPU loops (max diff ', dmax, ' eV)'
    end if
    do i = 1, numat
      ii = ijbo(i, i)
      ns = 1
      if (nao(i) >= 4) ns = 10
      if (nao(i) >= 9) ns = 15
      do k = 1, ns
        h(ii + slot_pos(k) + 1) = h(ii + slot_pos(k) + 1) - hv(k, i)
      end do
      en = en + en_atom(i)
    end do
    deallocate (nao, par, hv, en_atom)
    done = .true.
end subroutine qmmm_hcore_gpu
subroutine qmmm_grad_gpu (dxyz, e, done)
!
!  qmmm_grad_pairs on the device (MOPAC_QMMM_GPU).  The first call runs the CPU loops instead and compares;
!  the device is used from the next call on if they agree.  done = .false.: the caller runs the CPU loops.
!
    use iso_c_binding, only : c_int, c_double
    use qmmm_C, only : n_mm, mm_xyz, mm_q, mm_ron, mm_roff, mm_grad, mm_pot
    use molkst_C, only : numat
    use common_arrays_C, only : coord, p
    use funcon_C, only : ev, a0, fpc_9
    use chanel_C, only : iw
    use mozyme_gpu_gradient, only : mozyme_gpu_qmmm_enabled
    implicit none
    double precision, intent (inout) :: dxyz(3, numat), e
    logical, intent (out) :: done
    interface
      function mopac_cuda_qmmm_grad(numat, n_mm, nao, par, coord, mm_xyz, mm_q, pk, ron, roff, ev, a0, fpc9, &
          gq, mm_grad, mm_pot, e_out) bind(C, name='mopac_cuda_qmmm_grad') result(rc)
        import :: c_int, c_double
        integer(c_int), value :: numat, n_mm
        integer(c_int), intent(in) :: nao(*)
        real(c_double), intent(in) :: par(*), coord(*), mm_xyz(*), mm_q(*), pk(*)
        real(c_double), value :: ron, roff, ev, a0, fpc9
        real(c_double), intent(out) :: gq(*), mm_grad(*), mm_pot(*), e_out
        integer(c_int) :: rc
      end function mopac_cuda_qmmm_grad
    end interface
    integer, parameter :: slot_pos(15) = (/ 0, 1, 3, 6, 2, 4, 5, 7, 8, 9, 14, 20, 27, 35, 44 /)
    logical, save :: checked = .false., off = .false.
    integer(c_int), allocatable :: nao(:)
    double precision, allocatable :: par(:,:), pk(:,:), gq(:,:), g_mm(:,:), pot_mm(:), dx(:,:)
    double precision :: wk(45), e_gpu, e_cpu, dmax, ref
    integer :: i, k, ii, ns
    integer(c_int) :: rc
    integer, external :: ijbo
    done = .false.
    if (off .or. .not. mozyme_gpu_qmmm_enabled()) return
    call qmmm_wk (wk)
    allocate (nao(numat), par(6, numat), pk(15, numat), gq(3, numat), g_mm(3, n_mm), pot_mm(n_mm))
    call qmmm_atom_data (nao, par)
    pk = 0.d0
    do i = 1, numat
      ii = ijbo(i, i)
      ns = 1
      if (nao(i) >= 4) ns = 10
      if (nao(i) >= 9) ns = 15
      do k = 1, ns
        pk(k, i) = wk(slot_pos(k) + 1) * p(ii + slot_pos(k) + 1)
      end do
    end do
    rc = mopac_cuda_qmmm_grad(int(numat, c_int), int(n_mm, c_int), nao, par, coord, mm_xyz, mm_q, pk, mm_ron, &
      mm_roff, ev, a0, fpc_9, gq, g_mm, pot_mm, e_gpu)
    if (rc /= 0) then
      off = .true.
      write (iw, '(1x,a,i0,a)') '[MOZYME GPU qmmm] gradient failed (code ', rc, '); CPU loops from now on'
      deallocate (nao, par, pk, gq, g_mm, pot_mm)
      return
    end if
    if (.not. checked) then
      !  this call: the CPU loops' results, compared with the device's
      checked = .true.
      allocate (dx(3, numat))
      dx = 0.d0
      e_cpu = 0.d0
      call qmmm_grad_pairs (dx, e_cpu)
      dmax = max(maxval(abs(-gq - dx)), maxval(abs(g_mm - mm_grad(:, :n_mm))), &
        maxval(abs(pot_mm - mm_pot(:n_mm))), abs(e_gpu - e_cpu))
      ref = max(maxval(abs(dx)), maxval(abs(mm_grad(:, :n_mm))), maxval(abs(mm_pot(:n_mm))), abs(e_cpu))
      dxyz = dxyz + dx
      e = e + e_cpu
      deallocate (dx)
      if (dmax > 1.d-9 * max(1.d0, ref)) then
        off = .true.
        write (iw, '(1x,a,es10.2,a)') '[MOZYME GPU qmmm] gradient differs from the CPU loops by ', dmax, &
          '; CPU loops from now on'
      else
        write (iw, '(1x,a,es10.2,a)') '[MOZYME GPU qmmm] gradient as the CPU loops (max diff ', dmax, &
          '); device from the next call'
      end if
      deallocate (nao, par, pk, gq, g_mm, pot_mm)
      done = .true.
      return
    end if
    dxyz = dxyz - gq
    mm_grad(:, :n_mm) = g_mm
    mm_pot(:n_mm) = pot_mm
    e = e + e_gpu
    deallocate (nao, par, pk, gq, g_mm, pot_mm)
    done = .true.
end subroutine qmmm_grad_gpu
#endif
