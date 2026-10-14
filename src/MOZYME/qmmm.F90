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
!  and H(mu nu) on A gets -q v(mu nu).  The core of A sees q through eV/sqrt(r**2 + po(9,A)**2), the
!  gamma of the MOPAC core-core term.  With mm_roff > 0 every pair is switched off between mm_ron and
!  mm_roff by S(x) = 1 - 10x**3 + 15x**4 - 6x**5.
!
subroutine qmmm_read
!
!  Keyword MM_CHARGES="file": one charge per line, x y z q (Angstroms, electrons).
!
    use qmmm_C, only : n_mm, mm_xyz, mm_q, mm_grad, mm_numcal
    use molkst_C, only : keywrd_quoted, numcal, line
    use chanel_C, only : iw
    implicit none
    integer :: i, j, k, io, n
    double precision :: x(4)
    character(len=400) :: fname
    i = Index (keywrd_quoted, " MM_CHARGES=")
    if (i == 0 .or. mm_numcal == numcal) return
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
subroutine qmmm_pair (ni, d, nv, v, dv, gam, dgam, sw, dsw)
!
!  One QM atom (atomic number ni) and one unit point charge at d = x(J) - x(A) (Angstroms):
!  v(k), k = 1..nv (packed lower triangle of the atom's orbitals, eV), dv(:, k) = dv(k)/dd (eV/A),
!  gam (eV) and dgam = dgam/dd for the core, sw / dsw = switching function and its gradient.
!
    use parameters_C, only : natorb, dd, qq, am, ad, aq, po
    use funcon_C, only : ev, a0
    use qmmm_C, only : mm_ron, mm_roff
    implicit none
    integer, intent (in) :: ni
    double precision, intent (in) :: d(3)
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
    v = 0.d0
    dv = 0.d0
    !
    !  core: gamma with the additive term of the core-core interaction
    !
    gam = ev / Sqrt (r*r + po(9, ni)**2)
    dgam = -ev * r / (r*r + po(9, ni)**2)**1.5d0 / a0 * u
    aee = (0.5d0/am(ni))**2
    s1 = Sqrt (r*r + aee)
    ri1 = ev / s1
    d1 = -ev * r / s1**3 / a0
    if (nao < 4) then
      nv = 1
      v(1) = ri1
      dv(:, 1) = d1 * u
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
      d2 = (-0.5d0*ev*(r + da)/s2**3 + 0.5d0*ev*(r - da)/s3**3) / a0
      d3 = d1 + (-0.25d0*ev*(r + qa)/s4**3 - 0.25d0*ev*(r - qa)/s5**3 + 0.5d0*ev*r/s6**3) / a0
      d4 = d1 + (-0.5d0*ev*r/s7**3 + 0.5d0*ev*r/s6**3) / a0
      nv = 10
      v(1) = ri1
      dv(:, 1) = d1 * u
      do c = 1, 3
        k = idx(c)
        kl = (k*(k - 1))/2 + 1                     ! (p_c, s)
        v(kl) = -ri2 * u(c)
        dv(:, kl) = -d2 * u(c) * u
        dv(c, kl) = dv(c, kl) - ri2 / rr
        dv(:, kl) = dv(:, kl) + ri2 * u(c) * u / rr
        do e = 1, c
          l = idx(e)
          kl = (k*(k - 1))/2 + l                   ! (p_c, p_e)
          t = 0.d0
          if (c == e) t = 1.d0
          v(kl) = ri4 * t + (ri3 - ri4) * u(c) * u(e)
          dv(:, kl) = (d4 * t + (d3 - d4) * u(c) * u(e)) * u
          !  d(u_c u_e)/dd = ((e_c - u_c u) u_e + u_c (e_e - u_e u))/R
          dv(:, kl) = dv(:, kl) - (ri3 - ri4) * 2.d0 * u(c) * u(e) * u / rr
          dv(c, kl) = dv(c, kl) + (ri3 - ri4) * u(e) / rr
          dv(e, kl) = dv(e, kl) + (ri3 - ri4) * u(c) / rr
        end do
      end do
      if (nao == 9) then
        nv = 45
        do k = 5, 9
          kl = (k*(k + 1))/2
          v(kl) = ri1
          dv(:, kl) = d1 * u
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
    use qmmm_C, only : n_mm, mm_xyz, mm_q
    use molkst_C, only : numat
    use common_arrays_C, only : coord, nat
    use parameters_C, only : tore
    implicit none
    double precision, intent (inout) :: h(*), enuclr
    integer :: i, j, k, ii, nv
    double precision :: v(45), dv(3, 45), gam, dgam(3), sw, dsw(3), d(3)
    integer, external :: ijbo
    call qmmm_read
    if (n_mm == 0) return
    do i = 1, numat
      ii = ijbo (i, i)
      do j = 1, n_mm
        d = mm_xyz(:, j) - coord(:, i)
        call qmmm_pair (nat(i), d, nv, v, dv, gam, dgam, sw, dsw)
        if (sw == 0.d0) cycle
        do k = 1, nv
          h(ii + k) = h(ii + k) - mm_q(j) * sw * v(k)
        end do
        enuclr = enuclr + tore(nat(i)) * mm_q(j) * sw * gam
      end do
    end do
end subroutine qmmm_hcore
subroutine qmmm_grad (dxyz)
!
!  MOZYME: gradient of the QM-MM energy (kcal/(mol A)) on the QM atoms (added to dxyz) and on the
!  MM charges (mm_grad).  E(A,J) = q S (Z_A gam - Sum_k g_k P_k v_k), g = 1 on the diagonal, 2 off it.
!
    use qmmm_C, only : n_mm, mm_xyz, mm_q, mm_grad, e_qmmm
    use molkst_C, only : numat
    use common_arrays_C, only : coord, nat, p
    use parameters_C, only : tore
    use funcon_C, only : fpc_9
    use molkst_C, only : keywrd
    use chanel_C, only : iw
    implicit none
    double precision, intent (inout) :: dxyz(3, numat)
    integer :: i, j, k, ii, nv, m, n
    double precision :: v(45), dv(3, 45), gam, dgam(3), sw, dsw(3), d(3), pv, dpv(3), g(3), wk(45), e, pk
    integer, external :: ijbo
    if (n_mm == 0) return
    mm_grad = 0.d0
    e = 0.d0
    do i = 1, numat
      ii = ijbo (i, i)
      !
      !  weights g_k P_k of the atom's diagonal block
      !
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
      do j = 1, n_mm
        d = mm_xyz(:, j) - coord(:, i)
        call qmmm_pair (nat(i), d, nv, v, dv, gam, dgam, sw, dsw)
        if (sw == 0.d0) cycle
        pv = 0.d0
        dpv = 0.d0
        do k = 1, nv
          pk = wk(k) * p(ii + k)
          pv = pv + pk * v(k)
          dpv = dpv + pk * dv(:, k)
        end do
        e = e + mm_q(j) * sw * (tore(nat(i))*gam - pv)
        !  dE/dd, d = x(J) - x(A)
        g = mm_q(j) * (dsw * (tore(nat(i))*gam - pv) + sw * (tore(nat(i))*dgam - dpv)) * fpc_9
        mm_grad(:, j) = mm_grad(:, j) + g
        dxyz(:, i) = dxyz(:, i) - g
      end do
    end do
    e_qmmm = e * fpc_9
    if (Index (keywrd, " GRADIENTS") /= 0 .and. n_mm <= 2000) then
      write (iw, '(/10x,a,f14.6,a)') "QM/MM ELECTROSTATIC ENERGY =", e_qmmm, " KCAL/MOL"
      write (iw, '(10x,a)') "GRADIENT ON THE MM CHARGES (KCAL/MOL/ANGSTROM)"
      do j = 1, n_mm
        write (iw, '(a,i8,3f14.6)') " MM_GRAD", j, mm_grad(:, j)
      end do
    end if
end subroutine qmmm_grad
