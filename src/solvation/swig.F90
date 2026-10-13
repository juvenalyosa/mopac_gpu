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
!  Smooth COSMO surface: the switching/Gaussian (SWIG) discretization of
!  C. A. Lange and J. M. Herbert, J. Chem. Phys. 133, 244111 (2010).
!
!  The standard COSMO surface (coscav, coscanz) is built from grid points that are either on
!  or off the surface, segments that collect them, a local frame from the nearest neighbors,
!  and closure segments.  All of these change in steps as the atoms move: the energy jumps
!  by up to 0.03 kcal/mol when an atom of chignolin moves by 0.001 Angstroms, so geometry
!  optimizations in solvent cannot converge, and the analytic gradient is not the derivative
!  of the energy.  With SWIG (keyword SWIG) every atom carries a fixed set of points (dirsm,
!  NSPA points for heavy atoms, about a third of that for hydrogen) on its sphere, radius
!  srad.  Each point k, area a(k) = w(k)*R**2, holds a Gaussian charge, exponent
!  zeta(k) = 4.9/sqrt(a(k)), and is switched off smoothly as it enters another sphere:
!
!     F(k) = Prod(J) (1 - (erf(zeta(k)*(R(J) - r(kJ))) + erf(zeta(k)*(R(J) + r(kJ))))/2)
!
!     A(k,k) = zeta(k)*sqrt(2/pi)/F(k),   A(k,l) = erf(zeta(k,l)*r(kl))/r(kl),
!     zeta(k,l) = zeta(k)*zeta(l)/sqrt(zeta(k)**2 + zeta(l)**2)
!
!  so the energy and its gradient (swig_diegrd) are continuous.  Points with F < 1.d-4 are
!  left out: their charge is of that order (Lange and Herbert use 1.d-8; with 1.d-4 chignolin has
!  20 % fewer points, the energy moves by 0.001 kcal/mol, analytic - finite-difference gradient
!  0.012-0.014 as with 1.d-8; with 1.d-3, 0.009 kcal/mol and 0.038).
!
subroutine swig_surface (coord, numat)
    use cosmo_C, only : nps, lenabc, cosurf, iatsp, srad, n0, dirsm, area, cosvol, arat, &
      nipc, swig_zeta, swig_f
    use common_arrays_C, only : nat
    implicit none
    integer, intent (in) :: numat
    double precision, dimension (3, numat), intent (in) :: coord
    integer :: i, i0, j, jmax, k, nnb
    integer, dimension (:), allocatable :: nb
    double precision :: a, d2, f, r, ri, rcut, t, zeta, zeta_min
    double precision, dimension (3) :: u, x
    double precision, parameter :: zeta0 = 4.9d0, fmin = 1.d-4
    allocate (nb(numat))
    nps = 0
    area = 0.d0
    cosvol = 0.d0
    nipc = 0
    arat(1:numat) = 0.d0
    do i = 1, numat
      ri = srad(i)
      if (nat(i) == 1) then
        i0 = n0(1)
        jmax = n0(2)
      else
        i0 = 0
        jmax = n0(1)
      end if
      !
      !  Atoms that can switch off a point of atom I: beyond srad(J) + 8/zeta, f = 1 exactly
      !
      zeta_min = zeta0 / (ri*Sqrt (Maxval (dirsm(4, i0 + 1:i0 + jmax))))
      nnb = 0
      do j = 1, numat
        if (j == i) cycle
        rcut = ri + srad(j) + 8.d0 / zeta_min
        d2 = (coord(1, j) - coord(1, i))**2 + (coord(2, j) - coord(2, i))**2 + &
          (coord(3, j) - coord(3, i))**2
        if (d2 < rcut**2) then
          nnb = nnb + 1
          nb(nnb) = j
        end if
      end do
      do k = 1, jmax
        u = dirsm(1:3, i0 + k)
        a = dirsm(4, i0 + k) * ri**2
        zeta = zeta0 / Sqrt (a)
        x = coord(:, i) + ri*u
        f = 1.d0
        do j = 1, nnb
          r = Sqrt ((x(1) - coord(1, nb(j)))**2 + (x(2) - coord(2, nb(j)))**2 + &
            (x(3) - coord(3, nb(j)))**2)
          if (r > srad(nb(j)) + 8.d0/zeta) cycle
          t = 1.d0 - 0.5d0*(erf (zeta*(srad(nb(j)) - r)) + erf (zeta*(srad(nb(j)) + r)))
          f = f * t
          if (f < fmin) exit
        end do
        if (f < fmin) cycle
        nps = nps + 1
        if (nps > lenabc) then
          call mopend ("NPS IS GREATER THAN LENABC - USE SMALLER NSPA")
          return
        end if
        cosurf(1:3, nps) = x
        cosurf(4, nps) = a * f
        iatsp(nps) = i
        swig_zeta(nps) = zeta
        swig_f(nps) = f
        area = area + a*f
        cosvol = cosvol + a*f*(u(1)*x(1) + u(2)*x(2) + u(3)*x(3))
        arat(i) = arat(i) + a*f
      end do
    end do
    cosvol = cosvol / 3
    deallocate (nb)
end subroutine swig_surface
subroutine swig_amat
    use cosmo_C, only : nps, amat, nsetf, cosurf, swig_zeta, swig_f
    use funcon_C, only : pi
    implicit none
    integer :: k, l
    double precision :: r, z
    do k = 1, nps
      amat(((k + 1)*k)/2) = swig_zeta(k) * Sqrt (2.d0/pi) / swig_f(k)
      do l = 1, k - 1
        r = Sqrt ((cosurf(1, k) - cosurf(1, l))**2 + (cosurf(2, k) - cosurf(2, l))**2 + &
          (cosurf(3, k) - cosurf(3, l))**2)
        z = swig_zeta(k) * swig_zeta(l) / Sqrt (swig_zeta(k)**2 + swig_zeta(l)**2)
        amat(((k - 1)*k)/2 + l) = erf (z*r) / r
      end do
    end do
    call coscl1 (amat, nsetf, nps)
end subroutine swig_amat
subroutine swig_diegrd (dxyz)
!
!  The A-matrix part of the gradient of the dielectric energy for the SWIG surface:
!  E = (1/2f) q.A.q, so dE/dX = (1/f) Sum(k<l) q(k)q(l) dA(k,l)/dX + (1/2f) Sum(k) q(k)**2 dA(k,k)/dX.
!  Points move rigidly with their atom; A(k,k) depends on the other atoms through F(k).
!  (The part from the potential of the solute at the points is done in diegrd.)
!
    use molkst_C, only : numat
    use cosmo_C, only : nps, iatsp, cosurf, qscnet, fepsi, srad, swig_zeta, swig_f
    use common_arrays_C, only : coord
    use funcon_C, only : ev, a0, fpc_9, pi
    implicit none
    double precision, dimension (3, numat), intent (inout) :: dxyz
    integer :: iak, ial, j, k, l
    double precision :: coef, da, e1, e2, fact, ff, qk, r, rj, t, tp, z, zeta
    double precision, dimension (3) :: xk, xl
    fact = -ev * a0 * fpc_9
    do k = 1, nps
      iak = iatsp(k)
      xk = cosurf(1:3, k)
      qk = qscnet(k, 3)
      do l = 1, k - 1
        ial = iatsp(l)
        if (ial == iak) cycle
        xl = cosurf(1:3, l) - xk
        r = Sqrt (xl(1)**2 + xl(2)**2 + xl(3)**2)
        z = swig_zeta(k) * swig_zeta(l) / Sqrt (swig_zeta(k)**2 + swig_zeta(l)**2)
        !  da = d(erf(z*r)/r)/dr; beyond z*r = 6 erf is 1 to machine precision
        if (z*r > 6.d0) then
          da = -1.d0 / r**2
        else
          da = (2.d0*z/Sqrt (pi) * Exp (-(z*r)**2) * r - erf (z*r)) / r**2
        end if
        ff = -qk * qscnet(l, 3) * fact * da / (r*fepsi)
        dxyz(:, iak) = dxyz(:, iak) - xl * ff
        dxyz(:, ial) = dxyz(:, ial) + xl * ff
      end do
      !
      !  Self term: dA(k,k)/dX = -zeta*sqrt(2/pi)/F**2 dF/dX,
      !  dF/dX(J) = F f'(r)/f(r) (X(J) - x(k))/r, dF/dX(owner) = -Sum(J) dF/dX(J)
      !
      zeta = swig_zeta(k)
      coef = fact / (2.d0*fepsi) * qk**2 * zeta * Sqrt (2.d0/pi) / swig_f(k)
      do j = 1, numat
        if (j == iak) cycle
        rj = srad(j)
        xl = coord(:, j) - xk
        r = Sqrt (xl(1)**2 + xl(2)**2 + xl(3)**2)
        if (r > rj + 8.d0/zeta) cycle
        e1 = Exp (-(zeta*(rj - r))**2)
        e2 = Exp (-(zeta*(rj + r))**2)
        t = 1.d0 - 0.5d0*(erf (zeta*(rj - r)) + erf (zeta*(rj + r)))
        tp = zeta / Sqrt (pi) * (e1 - e2)
        ff = coef * tp / (t*r)
        dxyz(:, j) = dxyz(:, j) + xl * ff
        dxyz(:, iak) = dxyz(:, iak) - xl * ff
      end do
    end do
end subroutine swig_diegrd
