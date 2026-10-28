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

! Grimme's D3 dispersion of a solid (id /= 0) summed over all the images within the cutoff.
!
! edisp / ncoord / gdisp took one image per atom pair (the nearest, distance()) and no atom with
! its own images: in a crystal they missed 2-12% of the dispersion (urea 192 atoms 2%, diamond 64
! atoms 12%), and where two images are equally near (a = b cells) the energy had kinks.  Here
!
!   E = -(1/2) sum over i, j, T with (j, T) /= (i, 0), r = |x_j + T - x_i|, of
!       S(r) (s6 C6 f6(r) / r**6 + s8 C8 f8(r) / r**8),
!   CN(i) = sum over the same (j, T) of S(r) / (1 + exp(-16 (rcov_ij / r - 1))),
!
! S = 1 below 15 A (the cutoff of the molecules), 0 beyond 17 A and a C2 polynomial in between.  With
! a hard cutoff the pairs that cross it when the cell is strained make the energy jump, and the
! derivative with respect to Tv (not to the atoms) misses that surface term: ~1.5% of the D3 Tv
! gradient of a urea crystal against central differences.  The gradient is the exact derivative of
! the energy above (also through C6(CN)).  Coordinates and
! translation vectors are in bohr (dftd3 converts them).  The force on the image of j in cell
! T = n1 t1 + n2 t2 + n3 t3 goes into dcart's per-cell layout as (1 - n1 - n2 - n3) times it on
! the central-cell entry of j and n_k times it on the entry of j in the unit cell along t_k: the
! sum over j's entries is the force on j, and deriv's sum of (cell index) x (entry) is n_k times it,
! the derivative with respect to t_k, whatever the size of T.
module dftd3_pbc
  implicit none
  private
  public :: edisp_pbc, gdisp_pbc

  double precision, parameter :: ron_ang = 15.d0, rcut_ang = 17.d0, k1 = 16.d0
  integer, parameter :: max_elem = 94, maxc = 5

  integer :: nimg = 0
  integer, allocatable :: img_n(:,:)          ! (3, nimg) integer translation
  double precision, allocatable :: img_t(:,:) ! (3, nimg) translation, bohr

contains

  ! The translations T that can bring an image of one atom of the cell within the cutoff of another.
  subroutine build_images(rcut)
    use molkst_C, only : numat, id
    use common_arrays_C, only : coord, tvec
    double precision, intent(in) :: rcut
    double precision :: span, h(3), c(3), v, tt(3), r2
    integer :: nmax(3), i, j, a, b, cc, k
    span = 0.d0
    do i = 2, numat
      do j = 1, i - 1
        span = max(span, sum((coord(:,i) - coord(:,j))**2))
      end do
    end do
    span = sqrt(span)
    h = 0.d0
    if (id == 1) then
      h(1) = sqrt(sum(tvec(:,1)**2))
    else if (id == 2) then
      call cross(tvec(:,1), tvec(:,2), c)
      v = sqrt(sum(c**2))
      h(1) = v/sqrt(sum(tvec(:,2)**2))
      h(2) = v/sqrt(sum(tvec(:,1)**2))
    else
      call cross(tvec(:,2), tvec(:,3), c)
      v = abs(sum(tvec(:,1)*c))
      h(1) = v/sqrt(sum(c**2))
      call cross(tvec(:,3), tvec(:,1), c)
      h(2) = v/sqrt(sum(c**2))
      call cross(tvec(:,1), tvec(:,2), c)
      h(3) = v/sqrt(sum(c**2))
    end if
    nmax = 0
    do k = 1, min(id, 3)
      nmax(k) = ceiling((rcut + span)/h(k))
    end do
    if (allocated(img_n)) deallocate(img_n, img_t)
    allocate(img_n(3, (2*nmax(1) + 1)*(2*nmax(2) + 1)*(2*nmax(3) + 1)), &
      img_t(3, (2*nmax(1) + 1)*(2*nmax(2) + 1)*(2*nmax(3) + 1)))
    nimg = 0
    do a = -nmax(1), nmax(1)
      do b = -nmax(2), nmax(2)
        do cc = -nmax(3), nmax(3)
          tt = 0.d0
          if (id >= 1) tt = tt + a*tvec(:,1)
          if (id >= 2) tt = tt + b*tvec(:,2)
          if (id >= 3) tt = tt + cc*tvec(:,3)
          r2 = sum(tt**2)
          if (sqrt(r2) > rcut + span) cycle
          nimg = nimg + 1
          img_n(:, nimg) = [a, b, cc]
          img_t(:, nimg) = tt
        end do
      end do
    end do
  end subroutine build_images

  ! the switching function S(r) and dS/dr (r, ron, roff in bohr)
  pure subroutine switch(r, ron, roff, sw, dsw)
    double precision, intent(in) :: r, ron, roff
    double precision, intent(out) :: sw, dsw
    double precision :: x
    if (r <= ron) then
      sw = 1.d0
      dsw = 0.d0
    else if (r >= roff) then
      sw = 0.d0
      dsw = 0.d0
    else
      x = (r - ron)/(roff - ron)
      sw = 1.d0 - x**3*(10.d0 - 15.d0*x + 6.d0*x**2)
      dsw = -30.d0*x**2*(1.d0 - x)**2/(roff - ron)
    end if
  end subroutine switch

  subroutine cross(a, b, c)
    double precision, intent(in) :: a(3), b(3)
    double precision, intent(out) :: c(3)
    c(1) = a(2)*b(3) - a(3)*b(2)
    c(2) = a(3)*b(1) - a(1)*b(3)
    c(3) = a(1)*b(2) - a(2)*b(1)
  end subroutine cross

  subroutine ncoord_pbc(rcov, cn)
    use molkst_C, only : numat
    use common_arrays_C, only : coord, nat
    use funcon_C, only : a0
    double precision, intent(in) :: rcov(max_elem)
    double precision, intent(out) :: cn(numat)
    integer :: i, j, m
    double precision :: rcut2, d(3), r2, r, xn, sw, dsw
    rcut2 = (rcut_ang/a0)**2
!$omp parallel do schedule(dynamic, 4) default(shared) private(i, j, m, d, r2, r, xn, sw, dsw)
    do i = 1, numat
      xn = 0.d0
      do j = 1, numat
        do m = 1, nimg
          if (j == i .and. all(img_n(:, m) == 0)) cycle
          d = coord(:,j) + img_t(:,m) - coord(:,i)
          r2 = sum(d*d)
          if (r2 > rcut2) cycle
          r = sqrt(r2)
          call switch(r, ron_ang/a0, rcut_ang/a0, sw, dsw)
          xn = xn + sw/(1.d0 + exp(-k1*((rcov(nat(i)) + rcov(nat(j)))/r - 1.d0)))
        end do
      end do
      cn(i) = xn
    end do
!$omp end parallel do
  end subroutine ncoord_pbc

  ! e6, e8 as edisp (positive sums, without s6 / s8)
  subroutine edisp_pbc(c6ab, mxc, r2r4, r0ab, rcov, rs6, rs8, alp6, alp8, e6, e8)
    use molkst_C, only : numat
    use common_arrays_C, only : coord, nat
    use funcon_C, only : a0
    double precision, intent(in) :: c6ab(max_elem, max_elem, maxc, maxc, 3), r2r4(*), &
      r0ab(max_elem, max_elem), rcov(max_elem), rs6, rs8, alp6, alp8
    integer, intent(in) :: mxc(max_elem)
    double precision, intent(out) :: e6, e8
    double precision :: cn(numat), rcut2, d(3), r2, r, rr, damp6, damp8, c6, c8, r6, sw, dsw
    integer :: i, j, m
    call build_images(rcut_ang/a0)
    call ncoord_pbc(rcov, cn)
    rcut2 = (rcut_ang/a0)**2
    e6 = 0.d0
    e8 = 0.d0
!$omp parallel do schedule(dynamic, 4) default(shared) reduction(+:e6, e8) &
!$omp   private(i, j, m, d, r2, r, rr, damp6, damp8, c6, c8, r6, sw, dsw)
    do i = 1, numat
      do j = 1, numat
        do m = 1, nimg
          if (j == i .and. all(img_n(:, m) == 0)) cycle
          d = coord(:,j) + img_t(:,m) - coord(:,i)
          r2 = sum(d*d)
          if (r2 > rcut2) cycle
          r = sqrt(r2)
          rr = r0ab(nat(j), nat(i))/r
          damp6 = 1.d0/(1.d0 + 6.d0*(rs6*rr)**alp6)
          damp8 = 1.d0/(1.d0 + 6.d0*(rs8*rr)**alp8)
          call getc6(maxc, max_elem, c6ab, mxc, nat(i), nat(j), cn(i), cn(j), c6)
          r6 = r2**3
          c8 = 3.d0*c6*r2r4(nat(i))*r2r4(nat(j))
          call switch(r, ron_ang/a0, rcut_ang/a0, sw, dsw)
          e6 = e6 + 0.5d0*sw*c6*damp6/r6
          e8 = e8 + 0.5d0*sw*c8*damp8/(r6*r2)
        end do
      end do
    end do
!$omp end parallel do
  end subroutine edisp_pbc

  ! dxyz_temp(3, numat*l123) += d(s6 e6 + s8 e8)/dx in dcart's layout (Hartree/bohr), for the
  ! energy of edisp_pbc (call edisp_pbc first: it builds the image list)
  subroutine gdisp_pbc(r0ab, rs6, alp6, c6ab, s6, s8, mxc, r2r4, rcov, rs8, alp8, dxyz_temp)
    use molkst_C, only : numat, l123, l1u, l2u, l3u
    use common_arrays_C, only : coord, nat
    use funcon_C, only : a0
    double precision, intent(in) :: r0ab(max_elem, max_elem), rs6, alp6, c6ab(max_elem, max_elem, maxc, maxc, 3), &
      s6, s8, r2r4(max_elem), rcov(max_elem), rs8, alp8
    integer, intent(in) :: mxc(max_elem)
    double precision, intent(inout) :: dxyz_temp(3, numat*l123)
    double precision :: cn(numat), dc6i(numat), rcut2, d(3), r2, r, r0, r42, c6, dc6iji, dc6ijj, &
      r6, r7, r8, r9, t6, t8, damp6, damp8, tmp1, tmp2, drij, rest, rcovij, expterm, dcn, x1, g(3), &
      sw, dsw, damp
    integer :: i, j, m, k, icuc, ient, jent, ucell(3)
    call ncoord_pbc(rcov, cn)
    rcut2 = (rcut_ang/a0)**2
    icuc = (l123 + 1)/2
    ! entries of the unit cells along t1, t2, t3 (0 if the direction is not periodic)
    ucell = 0
    if (l1u > 0) ucell(1) = ((l1u + 1)*(2*l2u + 1) + l2u)*(2*l3u + 1) + l3u + 1
    if (l2u > 0) ucell(2) = (l1u*(2*l2u + 1) + l2u + 1)*(2*l3u + 1) + l3u + 1
    if (l3u > 0) ucell(3) = (l1u*(2*l2u + 1) + l2u)*(2*l3u + 1) + l3u + 1 + 1
    !
    ! dc6i(a) = d(s6 e6 + s8 e8)/dCN(a)
    !
    dc6i = 0.d0
!$omp parallel do schedule(dynamic, 4) default(shared) reduction(+:dc6i) &
!$omp   private(i, j, m, d, r2, r, r0, r42, c6, dc6iji, dc6ijj, r6, r8, t6, t8, damp6, damp8, rest, sw, dsw)
    do i = 1, numat
      do j = 1, numat
        do m = 1, nimg
          if (j == i .and. all(img_n(:, m) == 0)) cycle
          d = coord(:,j) + img_t(:,m) - coord(:,i)
          r2 = sum(d*d)
          if (r2 > rcut2) cycle
          r = sqrt(r2)
          r0 = r0ab(nat(j), nat(i))
          r42 = r2r4(nat(i))*r2r4(nat(j))
          call get_dC6_dCNij(maxc, max_elem, c6ab, mxc(nat(i)), mxc(nat(j)), cn(i), cn(j), nat(i), nat(j), &
            c6, dc6iji, dc6ijj)
          r6 = r2*r2*r2
          r8 = r6*r2
          t6 = (r/(rs6*r0))**(-alp6)
          damp6 = 1.d0/(1.d0 + 6.d0*t6)
          t8 = (r/(rs8*r0))**(-alp8)
          damp8 = 1.d0/(1.d0 + 6.d0*t8)
          call switch(r, ron_ang/a0, rcut_ang/a0, sw, dsw)
          rest = sw*(s6/r6*damp6 + 3.d0*s8*r42/r8*damp8)
          dc6i(i) = dc6i(i) + 0.5d0*rest*dc6iji
          dc6i(j) = dc6i(j) + 0.5d0*rest*dc6ijj
        end do
      end do
    end do
!$omp end parallel do
    !
    ! each ordered (i; j, T): half the radial derivative of the pair term, plus dCN(i)/dr times dc6i(i)
    !
!$omp parallel do schedule(dynamic, 4) default(shared) reduction(+:dxyz_temp) &
!$omp   private(i, j, m, k, d, r2, r, r0, r42, c6, dc6iji, dc6ijj, r6, r7, r8, r9, t6, t8, damp6, damp8, &
!$omp           tmp1, tmp2, drij, rcovij, expterm, dcn, x1, g, ient, jent, sw, dsw, damp)
    do i = 1, numat
      ient = l123*(i - 1) + icuc
      do j = 1, numat
        jent = l123*(j - 1)
        do m = 1, nimg
          if (j == i .and. all(img_n(:, m) == 0)) cycle
          d = coord(:,j) + img_t(:,m) - coord(:,i)
          r2 = sum(d*d)
          if (r2 > rcut2) cycle
          r = sqrt(r2)
          r0 = r0ab(nat(j), nat(i))
          r42 = r2r4(nat(i))*r2r4(nat(j))
          call get_dC6_dCNij(maxc, max_elem, c6ab, mxc(nat(i)), mxc(nat(j)), cn(i), cn(j), nat(i), nat(j), &
            c6, dc6iji, dc6ijj)
          r6 = r2*r2*r2
          r7 = r6*r
          r8 = r6*r2
          r9 = r8*r
          t6 = (r/(rs6*r0))**(-alp6)
          damp6 = 1.d0/(1.d0 + 6.d0*t6)
          t8 = (r/(rs8*r0))**(-alp8)
          damp8 = 1.d0/(1.d0 + 6.d0*t8)
          tmp1 = s6*6.d0*damp6*c6/r7
          tmp2 = s8*6.d0*c6*r42*damp8/r9
          drij = -tmp1 - 4.d0*tmp2 + tmp1*alp6*t6*damp6 + 3.d0*tmp2*alp8*t8*damp8
          ! switched pair term: S P' + S' P, P = s6 C6 f6 / r**6 + s8 C8 f8 / r**8 (C8 = 3 C6 r42)
          call switch(r, ron_ang/a0, rcut_ang/a0, sw, dsw)
          drij = sw*drij + dsw*(s6*c6*damp6/r6 + 3.d0*s8*c6*r42*damp8/r8)
          rcovij = rcov(nat(i)) + rcov(nat(j))
          expterm = exp(-k1*(rcovij/r - 1.d0))
          damp = 1.d0/(1.d0 + expterm)
          dcn = -k1*rcovij*expterm/(r*r*(expterm + 1.d0)*(expterm + 1.d0))
          dcn = sw*dcn + dsw*damp
          x1 = 0.5d0*drij + dcn*dc6i(i)
          g = x1*d/r
          dxyz_temp(:, ient) = dxyz_temp(:, ient) + g
          ! the image of j in cell T: (1 - n1 - n2 - n3) g centrally, n_k g in the unit cell along t_k
          dxyz_temp(:, jent + icuc) = dxyz_temp(:, jent + icuc) - (1 - sum(img_n(:, m)))*g
          do k = 1, 3
            if (img_n(k, m) /= 0) dxyz_temp(:, jent + ucell(k)) = dxyz_temp(:, jent + ucell(k)) - img_n(k, m)*g
          end do
        end do
      end do
    end do
!$omp end parallel do
  end subroutine gdisp_pbc
end module dftd3_pbc
