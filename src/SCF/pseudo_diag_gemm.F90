! Molecular Orbital PACkage (MOPAC)
! Copyright 2025-2026 Juvenal Yosa, PhD
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

subroutine pseudo_diag_gemm(f, c, eigs, n, nocc, xmax, gmax)
!
!  Pseudo-diagonalization (Stewart, Csaszar, Pulay, J. Comp. Chem. 3, 227 (1982)) with level-3
!  BLAS: one first-order rotation of all occupied-virtual pairs at once, instead of diag's 2 x 2
!  rotations one by one.
!
!  f:    Fock matrix, packed lower triangle (the same matrix the full diagonalization would see)
!  c:    c(n, n), orthonormal; columns 1..nocc occupied, nocc+1..n virtual.  Updated in place.
!  eigs: eigenvalues of the last full diagonalization (eV), used for the denominators, unchanged
!  xmax: largest rotation allowed (the step is scaled down to it)
!  gmax: on exit, the largest |F(occ, virt)| over the MOs before the rotation (eV)
!
!  G = Co^T F Cv; X(i,a) = -G(i,a) / max(e_a - e_i, dmin); Co' = Co + Cv X^T, Cv' = Cv - Co X.
!  Co'^T Cv' = -X + X = 0 exactly, so only each block is re-orthonormalized (Cholesky), which
!  keeps the occupied span and hence the density.
!
  implicit none
  integer, intent(in) :: n, nocc
  double precision, intent(in) :: f((n*(n + 1))/2), eigs(n), xmax
  double precision, intent(inout) :: c(n, n)
  double precision, intent(out) :: gmax
  double precision, parameter :: dmin = 0.5d0
  double precision, allocatable :: fu(:,:), w(:,:), g(:,:), co(:,:), cv(:,:), s(:,:)
  double precision :: scale, xm
  integer :: i, j, k, nv, info
  nv = n - nocc
  gmax = 0.d0
  if (nocc < 1 .or. nv < 1) return
  allocate(fu(n, n), w(n, nv), g(nocc, nv), co(n, nocc), cv(n, nv))
  k = 0
  do i = 1, n
    do j = 1, i
      k = k + 1
      fu(i, j) = f(k)
      fu(j, i) = f(k)
    end do
  end do
  co = c(:, 1:nocc)
  cv = c(:, nocc + 1:n)
!  G = Co^T (F Cv)
  call dgemm('N', 'N', n, nv, n, 1.d0, fu, n, cv, n, 0.d0, w, n)
  call dgemm('T', 'N', nocc, nv, n, 1.d0, co, n, w, n, 0.d0, g, nocc)
  deallocate(fu, w)
  gmax = maxval(abs(g))
!  X, in place of G
  xm = 0.d0
  do j = 1, nv
    do i = 1, nocc
      g(i, j) = -g(i, j)/max(eigs(nocc + j) - eigs(i), dmin)
      xm = max(xm, abs(g(i, j)))
    end do
  end do
  scale = 1.d0
  if (xm > xmax) scale = xmax/xm
!  Co' = Co + Cv X^T,  Cv' = Cv - Co X
  c(:, 1:nocc) = co
  call dgemm('N', 'T', n, nocc, nv, scale, cv, n, g, nocc, 1.d0, c(1, 1), n)
  c(:, nocc + 1:n) = cv
  call dgemm('N', 'N', n, nv, nocc, -scale, co, n, g, nocc, 1.d0, c(1, nocc + 1), n)
  deallocate(g, co, cv)
!  re-orthonormalize each block: C <- C L^-T, L L^T = C^T C
  allocate(s(nocc, nocc))
  call dsyrk('L', 'T', nocc, n, 1.d0, c(1, 1), n, 0.d0, s, nocc)
  call dpotrf('L', nocc, s, nocc, info)
  if (info == 0) call dtrsm('R', 'L', 'T', 'N', n, nocc, 1.d0, s, nocc, c(1, 1), n)
  deallocate(s)
  allocate(s(nv, nv))
  call dsyrk('L', 'T', nv, n, 1.d0, c(1, nocc + 1), n, 0.d0, s, nv)
  call dpotrf('L', nv, s, nv, info)
  if (info == 0) call dtrsm('R', 'L', 'T', 'N', n, nv, 1.d0, s, nv, c(1, nocc + 1), n)
  deallocate(s)
end subroutine pseudo_diag_gemm
