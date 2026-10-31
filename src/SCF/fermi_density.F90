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

subroutine fermi_density(c, eigs, norbs, nocc, kt, p, mts, efermi)
!
!  RHF density with Fermi-Dirac occupations (keyword SMEAR).
!
!  c(norbs, norbs), eigs(norbs): eigenvectors and eigenvalues (eV, ascending) of the Fock matrix
!  nocc: number of doubly occupied levels at kT = 0 (the electrons are 2*nocc)
!  kt:   electronic temperature times Boltzmann's constant, eV
!
!  On exit p (packed lower triangle) = sum_i f_i c_i c_i^T with f_i = 2/(1 + exp((e_i - mu)/kT)),
!  mu = efermi chosen so that sum_i f_i = 2*nocc; mts = -T*S in eV, S the entropy of the
!  occupations (two spin orbitals per level, g = f/2):  -TS = 2 kT sum_i [g ln g + (1-g) ln(1-g)].
!  The Mermin free energy E[p] - TS is variational in p, so the usual gradient applies.
!
  implicit none
  integer, intent(in) :: norbs, nocc
  double precision, intent(in) :: c(norbs, norbs), eigs(norbs), kt
  double precision, intent(out) :: p((norbs*(norbs + 1))/2), mts, efermi
  double precision, allocatable :: cs(:,:), full(:,:)
  double precision :: f(norbs), lo, hi, mu, n, g, x
  integer :: i, j, k, m, it
!
!  mu by bisection: the electron count is monotonic in mu
!
  lo = eigs(1) - 50.d0*kt - 1.d0
  hi = eigs(norbs) + 50.d0*kt + 1.d0
  do it = 1, 200
    mu = 0.5d0*(lo + hi)
    n = 0.d0
    do i = 1, norbs
      x = (eigs(i) - mu)/kt
      if (x < 40.d0) n = n + 2.d0/(1.d0 + exp(x))
    end do
    if (n > 2.d0*nocc) then
      hi = mu
    else
      lo = mu
    end if
    if (hi - lo < 1.d-13*max(1.d0, abs(mu))) exit
  end do
  efermi = 0.5d0*(lo + hi)
!
!  occupations, -TS, and the levels that hold any electrons
!
  mts = 0.d0
  m = 0
  do i = 1, norbs
    x = (eigs(i) - efermi)/kt
    if (x > 40.d0) then
      f(i) = 0.d0
    else
      f(i) = 2.d0/(1.d0 + exp(x))
    end if
    g = 0.5d0*f(i)
    if (g > 1.d-300 .and. g < 1.d0) mts = mts + 2.d0*kt*g*log(g)
    if (1.d0 - g > 1.d-300 .and. g > 0.d0) mts = mts + 2.d0*kt*(1.d0 - g)*log(1.d0 - g)
    if (f(i) > 1.d-14) m = i
  end do
!
!  p = cs cs^T, cs(:, i) = sqrt(f_i) c(:, i)
!
  allocate(cs(norbs, max(1, m)), full(norbs, norbs))
  do i = 1, m
    cs(:, i) = sqrt(f(i))*c(:, i)
  end do
  call dsyrk('L', 'N', norbs, m, 1.d0, cs, norbs, 0.d0, full, norbs)
  k = 0
  do i = 1, norbs
    do j = 1, i
      k = k + 1
      p(k) = full(i, j)
    end do
  end do
  deallocate(cs, full)
end subroutine fermi_density

