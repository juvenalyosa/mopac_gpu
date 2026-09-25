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
!  Temperature control for the DRC:
!
!    TEMPERATURE=n   starting velocities drawn from the Maxwell-Boltzmann distribution at n K
!    BUSSI=n         stochastic velocity rescaling thermostat (G. Bussi, D. Donadio and
!                    M. Parrinello, J. Chem. Phys. 126, 014101 (2007)), time constant n fs,
!                    at the temperature given by TEMPERATURE
!    SEED=n          seed for the random numbers
!
      subroutine drc_seed_random (keywrd)
      implicit none
      character (len=*), intent(in) :: keywrd
      integer :: n, k, seed0
      integer, allocatable :: seed(:)
      double precision, external :: reada
      seed0 = 20260925
      if (index(keywrd, " SEED=") /= 0) seed0 = nint(reada(keywrd, index(keywrd, " SEED=")))
      call random_seed (size = n)
      allocate (seed(n))
      do k = 1, n
        seed(k) = seed0 + 7919*k
      end do
      call random_seed (put = seed)
      deallocate (seed)
      end subroutine drc_seed_random

      double precision function drc_gauss ()
!
!  Normal deviate (Box-Muller)
!
      implicit none
      double precision :: u1, u2
      call random_number (u1)
      call random_number (u2)
      u1 = max(u1, 1.d-300)
      drc_gauss = sqrt(-2.d0*log(u1))*cos(6.283185307179586d0*u2)
      end function drc_gauss

      double precision function drc_gamma (shape)
!
!  Gamma deviate, shape >= 1 (G. Marsaglia and W. W. Tsang, ACM TOMS 26, 363 (2000))
!
      implicit none
      double precision, intent(in) :: shape
      double precision :: d, c, x, v, u
      double precision, external :: drc_gauss
      d = shape - 1.d0/3.d0
      c = 1.d0/sqrt(9.d0*d)
      do
        x = drc_gauss()
        v = (1.d0 + c*x)**3
        if (v <= 0.d0) cycle
        call random_number (u)
        if (log(max(u, 1.d-300)) < 0.5d0*x*x + d - d*v + d*log(v)) exit
      end do
      drc_gamma = d*v
      end function drc_gamma

      double precision function drc_bussi_factor (kk, sigma, nn, dt_over_tau)
!
!  Factor by which the velocities are scaled in one step of the Bussi thermostat.
!
!  kk    : current kinetic energy
!  sigma : target kinetic energy, 0.5*nn*k*T (same units as kk)
!  nn    : number of degrees of freedom
!
      implicit none
      double precision, intent(in) :: kk, sigma, dt_over_tau
      integer, intent(in) :: nn
      double precision :: factor, rr, sumn, knew
      double precision, external :: drc_gauss, drc_gamma
      factor = exp(-dt_over_tau)
      rr = drc_gauss()
      if (nn > 2) then
        sumn = 2.d0*drc_gamma(0.5d0*(nn - 1))
      else if (nn == 2) then
        sumn = drc_gauss()**2
      else
        sumn = 0.d0
      end if
      knew = kk + (1.d0 - factor)*(sigma*(sumn + rr**2)/nn - kk) + &
        2.d0*rr*sqrt(kk*sigma/nn*(1.d0 - factor)*factor)
      drc_bussi_factor = sqrt(max(knew, 0.d0)/kk)
      if (rr + sqrt(factor*nn*kk/max((1.d0 - factor)*sigma, 1.d-300)) < 0.d0) &
        drc_bussi_factor = -drc_bussi_factor
      end function drc_bussi_factor

      subroutine drc_maxwell_boltzmann (v, temp, numat, atmass, ndof)
!
!  Velocities (cm/sec, the units of VELOCITY) from the Maxwell-Boltzmann distribution at
!  temp K, with zero net momentum, scaled to a kinetic energy of exactly 0.5*ndof*k*temp.
!
      implicit none
      integer, intent(in) :: numat, ndof
      double precision, intent(in) :: temp, atmass(numat)
      double precision, intent(out) :: v(3*numat)
      double precision, parameter :: r_erg = 8.314462618d7    ! erg/(mol K)
      double precision, parameter :: kb_kcal = 0.0019872041d0 ! kcal/(mol K)
      double precision :: p(3), mtot, ke, target
      integer :: i, j, k
      double precision, external :: drc_gauss
      p = 0.d0
      mtot = 0.d0
      do i = 1, numat
        do j = 1, 3
          k = 3*(i - 1) + j
          v(k) = drc_gauss()*sqrt(r_erg*temp/atmass(i))
          p(j) = p(j) + atmass(i)*v(k)
        end do
        mtot = mtot + atmass(i)
      end do
      ke = 0.d0
      do i = 1, numat
        do j = 1, 3
          k = 3*(i - 1) + j
          v(k) = v(k) - p(j)/mtot
          ke = ke + atmass(i)*v(k)**2
        end do
      end do
      ke = 0.5d0*ke/4.184d10
      target = 0.5d0*ndof*kb_kcal*temp
      if (ke > 0.d0) v = v*sqrt(target/ke)
      end subroutine drc_maxwell_boltzmann
