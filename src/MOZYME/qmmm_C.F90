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

module qmmm_C
!
!  Electrostatic embedding of the QM system in the point charges of an MM environment (qmmm.F90).
!  The charges are set by the keyword MM_CHARGES="file" (lines: x y z q, Angstroms and electrons) or
!  by a program that calls MOPAC (OpenMM through the API), which also reads back mm_grad.
!
  implicit none
  integer :: n_mm = 0                  ! number of MM point charges
  integer :: mm_numcal = -1            ! calculation for which MM_CHARGES was last read
  double precision, dimension(:,:), allocatable :: mm_xyz  ! (3, n_mm), Angstroms
  double precision, dimension(:),   allocatable :: mm_q    ! (n_mm), electrons
  double precision, dimension(:,:), allocatable :: mm_grad ! (3, n_mm), kcal/(mol Angstrom), from qmmm_grad
  double precision :: mm_ron = 0.d0    ! switching of each QM atom - MM charge pair from mm_ron to mm_roff
  double precision :: mm_roff = 0.d0   ! (Angstroms); mm_roff <= 0: no cut-off
  double precision :: e_qmmm = 0.d0    ! QM - MM electrostatic energy at the last SCF, kcal/mol (information)
end module qmmm_C
