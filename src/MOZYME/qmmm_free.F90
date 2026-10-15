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

subroutine qmmm_free
!
!  Forget the MM point charges and the external potential of a QM/MM session (qmmm_C), so that they
!  cannot act on later calculations in the same process.
!
    use qmmm_C, only : n_mm, mm_xyz, mm_q, mm_grad, mm_pot, qm_pot, qmmm_session, mm_numcal, e_qmmm
    implicit none
    n_mm = 0
    mm_numcal = -1
    e_qmmm = 0.d0
    qmmm_session = .false.
    if (allocated(mm_xyz)) deallocate (mm_xyz)
    if (allocated(mm_q)) deallocate (mm_q)
    if (allocated(mm_grad)) deallocate (mm_grad)
    if (allocated(mm_pot)) deallocate (mm_pot)
    if (allocated(qm_pot)) deallocate (qm_pot)
end subroutine qmmm_free
