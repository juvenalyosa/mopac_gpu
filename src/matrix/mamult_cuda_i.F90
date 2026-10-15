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

module mamult_cuda_i
  implicit none
contains
  subroutine mamult_gpu(a, b, c, ndim, mdim, ifact, beta, igrid, iblock, tt, flag)
    implicit none
    integer, intent(in) :: ndim, mdim, igrid, iblock, flag
    integer, intent(in) :: ifact(*)
    double precision, intent(in) :: a(mdim), b(mdim), beta
    double precision, intent(inout) :: c(mdim)
    double precision, intent(inout) :: tt
    ! This is a stub; should never be called in current code paths.
    stop 'mamult_gpu stub called unexpectedly'
  end subroutine mamult_gpu
end module mamult_cuda_i

