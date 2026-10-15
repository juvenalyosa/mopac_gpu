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

module gpu_diag_state
  implicit none
  logical :: have_device_eigvecs = .false.
  integer :: device_eigvecs_n = 0
contains
  subroutine gpu_diag_mark(n)
    implicit none
    integer, intent(in) :: n
    have_device_eigvecs = .true.
    device_eigvecs_n = n
  end subroutine gpu_diag_mark

  subroutine gpu_diag_clear()
    implicit none
    have_device_eigvecs = .false.
    device_eigvecs_n = 0
  end subroutine gpu_diag_clear
end module gpu_diag_state

