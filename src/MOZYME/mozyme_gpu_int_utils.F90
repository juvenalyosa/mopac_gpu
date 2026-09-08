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

module mozyme_gpu_int_utils
  use iso_c_binding, only: c_int
  implicit none
  private
  public :: mozyme_c_int_checked
  public :: mozyme_c_int_nonnegative_or_zero
  public :: mozyme_c_int_positive_or_zero

contains

  integer(c_int) function mozyme_c_int_checked(value, fallback) &
      result(converted)
    implicit none
    integer, intent(in) :: value
    integer(c_int), intent(in), optional :: fallback

    converted = 0_c_int
    if (present(fallback)) converted = fallback
    if (value >= -huge(converted) .and. value <= huge(converted)) then
      converted = int(value, c_int)
    end if
  end function mozyme_c_int_checked

  integer(c_int) function mozyme_c_int_nonnegative_or_zero(value) &
      result(converted)
    implicit none
    integer, intent(in) :: value

    converted = 0_c_int
    if (value >= 0 .and. value <= huge(converted)) then
      converted = int(value, c_int)
    end if
  end function mozyme_c_int_nonnegative_or_zero

  integer(c_int) function mozyme_c_int_positive_or_zero(value) &
      result(converted)
    implicit none
    integer, intent(in) :: value

    converted = 0_c_int
    if (value > 0 .and. value <= huge(converted)) then
      converted = int(value, c_int)
    end if
  end function mozyme_c_int_positive_or_zero

end module mozyme_gpu_int_utils
