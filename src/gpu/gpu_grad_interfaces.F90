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

module gpu_grad_interfaces
  use iso_c_binding
  implicit none

  type, bind(C) :: gpu_grad_pair
    integer(c_int) :: atom_i
    integer(c_int) :: atom_j
    integer(c_int) :: span_i_first
    integer(c_int) :: span_i_last
    integer(c_int) :: span_j_first
    integer(c_int) :: span_j_last
    integer(c_int) :: image_code
    integer(c_int) :: flags
    real(c_double) :: displacement(3)
    real(c_double) :: distance2
    real(c_double) :: weight
  end type gpu_grad_pair

  interface
    function mopac_cuda_cart_gradient(numat, l123, coord_ptr, grad_ptr, charge_ptr, &
                                      near_pairs_ptr, near_count, far_pairs_ptr, far_count) &
      bind(C, name='mopac_cuda_cart_gradient') result(ok)
      import :: c_int, c_bool, c_ptr
      integer(c_int), value :: numat
      integer(c_int), value :: l123
      type(c_ptr), value :: coord_ptr
      type(c_ptr), value :: grad_ptr
      type(c_ptr), value :: charge_ptr
      type(c_ptr), value :: near_pairs_ptr
      integer(c_int), value :: near_count
      type(c_ptr), value :: far_pairs_ptr
      integer(c_int), value :: far_count
      logical(c_bool) :: ok
    end function mopac_cuda_cart_gradient
  end interface
end module gpu_grad_interfaces
