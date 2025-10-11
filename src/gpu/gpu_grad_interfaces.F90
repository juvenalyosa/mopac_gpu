! Developed by Dr. Juvenal Yosa Reyes, UMCG Groningen, Universidad Simon Bolivar - Barranquilla - Colombia
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
