module gpu_grad_interfaces
  use iso_c_binding
  implicit none
  interface
    function mopac_cuda_cart_gradient(numat, l123, coord_ptr, grad_ptr, charge_ptr) &
      bind(C, name='mopac_cuda_cart_gradient') result(ok)
      import :: c_int, c_bool, c_ptr
      integer(c_int), value :: numat
      integer(c_int), value :: l123
      type(c_ptr), value :: coord_ptr
      type(c_ptr), value :: grad_ptr
      type(c_ptr), value :: charge_ptr
      logical(c_bool) :: ok
    end function mopac_cuda_cart_gradient
  end interface
end module gpu_grad_interfaces
