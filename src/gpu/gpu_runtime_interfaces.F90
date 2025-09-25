module gpu_runtime_interfaces
  use iso_c_binding
  implicit none
  interface
    subroutine mopac_cuda_destroy_resources() bind(C, name='mopac_cuda_destroy_resources')
    end subroutine mopac_cuda_destroy_resources
    subroutine mopac_cuda_set_resident_mode(flag) bind(C, name='mopac_cuda_set_resident_mode')
      import :: c_int
      integer(c_int), value :: flag
    end subroutine mopac_cuda_set_resident_mode
    function mopac_cuda_get_resident_mode() bind(C, name='mopac_cuda_get_resident_mode') result(flag)
      import :: c_int
      integer(c_int) :: flag
    end function mopac_cuda_get_resident_mode
  end interface
end module gpu_runtime_interfaces
