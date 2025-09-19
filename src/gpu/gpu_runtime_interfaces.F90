module gpu_runtime_interfaces
  use iso_c_binding
  implicit none
  interface
    subroutine mopac_cuda_destroy_resources() bind(C, name='mopac_cuda_destroy_resources')
    end subroutine mopac_cuda_destroy_resources
  end interface
end module gpu_runtime_interfaces

