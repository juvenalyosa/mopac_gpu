! Developed by Dr. Juvenal Yosa Reyes, UMCG Groningen, Universidad Simon Bolivar - Barranquilla - Colombia
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
    subroutine mopac_cuda_set_active_stream(stream_ptr) bind(C, name='mopac_cuda_set_active_stream')
      import :: c_ptr
      type(c_ptr), value :: stream_ptr
    end subroutine mopac_cuda_set_active_stream
    subroutine mopac_cuda_clear_active_stream() bind(C, name='mopac_cuda_clear_active_stream')
    end subroutine mopac_cuda_clear_active_stream
    function mopac_cuda_get_fock_device_ptr() bind(C, name='mopac_cuda_get_fock_device_ptr') result(ptr)
      import :: c_ptr
      type(c_ptr) :: ptr
    end function mopac_cuda_get_fock_device_ptr
    function mopac_cuda_get_density_device_ptr() bind(C, name='mopac_cuda_get_density_device_ptr') result(ptr)
      import :: c_ptr
      type(c_ptr) :: ptr
    end function mopac_cuda_get_density_device_ptr
    function mopac_cuda_fetch_fock(host_ptr, linear) bind(C, name='mopac_cuda_fetch_fock') result(ok)
      import :: c_ptr, c_size_t, c_bool, c_double
      type(c_ptr), value :: host_ptr
      integer(c_size_t), value :: linear
      logical(c_bool) :: ok
    end function mopac_cuda_fetch_fock
    function mopac_cuda_fetch_density(host_ptr, n, ld) bind(C, name='mopac_cuda_fetch_density') result(ok)
      import :: c_ptr, c_int, c_bool
      type(c_ptr), value :: host_ptr
      integer(c_int), value :: n, ld
      logical(c_bool) :: ok
    end function mopac_cuda_fetch_density
    function mopac_cuda_fetch_packed_density(host_ptr, linear) bind(C, name='mopac_cuda_fetch_packed_density') result(ok)
      import :: c_ptr, c_size_t, c_bool
      type(c_ptr), value :: host_ptr
      integer(c_size_t), value :: linear
      logical(c_bool) :: ok
    end function mopac_cuda_fetch_packed_density
  end interface
end module gpu_runtime_interfaces
