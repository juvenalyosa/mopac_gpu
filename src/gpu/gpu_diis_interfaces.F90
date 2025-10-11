! Developed by Dr. Juvenal Yosa Reyes, UMCG Groningen, Universidad Simon Bolivar - Barranquilla - Colombia
module gpu_diis_interfaces
  use iso_c_binding
  implicit none
  interface
    subroutine mopac_cuda_diis_init(linear, maxfock) bind(C,name='mopac_cuda_diis_init')
      import :: c_int
      integer(c_int), value :: linear, maxfock
    end subroutine mopac_cuda_diis_init

    subroutine mopac_cuda_diis_store(linear, col, r) bind(C,name='mopac_cuda_diis_store')
      import :: c_int, c_double
      integer(c_int), value :: linear, col
      real(c_double)        :: r(linear)
    end subroutine mopac_cuda_diis_store

    function mopac_cuda_diis_residual_resident(n, linear, col, f_host, p_host, host_out, copy_back) &
      bind(C,name='mopac_cuda_diis_residual_resident') result(ok)
      import :: c_int, c_ptr, c_bool
      integer(c_int), value :: n, linear, col
      type(c_ptr), value :: f_host, p_host, host_out
      integer(c_int), value :: copy_back
      logical(c_bool) :: ok
    end function mopac_cuda_diis_residual_resident

    subroutine mopac_cuda_diis_bcol(linear, nfock, lfock, out) bind(C,name='mopac_cuda_diis_bcol')
      import :: c_int, c_double
      integer(c_int), value :: linear, nfock, lfock
      real(c_double)        :: out(nfock)
    end subroutine mopac_cuda_diis_bcol

    subroutine mopac_cuda_diis_release() bind(C,name='mopac_cuda_diis_release')
    end subroutine mopac_cuda_diis_release
  end interface
end module gpu_diis_interfaces
