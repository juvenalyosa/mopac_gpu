! Developed by Dr. Juvenal Yosa Reyes, UMCG Groningen, Universidad Simon Bolivar - Barranquilla - Colombia
module gpu_bmat_interfaces
  use iso_c_binding
  implicit none
  interface
    subroutine mopac_cuda_bcol_from_residuals(linear, nfock, fppf, lfock, out) bind(C,name='mopac_cuda_bcol_from_residuals')
      import :: c_int, c_double
      integer(c_int), value :: linear, nfock, lfock
      real(c_double)        :: fppf(linear*nfock)
      real(c_double)        :: out(nfock)
    end subroutine mopac_cuda_bcol_from_residuals

    subroutine mopac_cuda_bfull_from_host(linear, nfock, fppf, bout) bind(C,name='mopac_cuda_bfull_from_host')
      import :: c_int, c_double
      integer(c_int), value :: linear, nfock
      real(c_double)        :: fppf(linear*nfock)
      real(c_double)        :: bout(nfock, nfock)
    end subroutine mopac_cuda_bfull_from_host

    subroutine mopac_cuda_bfull_from_device(linear, nfock, bout) bind(C,name='mopac_cuda_bfull_from_device')
      import :: c_int, c_double
      integer(c_int), value :: linear, nfock
      real(c_double)        :: bout(nfock, nfock)
    end subroutine mopac_cuda_bfull_from_device
  end interface
end module gpu_bmat_interfaces
