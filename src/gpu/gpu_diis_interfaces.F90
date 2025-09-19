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

    subroutine mopac_cuda_diis_bcol(linear, nfock, lfock, out) bind(C,name='mopac_cuda_diis_bcol')
      import :: c_int, c_double
      integer(c_int), value :: linear, nfock, lfock
      real(c_double)        :: out(nfock)
    end subroutine mopac_cuda_diis_bcol

    subroutine mopac_cuda_diis_release() bind(C,name='mopac_cuda_diis_release')
    end subroutine mopac_cuda_diis_release
  end interface
end module gpu_diis_interfaces

