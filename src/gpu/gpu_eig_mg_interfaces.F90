! Fortran interface for (future) cuSOLVERMg eigensolver integration
module gpu_eig_mg_interfaces
  use iso_c_binding
  implicit none
  interface
    subroutine mopac_cusolvermg_dsyevd(n, a, lda, w, info) bind(C, name='mopac_cusolvermg_dsyevd')
      import :: c_int, c_double
      integer(c_int), value :: n, lda
      real(c_double)        :: a(lda, n)
      real(c_double)        :: w(n)
      integer(c_int)        :: info
    end subroutine mopac_cusolvermg_dsyevd

    logical(c_bool) function mopac_cuda_has_cusolvermg() bind(C, name='mopac_cuda_has_cusolvermg')
      import :: c_bool
    end function mopac_cuda_has_cusolvermg
  end interface
end module gpu_eig_mg_interfaces
