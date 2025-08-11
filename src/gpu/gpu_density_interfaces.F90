! Fortran interfaces to GPU density builders that consume device-resident eigenvectors
module gpu_density_interfaces
  use iso_c_binding
  implicit none
  interface
    subroutine mopac_cuda_density_from_dev_syrk(n, ndubl, alpha, c_full, ldc) &
      bind(C,name='mopac_cuda_density_from_dev_syrk')
      import :: c_int, c_double
      integer(c_int), value :: n, ndubl, ldc
      real(c_double), value :: alpha
      real(c_double)        :: c_full(ldc, n)
    end subroutine mopac_cuda_density_from_dev_syrk

    subroutine mopac_cuda_density_from_dev_gemm(n, nl2, nu2, nl1, nu1, sign, frac, xmat, ldx) &
      bind(C,name='mopac_cuda_density_from_dev_gemm')
      import :: c_int, c_double
      integer(c_int), value :: n, nl2, nu2, nl1, nu1, ldx
      real(c_double), value :: sign, frac
      real(c_double)        :: xmat(ldx, n)
    end subroutine mopac_cuda_density_from_dev_gemm
  end interface
end module gpu_density_interfaces

