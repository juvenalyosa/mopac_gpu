! Fortran interfaces to GPU density builders that consume device-resident eigenvectors
module gpu_density_interfaces
  use iso_c_binding, only : c_int, c_double, c_size_t, c_bool, c_ptr
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

    subroutine mopac_cuda_density_add_diag(n, value) bind(C,name='mopac_cuda_density_add_diag')
      import :: c_int, c_double
      integer(c_int), value :: n
      real(c_double), value :: value
    end subroutine mopac_cuda_density_add_diag

    subroutine mopac_cuda_register_packed_density(linear, packed) bind(C,name='mopac_cuda_register_packed_density')
      import :: c_int, c_double
      integer(c_int), value :: linear
      real(c_double)        :: packed(*)
    end subroutine mopac_cuda_register_packed_density

    function mopac_cuda_fetch_packed_density(host_ptr, linear) bind(C,name='mopac_cuda_fetch_packed_density') result(ok)
      import :: c_double, c_size_t, c_bool
      real(c_double)        :: host_ptr(*)
      integer(c_size_t), value :: linear
      logical(c_bool)        :: ok
    end function mopac_cuda_fetch_packed_density

    subroutine mopac_cuda_clear_density_cache() bind(C,name='mopac_cuda_clear_density_cache')
    end subroutine mopac_cuda_clear_density_cache
  end interface
end module gpu_density_interfaces
