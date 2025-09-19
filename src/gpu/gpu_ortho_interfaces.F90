! Fortran interfaces for GPU orthogonalization helpers (Phase 2)
module gpu_ortho_interfaces
  use iso_c_binding
  implicit none
  interface
    subroutine mopac_cuda_potrf_upper(n, s, lds, info) bind(C,name='mopac_cuda_potrf_upper')
      import :: c_int, c_double
      integer(c_int), value :: n, lds
      real(c_double)        :: s(lds, n)
      integer(c_int)        :: info
    end subroutine mopac_cuda_potrf_upper

    subroutine mopac_cuda_transform_fock_with_s(n, s, lds, f, ldf, info) bind(C,name='mopac_cuda_transform_fock_with_s')
      import :: c_int, c_double
      integer(c_int), value :: n, lds, ldf
      real(c_double)        :: s(lds, n)
      real(c_double)        :: f(ldf, n)
      integer(c_int)        :: info
    end subroutine mopac_cuda_transform_fock_with_s

    subroutine mopac_cuda_build_c_from_u(n, nocc, u, ldu, uocc, lduocc, cocc, ldc) bind(C,name='mopac_cuda_build_c_from_u')
      import :: c_int, c_double
      integer(c_int), value :: n, nocc, ldu, lduocc, ldc
      real(c_double)        :: u(ldu, n)
      real(c_double)        :: uocc(lduocc, nocc)
      real(c_double)        :: cocc(ldc, nocc)
    end subroutine mopac_cuda_build_c_from_u

    subroutine mopac_cuda_density_from_c(n, nocc, cocc, ldc, p, ldp, scale) bind(C,name='mopac_cuda_density_from_c')
      import :: c_int, c_double
      integer(c_int), value :: n, nocc, ldc, ldp
      real(c_double)        :: cocc(ldc, nocc)
      real(c_double)        :: p(ldp, n)
      real(c_double), value :: scale
    end subroutine mopac_cuda_density_from_c
  end interface
end module gpu_ortho_interfaces

