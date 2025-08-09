! Minimal CU-BLAS interfaces used by MOPAC GPU paths
module mopac_cublas_interfaces
  use iso_c_binding
  implicit none
  private
  public :: gemm_cublas, syrk_cublas
  public :: gemm_cublas_2gpu, syrk_cublas_2gpu

  interface gemm_cublas
    subroutine gemm_cublas(tra, trb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
      bind(C, name='call_gemm_cublas')
      use iso_c_binding
      implicit none
      character(c_char), value :: tra, trb
      integer(c_int), value    :: m, n, k, lda, ldb, ldc
      real(c_double)           :: a(lda,*)
      real(c_double)           :: b(ldb,*)
      real(c_double)           :: c(ldc,*)
      real(c_double), value    :: alpha, beta
    end subroutine gemm_cublas
  end interface

  interface syrk_cublas
    subroutine syrk_cublas(uplo, tra, n, k, alpha, a, lda, beta, c, ldc) &
      bind(C, name='call_syrk_cublas')
      use iso_c_binding
      implicit none
      character(c_char), value :: uplo, tra
      integer(c_int), value    :: n, k, lda, ldc
      real(c_double)           :: a(lda,*)
      real(c_double)           :: c(ldc,*)
      real(c_double), value    :: alpha, beta
    end subroutine syrk_cublas
  end interface

  ! Optional 2-GPU specialized outer-product interfaces (k=1)
  interface gemm_cublas_2gpu
    subroutine gemm_cublas_2gpu(tra, trb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
      bind(C, name='call_gemm_cublas_2gpu')
      use iso_c_binding
      implicit none
      character(c_char), value :: tra, trb
      integer(c_int), value    :: m, n, k, lda, ldb, ldc
      real(c_double)           :: a(lda,*)
      real(c_double)           :: b(ldb,*)
      real(c_double)           :: c(ldc,*)
      real(c_double), value    :: alpha, beta
    end subroutine gemm_cublas_2gpu
  end interface

  interface syrk_cublas_2gpu
    subroutine syrk_cublas_2gpu(uplo, tra, n, k, alpha, a, lda, beta, c, ldc) &
      bind(C, name='call_syrk_cublas_2gpu')
      use iso_c_binding
      implicit none
      character(c_char), value :: uplo, tra
      integer(c_int), value    :: n, k, lda, ldc
      real(c_double)           :: a(lda,*)
      real(c_double)           :: c(ldc,*)
      real(c_double), value    :: alpha, beta
    end subroutine syrk_cublas_2gpu
  end interface

end module mopac_cublas_interfaces
