! Molecular Orbital PACkage (MOPAC)
! Portable CUDA-based eigensolver interop using ISO_C_BINDING

module eigenvectors_cuda_mod
  use iso_c_binding
  implicit none
  private
  public :: eigenvectors_CUDA
  public :: eigenvectors_CUDA_keep
  public :: eigenvectors_CUDA_fetch

  interface
    subroutine mopac_cuda_dsyevd(n, a, lda, w, info) bind(C, name="mopac_cuda_dsyevd")
      use iso_c_binding
      implicit none
      integer(c_int), value :: n
      integer(c_int), value :: lda
      real(c_double)        :: a(lda, n)
      real(c_double)        :: w(n)
      integer(c_int)        :: info
    end subroutine mopac_cuda_dsyevd
    subroutine mopac_cuda_dsyevd_keep(n, a, lda, w, info) bind(C, name="mopac_cuda_dsyevd_keep")
      use iso_c_binding
      implicit none
      integer(c_int), value :: n
      integer(c_int), value :: lda
      real(c_double)        :: a(lda, n)
      real(c_double)        :: w(n)
      integer(c_int)        :: info
    end subroutine mopac_cuda_dsyevd_keep
    subroutine mopac_cuda_fetch_eigenvectors(n, a, lda) bind(C, name="mopac_cuda_fetch_eigenvectors")
      use iso_c_binding
      implicit none
      integer(c_int), value :: n
      integer(c_int), value :: lda
      real(c_double)        :: a(lda, n)
    end subroutine mopac_cuda_fetch_eigenvectors
  end interface

contains

  subroutine eigenvectors_CUDA(eigenvecs, xmat, eigvals, ndim)
    implicit none
    integer, intent(in) :: ndim
    real(c_double), intent(out)   :: eigenvecs(:,:)
    real(c_double), intent(out)   :: eigvals(:)
    real(c_double), intent(inout) :: xmat(:)
    integer :: info

    ! Unpack packed upper-triangular into full matrix (column-major)
    call dtpttr('U', ndim, xmat, eigenvecs, ndim, info)
    if (info /= 0) stop 'eigenvectors_CUDA: error in dtpttr'

    ! Compute eigen-decomposition on GPU: eigenvecs overwritten with eigenvectors, eigvals filled
    call mopac_cuda_dsyevd(ndim, eigenvecs, ndim, eigvals, info)

  end subroutine eigenvectors_CUDA

  subroutine eigenvectors_CUDA_keep(eigenvecs, eigvals, ndim)
    use gpu_diag_state, only: gpu_diag_mark
    implicit none
    integer, intent(in) :: ndim
    real(c_double), intent(out)   :: eigenvecs(:,:)
    real(c_double), intent(out)   :: eigvals(:)
    integer :: info
    ! This variant computes eigenvectors on device and keeps them there.
    ! Host-side eigenvecs is left unchanged; caller should use device-aware density build.
    ! Pass a dummy leading array to satisfy interface; it will be ignored.
    call mopac_cuda_dsyevd_keep(ndim, eigenvecs, ndim, eigvals, info)
    call gpu_diag_mark(ndim)
  end subroutine eigenvectors_CUDA_keep

  subroutine eigenvectors_CUDA_fetch(eigenvecs, ndim)
    implicit none
    integer, intent(in) :: ndim
    real(c_double), intent(out)   :: eigenvecs(:,:)
    call mopac_cuda_fetch_eigenvectors(ndim, eigenvecs, ndim)
  end subroutine eigenvectors_CUDA_fetch

end module eigenvectors_cuda_mod
