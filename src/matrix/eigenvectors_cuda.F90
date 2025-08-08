! Molecular Orbital PACkage (MOPAC)
! Portable CUDA-based eigensolver interop using ISO_C_BINDING

module eigenvectors_cuda
  use iso_c_binding
  implicit none
  private
  public :: eigenvectors_CUDA

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
  end interface

contains

  subroutine eigenvectors_CUDA(eigenvecs, xmat, eigvals, ndim)
    use iso_c_binding
    implicit none
    integer, intent(in) :: ndim
    real(c_double), intent(out) :: eigenvecs(ndim,ndim)
    real(c_double), intent(out) :: eigvals(ndim)
    real(c_double), intent(inout) :: xmat((ndim*(ndim+1))/2)
    integer :: info

    ! Unpack packed upper-triangular into full matrix (column-major)
    call dtpttr('U', ndim, xmat, eigenvecs, ndim, info)
    if (info /= 0) stop 'eigenvectors_CUDA: error in dtpttr'

    ! Compute eigen-decomposition on GPU: eigenvecs overwritten with eigenvectors, eigvals filled
    call mopac_cuda_dsyevd(ndim, eigenvecs, ndim, eigvals, info)

  end subroutine eigenvectors_CUDA

end module eigenvectors_cuda

