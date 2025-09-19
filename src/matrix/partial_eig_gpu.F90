module partial_eig_gpu
  use iso_c_binding
#ifdef GPU
  use mopac_cublas_interfaces
#endif
  implicit none
contains

  subroutine block_subspace_smallest(F, n, k, evals, evecs, maxit, tol)
    implicit none
    integer, intent(in) :: n, k, maxit
    double precision, intent(in)  :: tol
    double precision, intent(in)  :: F(n,n)
    double precision, intent(out) :: evals(k)
    double precision, intent(out) :: evecs(n,k)
    ! Workspace
    double precision, allocatable :: X(:,:), AX(:,:), M(:,:), work(:), R(:,:)
    double precision :: nrm, nrm_max
    integer :: i, it, info

    allocate(X(n,k), AX(n,k), M(k,k), R(n,k))
    ! Initialize X as first k columns of identity
    X = 0.d0
    do i=1,k
      if (i <= n) X(i,i) = 1.d0
    end do

    do it = 1, maxit
#ifdef GPU
      call gemm_cublas('N','N', n, k, n, 1.d0, F, n, X, n, 0.d0, AX, n)
#else
      call dgemm('N','N', n, k, n, 1.d0, F, n, X, n, 0.d0, AX, n)
#endif
      ! Form M = X^T * AX (k x k)
      call dsyrk('U','T', k, n, 1.d0, X, n, 0.d0, M, k)
      ! Symmetrize lower
      do i=1,k-1
        call dcopy(k-i, M(i,i+1), k, M(i+1,i), 1)
      end do
      ! Solve small eigenproblem M Y = Y Lam
      allocate(work(3*k))
      call dsyev('V','U', k, M, k, evals, work, 3*k, info)
      deallocate(work)
      ! Update X := X * Y (overwrite X)
#ifdef GPU
      call gemm_cublas('N','N', n, k, k, 1.d0, X, n, M, k, 0.d0, evecs, n)
#else
      call dgemm('N','N', n, k, k, 1.d0, X, n, M, k, 0.d0, evecs, n)
#endif
      X = evecs
      ! Compute residual R = AX - X * diag(evals)
      R = AX
      do i=1,k
        call daxpy(n, -evals(i), X(1,i), 1, R(1,i), 1)
      end do
      ! Check convergence by max 2-norm across columns (Frobenius approx)
      nrm_max = 0.d0
      do i=1,k
        nrm = dnrm2(n, R(1,i), 1)
        if (nrm > nrm_max) nrm_max = nrm
      end do
      if (nrm_max < tol) exit
      ! Orthonormalize X via modified Gram-Schmidt
      call mgs_orthonormalize(X, n, k)
    end do
    evecs = X
    deallocate(X, AX, M, R)
  end subroutine block_subspace_smallest

  subroutine mgs_orthonormalize(Q, n, k)
    implicit none
    integer, intent(in) :: n, k
    double precision, intent(inout) :: Q(n,k)
    integer :: i,j
    double precision :: r
    do i = 1, k
      do j = 1, i-1
        r = ddot(n, Q(1,j), 1, Q(1,i), 1)
        call daxpy(n, -r, Q(1,j), 1, Q(1,i), 1)
      end do
      r = dnrm2(n, Q(1,i), 1)
      if (r > 0.d0) call dscal(n, 1.d0/r, Q(1,i), 1)
    end do
  end subroutine mgs_orthonormalize

end module partial_eig_gpu

