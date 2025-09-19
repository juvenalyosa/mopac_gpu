module purify_gpu
  use iso_c_binding
  use overlap_build, only: build_overlap_packed, unpack_upper_to_full
#ifdef GPU
  use mopac_cublas_interfaces
  use gpu_ortho_interfaces
#endif
  implicit none
contains

  subroutine purify_density_from_fock(fpack, n, nelecs, pp, tol, maxit)
    implicit none
    integer, intent(in) :: n, nelecs, maxit
    double precision, intent(in) :: tol
    double precision, intent(in)  :: fpack((n*(n+1))/2)
    double precision, intent(out) :: pp((n*(n+1))/2)
    double precision, allocatable :: Sfull(:,:), Ffull(:,:), U(:,:), Uinv(:,:), Fp(:,:), X(:,:), X2(:,:), T(:,:), Pao(:,:)
    double precision, allocatable :: Spack(:)
    integer :: i, info, k, stat_env
    double precision :: trX, target, delta, sigma, err, lam_min, lam_max, mu, beta
    double precision :: one, zero
    character(len=32) :: env
    logical :: use_gpu
    double precision :: tol_l
    integer :: maxit_l
    one = 1.d0; zero = 0.d0

    allocate(Sfull(n,n), Ffull(n,n), U(n,n), Uinv(n,n), Fp(n,n), X(n,n), X2(n,n), T(n,n), Pao(n,n))
    allocate(Spack((n*(n+1))/2))

    ! Build overlap S and factor S = U^T U (upper)
    call build_overlap_packed(Spack)
    call unpack_upper_to_full(n, Spack, Sfull)
    U = Sfull
    call dpotrf('U', n, U, n, info)
    if (info /= 0) stop 'purify: Cholesky failed on S'

    ! Unpack Fock to full
    call dtpttr('U', n, fpack, Ffull, n, info)
    if (info /= 0) stop 'purify: dtpttr on F failed'

    ! Compute Uinv = U^{-1}
    Uinv = U
    call dtrtri('U', 'N', n, Uinv, n, info)
    if (info /= 0) stop 'purify: dtrtri failed on U'

    ! Optional: use GPU helper to transform F to F' via TRSM if requested
#ifdef GPU
    stat_env = 1 ; env = '' ; use_gpu = .false.
    call get_environment_variable('MOPAC_PURIFY_GPU', env, status=stat_env)
    if (stat_env == 0) use_gpu = (trim(adjustl(env)) /= '')
    if (use_gpu) then
      ! mopac_cuda_transform_fock_with_s overwrites Sfull with its factor on host; Ffull becomes F'
      info = 0
      call mopac_cuda_transform_fock_with_s(n, Sfull, n, Ffull, n, info)
      if (info /= 0) then
        ! fallback to CPU
        call dgemm('N','N', n, n, n, one, Ffull, n, Uinv, n, zero, T, n)
        call dgemm('T','N', n, n, n, one, Uinv, n, T, n, zero, Fp, n)
      else
        Fp = Ffull
      end if
    else
      call dgemm('N','N', n, n, n, one, Ffull, n, Uinv, n, zero, T, n)
      call dgemm('T','N', n, n, n, one, Uinv, n, T, n, zero, Fp, n)
    end if
#else
    call dgemm('N','N', n, n, n, one, Ffull, n, Uinv, n, zero, T, n)
    call dgemm('T','N', n, n, n, one, Uinv, n, T, n, zero, Fp, n)
#endif

    ! Estimate spectrum bounds (diagonal min/max + safety margin)
    lam_min = Fp(1,1)
    lam_max = Fp(1,1)
    do i=1,n
      lam_min = min(lam_min, Fp(i,i))
      lam_max = max(lam_max, Fp(i,i))
    end do
    beta = lam_max - lam_min
    if (beta <= 0.d0) beta = 1.d0
    mu = 0.5d0*(lam_max + lam_min)

    ! X0 = 1/2 I - (F' - mu I)/beta
    X = 0.d0
    do i=1,n
      X(i,i) = 0.5d0 - (Fp(i,i) - mu)/beta
    end do
    ! Use tolerances passed by caller
    tol_l = tol
    maxit_l = maxit

    do k=1,maxit_l
      ! X2 = X*X
#ifdef GPU
      if (use_gpu) then
        call gemm_cublas('N','N', n, n, n, 1.d0, X, n, X, n, 0.d0, X2, n)
      else
        call dgemm('N','N', n, n, n, one, X, n, X, n, zero, X2, n)
      end if
#else
      call dgemm('N','N', n, n, n, one, X, n, X, n, zero, X2, n)
#endif
      ! Trace(X)
      trX = 0.d0
      do i=1,n
        trX = trX + X(i,i)
      end do
      target = 0.5d0 * dble(nelecs)
      delta = target - trX
      sigma = 0.d0
      if (delta > 0.d0) then
        sigma = 1.d0
      else if (delta < 0.d0) then
        sigma = -1.d0
      end if
      ! X <- X + sigma*(X - X2_orig) = (1+sigma)X - sigma*X2_orig
      call dscal(n*n, 1.d0 + sigma, X, 1)
      call daxpy(n*n, -sigma, X2, 1, X, 1)
      ! Convergence: ||X - X^2||_F
      ! Recompute X2 = X*X for error check
#ifdef GPU
      if (use_gpu) then
        call gemm_cublas('N','N', n, n, n, 1.d0, X, n, X, n, 0.d0, X2, n)
      else
        call dgemm('N','N', n, n, n, one, X, n, X, n, zero, X2, n)
      end if
#else
      call dgemm('N','N', n, n, n, one, X, n, X, n, zero, X2, n)
#endif
      ! err = ||X - X^2||_F
      s = 0.d0
      do j=1,n
        do i=1,n
          diff = X(i,j) - X2(i,j)
          s = s + diff*diff
        end do
      end do
      err = dsqrt(s)
      if (abs(delta) < max(1.d0, target)*tol_l .and. err < tol_l) exit
    end do

    ! Transform back to AO: Pao = 2 * Uinv^T * X * Uinv
#ifdef GPU
    if (use_gpu) then
      call gemm_cublas('N','N', n, n, n, 1.d0, X, n, Uinv, n, 0.d0, T, n)
      call gemm_cublas('T','N', n, n, n, 2.d0, Uinv, n, T, n, 0.d0, Pao, n)
    else
      call dgemm('N','N', n, n, n, one, X, n, Uinv, n, zero, T, n)
      call dgemm('T','N', n, n, n, 2.d0, Uinv, n, T, n, zero, Pao, n)
    end if
#else
    call dgemm('N','N', n, n, n, one, X, n, Uinv, n, zero, T, n)
    call dgemm('T','N', n, n, n, 2.d0, Uinv, n, T, n, zero, Pao, n)
#endif
    call dtrttp('U', n, Pao, n, pp, info)
    if (info /= 0) stop 'purify: dtrttp on P failed'

    deallocate(Sfull, Ffull, U, Uinv, Fp, X, X2, T, Pao, Spack)
  end subroutine purify_density_from_fock

  ! No separate norm function; computed inline above for simplicity/compliance

end module purify_gpu
