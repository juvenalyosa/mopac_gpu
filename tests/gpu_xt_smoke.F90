program gpu_xt_smoke
#ifdef GPU
  use mopac_cublas_interfaces
  use mod_vars_cuda, only: ngpus
#endif
  implicit none
  integer, parameter :: n=256
  real*8, allocatable :: A(:,:), B(:,:), C(:,:)
  integer :: i, j
  allocate(A(n,n), B(n,n), C(n,n))
  do j=1,n
    do i=1,n
      A(i,j) = dble(mod(i*j,97))/97.d0
      B(i,j) = dble(mod(i+2*j,89))/89.d0
    end do
  end do
#ifdef GPU
  if (ngpus > 1) then
    call gemm_cublas_multi('N','N', n, n, n, 1.d0, A, n, B, n, 0.d0, C, n)
  else
    call gemm_cublas('N','N', n, n, n, 1.d0, A, n, B, n, 0.d0, C, n)
  end if
  print *, 'gpu_xt_smoke: PASS (ngpus=', ngpus, ')'
#else
  print *, 'gpu_xt_smoke: GPU not enabled'
#endif
end program gpu_xt_smoke

