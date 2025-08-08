program gpu_verify
  use iso_fortran_env, only: wp => real64
#ifdef GPU
  use eigenvectors_cuda, only: eigenvectors_CUDA
#endif
  implicit none
  integer, parameter :: n = 64
  real(wp) :: A_full(n,n), A_pack((n*(n+1))/2)
  real(wp) :: eig_gpu(n), eig_cpu(n)
  real(wp) :: vec_gpu(n,n), vec_cpu(n,n)
  real(wp) :: diff, normA
  integer :: i, j, info

  call random_seed()
  call random_number(A_full)
  ! Symmetrize
  do i=1,n
     do j=1,n
        A_full(i,j) = 0.5_wp*(A_full(i,j) + A_full(j,i))
     end do
  end do
  ! Pack upper triangle
  call dtrttp('U', n, A_full, n, A_pack, info)

#ifdef GPU
  call eigenvectors_CUDA(vec_gpu, A_pack, eig_gpu, n)
#endif

  ! CPU reference path using LAPACK dsyevd
  vec_cpu = A_full
  call cpu_dsyevd(vec_cpu, eig_cpu, n)

#ifdef GPU
  diff = maxval(abs(eig_gpu - eig_cpu))
  normA = max(1.0_wp, maxval(abs(eig_cpu)))
  print *, 'Max eig diff (abs):', diff
  print *, 'Rel eig diff:', diff/normA
#else
  print *, 'GPU not enabled; build with -DGPU=ON to run this check.'
#endif

contains

  subroutine cpu_dsyevd(a, w, n)
    implicit none
    integer, intent(in) :: n
    real(wp), intent(inout) :: a(n,n)
    real(wp), intent(out) :: w(n)
    integer :: lwork, liwork, info
    real(wp), allocatable :: work(:)
    integer, allocatable :: iwork(:)
    real(wp) :: workq(1)
    integer :: iworkq(1)
    lwork = -1; liwork = -1
    call dsyevd('V','U', n, a, n, w, workq, lwork, iworkq, liwork, info)
    lwork = int(workq(1)); liwork = iworkq(1)
    allocate(work(lwork), iwork(liwork))
    call dsyevd('V','U', n, a, n, w, work, lwork, iwork, liwork, info)
    deallocate(work, iwork)
  end subroutine cpu_dsyevd

end program gpu_verify

