program gpu_density_verify
  implicit none
  integer, parameter :: wp = kind(1.0d0)
  integer, parameter :: n = 64
  integer :: ndubl, nsingl, norbs, mpack, mode, i
  real(wp) :: fract, occ
  real(wp), allocatable :: c(:,:), pp_gpu(:), pp_cpu(:)
  real(wp) :: max_abs_diff, denom

  norbs = n
  ndubl = n/2
  nsingl = ndubl
  mpack = (n*(n+1))/2
  mode = 1
  fract = 0.0_wp
  occ = 2.0_wp

  allocate(c(n,n), pp_gpu(mpack), pp_cpu(mpack))
  call random_seed()
  call random_number(c)

  ! Normalize columns to avoid ill-conditioning (optional)
  do i=1,n
     c(:,i) = c(:,i) / max(1.0_wp, sqrt(sum(c(:,i)**2)))
  end do

  ! GPU: SYRK path (iopc=4)
  call density_for_GPU(c, fract, ndubl, nsingl, occ, mpack, norbs, mode, pp_gpu, 4)

  ! CPU: SYRK path (iopc=5)
  call density_for_GPU(c, fract, ndubl, nsingl, occ, mpack, norbs, mode, pp_cpu, 5)

  max_abs_diff = maxval(abs(pp_gpu - pp_cpu))
  denom = max(1.0_wp, maxval(abs(pp_cpu)))
  print *, 'Density max abs diff:', max_abs_diff
  print *, 'Density rel diff:', max_abs_diff/denom

  deallocate(c, pp_gpu, pp_cpu)
end program gpu_density_verify

