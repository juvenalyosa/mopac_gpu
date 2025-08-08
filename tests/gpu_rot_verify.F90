program gpu_rot_verify
  implicit none
  integer, parameter :: wp = kind(1.0d0)
  integer, parameter :: n = 96
  integer, parameter :: nocc = 40
  integer :: lumo, i, j, ij
  real(wp), allocatable :: V_gpu(:,:), V_cpu(:,:), eig(:), fmo(:)
  real(wp) :: bigeps, tiny

  allocate(V_gpu(n,n), V_cpu(n,n), eig(n), fmo(nocc*(n-nocc)))
  call random_seed()
  call random_number(V_gpu)
  V_cpu = V_gpu
  call random_number(eig)
  call random_number(fmo)
  eig = eig*10.0_wp
  bigeps = 1.5d-7
  tiny = 1.0d-6
  lumo = nocc + 1

  ! Apply GPU rotation
  call rot_cuda(fmo, eig, V_gpu, V_gpu(:,1:nocc), V_gpu(:,lumo:n), nocc, lumo, n, bigeps, tiny)

  ! CPU reference rotation (same math as call_rot_cuda CPU path)
  ij = 0
  do i = 1, nocc
    do j = lumo, n
      real(wp) :: x, a, b, d, e, alpha, beta
      ij = ij + 1
      x = fmo(ij)
      if (abs(x) < tiny) cycle
      a = eig(i)
      b = eig(j)
      d = a - b
      if (abs(x/d) < bigeps) cycle
      e = sign(sqrt(4.0_wp*x*x + d*d), d)
      alpha = sqrt(0.5_wp*(1.0_wp + d/e))
      beta  = -sign(sqrt(1.0_wp - alpha*alpha), x)
      call drot(n, V_cpu(1:n,i), 1, V_cpu(1:n,j), 1, alpha, beta)
    end do
  end do

  print *, 'Rotation max abs diff:', maxval(abs(V_gpu - V_cpu))

  deallocate(V_gpu, V_cpu, eig, fmo)
end program gpu_rot_verify

