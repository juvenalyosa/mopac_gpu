program gpu_rot_2gpu_verify
  use iso_fortran_env, only: wp => real64
  use iso_c_binding,   only: c_bool, c_int, c_char, c_size_t
  use call_rot_cuda,   only: rot_cuda_2gpu
  use gpu_info,        only: gpuInfo
  use settingGPUcard,  only: setMGpuPair
  implicit none

  integer, parameter :: n    = 96
  integer, parameter :: nocc = 40
  integer :: lumo, i, j, ij
  real(wp), allocatable :: V_gpu(:,:), V_cpu(:,:), eig(:), fmo(:)
  real(wp) :: bigeps, tiny
  real(wp) :: x, a, b, d, e, alpha, beta
  logical(c_bool) :: hasGpu
  integer(c_int)  :: nDevices
  logical(c_bool) :: hasDouble(6)
  character(kind=c_char) :: name(6)
  integer(c_int)  :: name_size(6), clockRate(6), major(6), minor(6)
  integer(c_size_t) :: totalMem(6)

  ! Query GPU availability and require at least two devices
  call gpuInfo(hasGpu, hasDouble, nDevices, name, name_size, totalMem, clockRate, major, minor)
  if ((.not. hasGpu) .or. (nDevices < 2_c_int)) then
    print *, 'Requires >= 2 GPUs; skipping 2-GPU rotation check.'
    stop 0
  end if

  ! Choose the first two devices for the pair
  call setMGpuPair(0_c_int, 1_c_int)

  allocate(V_gpu(n,n), V_cpu(n,n), eig(n), fmo(nocc*(n-nocc)))
  call random_seed()
  call random_number(V_gpu)
  V_cpu = V_gpu
  call random_number(eig)
  call random_number(fmo)
  eig = eig*10.0_wp
  bigeps = 1.5d-7
  tiny   = 1.0d-6
  lumo   = nocc + 1

  ! Apply 2-GPU rotation
  call rot_cuda_2gpu(fmo, eig, V_gpu, V_gpu(:,1:nocc), V_gpu(:,lumo:n), nocc, lumo, n, bigeps, tiny)

  ! CPU reference rotation (same math used in GPU kernels)
  ij = 0
  do i = 1, nocc
    do j = lumo, n
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

  print *, '2-GPU rotation max abs diff:', maxval(abs(V_gpu - V_cpu))

  deallocate(V_gpu, V_cpu, eig, fmo)
end program gpu_rot_2gpu_verify

