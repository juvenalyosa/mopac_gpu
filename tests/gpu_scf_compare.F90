program gpu_scf_compare
  use mopac_api_f
  use Common_arrays_C, only: f
  use molkst_C,       only: mpack
#ifdef GPU
  use mod_vars_cuda, only: lgpu
#endif
  implicit none
  type(mopac_system_f) :: sys
  type(mopac_state_f) :: st_cpu, st_gpu
  type(mopac_properties_f) :: props
  integer :: n
  double precision, allocatable :: f_cpu(:), f_gpu(:)
  double precision :: d_pa, d_f, denom_pa, denom_f

  ! Build a small H2O system
  sys%natom = 3
  sys%natom_move = 0
  allocate(sys%atom(3))
  sys%atom = [1, 1, 8]
  allocate(sys%coord(9))
  sys%coord = [0.76d0, 0.59d0, 0.0d0, &
               -0.76d0, 0.59d0, 0.0d0, &
               0.0d0, 0.0d0, 0.0d0]
  sys%model = 0  ! PM7
  sys%epsilon = 1.0d0
  sys%spin = 0
  sys%tolerance = 1.0d0
  sys%max_time = 60

  ! CPU run
#ifdef GPU
  lgpu = .false.
#endif
  call mopac_scf_f(sys, st_cpu, props)
  if (st_cpu%mpack <= 0) then
    print *, 'CPU state not available'
    stop 1
  end if
  allocate(f_cpu(st_cpu%mpack))
  f_cpu = f(:st_cpu%mpack)

  ! GPU run (force GPU)
#ifdef GPU
  lgpu = .true.
#endif
  call mopac_scf_f(sys, st_gpu, props)
  if (st_gpu%mpack <= 0) then
    print *, 'GPU state not available'
    stop 1
  end if
  allocate(f_gpu(st_gpu%mpack))
  f_gpu = f(:st_gpu%mpack)

  ! Compare densities and Fock
  n = min(size(st_cpu%pa), size(st_gpu%pa))
  d_pa = maxval(abs(st_cpu%pa(:n) - st_gpu%pa(:n)))
  denom_pa = max(1.0d0, maxval(abs(st_cpu%pa(:n))))
  print *, 'SCF density max abs diff:', d_pa
  print *, 'SCF density rel diff:', d_pa/denom_pa

  n = min(size(f_cpu), size(f_gpu))
  d_f = maxval(abs(f_cpu(:n) - f_gpu(:n)))
  denom_f = max(1.0d0, maxval(abs(f_cpu(:n))))
  print *, 'Fock max abs diff:', d_f
  print *, 'Fock rel diff:', d_f/denom_f

contains
#endif
end program gpu_scf_compare
