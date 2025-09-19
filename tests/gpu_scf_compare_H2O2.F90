program gpu_scf_compare_H2O2
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
  double precision, allocatable :: pa_cpu(:), pa_gpu(:), f_cpu(:), f_gpu(:)
  double precision :: d_pa, denom_pa, d_f, denom_f

  ! Hydrogen peroxide H2O2
  sys%natom = 4
  sys%natom_move = 0
  allocate(sys%atom(4))
  sys%atom = [1, 1, 8, 8]
  allocate(sys%coord(12))
  sys%coord = [ -1.0d0, 0.0d0, 0.0d0, &
                 1.0d0, 0.0d0, 0.0d0, &
                 0.0d0, 0.9d0, 0.0d0, &
                 0.0d0,-0.9d0, 0.0d0 ]
  sys%model = 0
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
    print *, 'CPU state not available (H2O2)'
    stop 1
  end if
  allocate(pa_cpu(st_cpu%mpack))
  pa_cpu = st_cpu%pa
  allocate(f_cpu(st_cpu%mpack))
  f_cpu = f(:st_cpu%mpack)

  ! GPU run
#ifdef GPU
  lgpu = .true.
#endif
  call mopac_scf_f(sys, st_gpu, props)
  if (st_gpu%mpack <= 0) then
    print *, 'GPU state not available (H2O2)'
    stop 1
  end if
  allocate(pa_gpu(st_gpu%mpack))
  pa_gpu = st_gpu%pa
  allocate(f_gpu(st_gpu%mpack))
  f_gpu = f(:st_gpu%mpack)

  n = min(size(pa_cpu), size(pa_gpu))
  d_pa = maxval(abs(pa_cpu(:n) - pa_gpu(:n)))
  denom_pa = max(1.0d0, maxval(abs(pa_cpu(:n))))
  print *, 'H2O2 SCF density max abs diff:', d_pa
  print *, 'H2O2 SCF density rel diff:', d_pa/denom_pa
  call scf_assert('H2O2_SCF', d_pa, denom_pa)

  n = min(size(f_cpu), size(f_gpu))
  d_f = maxval(abs(f_cpu(:n) - f_gpu(:n)))
  denom_f = max(1.0d0, maxval(abs(f_cpu(:n))))
  print *, 'H2O2 Fock max abs diff:', d_f
  print *, 'H2O2 Fock rel diff:', d_f/denom_f
  call scf_assert_fock('H2O2_FOCK', d_f, denom_f)

contains
  subroutine scf_assert(label, diff_abs, denom)
    character(*), intent(in) :: label
    double precision, intent(in) :: diff_abs, denom
    character(len=64) :: env
    integer :: ist
    double precision :: tol_rel, tol_abs
    tol_rel = -1.d0; tol_abs = -1.d0
    call get_environment_variable('MOPAC_SCF_REL_TOL', env, status=ist)
    if (ist == 0) read(env,*,err=10,end=10) tol_rel
10  continue
    call get_environment_variable('MOPAC_SCF_ABS_TOL', env, status=ist)
    if (ist == 0) read(env,*,err=20,end=20) tol_abs
20  continue
    if ((tol_rel > 0.d0 .and. diff_abs/max(1.d0,denom) > tol_rel) .or. &
        (tol_abs > 0.d0 .and. diff_abs > tol_abs)) then
      print *, trim(label)//' SCF check FAIL'
      stop 1
    else
      print *, trim(label)//' SCF check PASS'
    end if
  end subroutine scf_assert
  subroutine scf_assert_fock(label, diff_abs, denom)
    character(*), intent(in) :: label
    double precision, intent(in) :: diff_abs, denom
    character(len=64) :: env
    integer :: ist
    double precision :: tol_rel, tol_abs
    tol_rel = -1.d0; tol_abs = -1.d0
    call get_environment_variable('MOPAC_SCF_FOCK_REL_TOL', env, status=ist)
    if (ist == 0) read(env,*,err=10,end=10) tol_rel
10  continue
    call get_environment_variable('MOPAC_SCF_FOCK_ABS_TOL', env, status=ist)
    if (ist == 0) read(env,*,err=20,end=20) tol_abs
20  continue
    if ((tol_rel > 0.d0 .and. diff_abs/max(1.d0,denom) > tol_rel) .or. &
        (tol_abs > 0.d0 .and. diff_abs > tol_abs)) then
      print *, trim(label)//' FOCK check FAIL'
      stop 1
    else
      print *, trim(label)//' FOCK check PASS'
    end if
  end subroutine scf_assert_fock
end program gpu_scf_compare_H2O2
