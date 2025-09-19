program gpu_grad_compare_OO
  use mopac_api_f
#ifdef GPU
  use mod_vars_cuda, only: lgpu
#endif
  implicit none
  type(mopac_system_f) :: sys
  type(mopac_state_f) :: st
  type(mopac_properties_f) :: props
  double precision, allocatable :: g_cpu(:), g_gpu(:)
  integer :: nmove
  double precision :: diff, denom

  ! O–O diatomic (heavy–heavy case)
  sys%natom = 2
  sys%natom_move = 2
  allocate(sys%atom(2))
  sys%atom = [8, 8]
  allocate(sys%coord(6))
  sys%coord = [ -0.6d0, 0.0d0, 0.0d0, &
                 0.6d0, 0.0d0, 0.0d0 ]
  sys%model = 0   ! PM7
  sys%epsilon = 1.0d0
  sys%spin = 0
  sys%tolerance = 1.0d0
  sys%max_time = 60

  ! CPU gradient
#ifdef GPU
  lgpu = .false.
#endif
  call mopac_relax_f(sys, st, props)
  nmove = 3*sys%natom_move
  if (.not. allocated(props%coord_deriv)) then
    print *, 'No gradient from CPU relax'
    stop 1
  end if
  allocate(g_cpu(nmove))
  g_cpu = props%coord_deriv(:nmove)

  ! GPU gradient
#ifdef GPU
  lgpu = .true.
#endif
  call mopac_relax_f(sys, st, props)
  if (.not. allocated(props%coord_deriv)) then
    print *, 'No gradient from GPU relax'
    stop 1
  end if
  allocate(g_gpu(nmove))
  g_gpu = props%coord_deriv(:nmove)

  diff = maxval(abs(g_cpu - g_gpu))
  denom = max(1.0d0, maxval(abs(g_cpu)))
  print *, 'OO Grad max abs diff:', diff
  print *, 'OO Grad rel diff:', diff/denom
  call grad_assert('OO', diff, denom)
contains
  subroutine grad_assert(label, diff_abs, denom)
    character(*), intent(in) :: label
    double precision, intent(in) :: diff_abs, denom
    character(len=64) :: env
    integer :: ist
    double precision :: tol_rel, tol_abs
    tol_rel = -1.d0; tol_abs = -1.d0
    call get_environment_variable('MOPAC_GRAD_REL_TOL', env, status=ist)
    if (ist == 0) read(env,*,err=10,end=10) tol_rel
10  continue
    call get_environment_variable('MOPAC_GRAD_ABS_TOL', env, status=ist)
    if (ist == 0) read(env,*,err=20,end=20) tol_abs
20  continue
    if ((tol_rel > 0.d0 .and. diff_abs/max(1.d0,denom) > tol_rel) .or. &
        (tol_abs > 0.d0 .and. diff_abs > tol_abs)) then
      print *, trim(label)//' gradient check FAIL'
      stop 1
    else
      print *, trim(label)//' gradient check PASS'
    end if
  end subroutine grad_assert
end program gpu_grad_compare_OO
