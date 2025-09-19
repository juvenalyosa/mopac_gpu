program gpu_grad_compare_H2O2
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

  ! Hydrogen peroxide H2O2 (mixed heavy-heavy and light)
  sys%natom = 4
  sys%natom_move = 4
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

  ! CPU gradient
#ifdef GPU
  lgpu = .false.
#endif
  call mopac_relax_f(sys, st, props)
  nmove = 3*sys%natom_move
  if (.not. allocated(props%coord_deriv)) then
    print *, 'No gradient from CPU relax (H2O2)'
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
    print *, 'No gradient from GPU relax (H2O2)'
    stop 1
  end if
  allocate(g_gpu(nmove))
  g_gpu = props%coord_deriv(:nmove)

  diff = maxval(abs(g_cpu - g_gpu))
  denom = max(1.0d0, maxval(abs(g_cpu)))
  print *, 'H2O2 Grad max abs diff:', diff
  print *, 'H2O2 Grad rel diff:', diff/denom
  call grad_assert('H2O2', diff, denom)
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
end program gpu_grad_compare_H2O2
