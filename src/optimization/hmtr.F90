! Hierarchical memetic trust-region GPU optimizer (prototype)

module hmtr_optimizer_mod
  use iso_c_binding, only : c_double, c_int, c_bool, c_ptr, c_null_ptr, c_associated
#ifdef GPU
  use gpu_hmtr_interfaces
  use gpu_runtime_interfaces, only: mopac_cuda_set_active_stream, mopac_cuda_clear_active_stream
#endif
  use chanel_C, only : iw
#ifdef _OPENMP
  use omp_lib
#endif
#ifdef GPU
  use mod_vars_cuda, only : ngpus
#endif
  implicit none

  integer, parameter :: dp = c_double
  real(dp), parameter :: PI = 3.14159265358979323846264338327950288_dp
  real(dp), parameter :: TWO_PI = 2.0_dp * PI
  integer, parameter :: HMTR_DEFAULT_POPULATION = 32
  integer, parameter :: HMTR_DEFAULT_GLOBAL_ITERS = 100
  real(dp), parameter :: HMTR_MICRO_RADIUS = 0.05_dp
  real(dp), parameter :: HMTR_MICRO_TOL = 1.0e-8_dp
  real(dp), parameter :: HMTR_MICRO_RADIUS_MIN = 1.0e-4_dp
  real(dp), parameter :: HMTR_MICRO_RADIUS_MAX = 0.5_dp
  real(dp), parameter :: HMTR_MICRO_ETA1 = 0.25_dp
  real(dp), parameter :: HMTR_MICRO_ETA2 = 0.75_dp
  real(dp), parameter :: HMTR_CACHE_TOL = 1.0e-8_dp

  abstract interface
     subroutine hmtr_evaluator(coords, energy, grad, ierr)
       import :: dp
       real(dp), intent(in) :: coords(:)
       real(dp), intent(out) :: energy
       real(dp), intent(out) :: grad(:)
       integer, intent(out) :: ierr
     end subroutine hmtr_evaluator
  end interface

  interface
     subroutine compfg(xparam, int, escf, fulscf, grad, lgrad)
       import :: dp
       real(dp), intent(in) :: xparam(:)
       logical, intent(in) :: int
       real(dp), intent(out) :: escf
       logical, intent(in) :: fulscf
       real(dp), intent(out) :: grad(:)
       logical, intent(in) :: lgrad
     end subroutine compfg
  end interface

#ifdef GPU
  logical, parameter :: hmtr_gpu_available = .true.
#else
  logical, parameter :: hmtr_gpu_available = .false.
#endif
  logical :: hmtr_force_gpu_eval = .false.
  real(dp) :: hmtr_rho_sum = 0.0_dp
  integer :: hmtr_rho_count = 0
  integer :: hmtr_rho_expand = 0
  integer :: hmtr_rho_shrink = 0
#ifdef GPU
  integer, allocatable :: hmtr_device_map(:)
  integer, allocatable :: hmtr_thread_device(:)
  integer, allocatable :: hmtr_device_total(:)
  integer, allocatable :: hmtr_device_batch(:)
  type(c_ptr), allocatable :: hmtr_thread_stream(:)
  integer :: hmtr_device_cursor = 0
  logical :: hmtr_device_map_ready = .false.
#endif

  type :: hmtr_params_type
     real(dp) :: inertia = 0.7_dp
     real(dp) :: cognitive = 1.6_dp
     real(dp) :: social = 1.6_dp
     real(dp) :: max_velocity = 0.35_dp
     logical :: use_wrap = .true.
  end type hmtr_params_type

  type :: hmtr_population
     integer :: population = 0
     integer :: dim = 0
     logical :: gpu_enabled = .false.
     type(hmtr_params_type) :: params
    real(dp), allocatable :: torsions(:,:)
    real(dp), allocatable :: velocities(:,:)
    real(dp), allocatable :: pbest(:,:)
    real(dp), allocatable :: pbest_energy(:)
    real(dp), allocatable :: pbest_grad(:,:)
    real(dp), allocatable :: gbest(:)
    real(dp), allocatable :: gbest_grad(:)
    real(dp), allocatable :: rand1(:,:)
    real(dp), allocatable :: rand2(:,:)
    integer, allocatable :: torsion_idx(:)
    real(dp), allocatable :: base_coords(:)
    real(dp), allocatable :: pbest_grad_full(:,:)
    real(dp), allocatable :: gbest_grad_full(:)
    real(dp), allocatable :: cache_torsions(:,:)
    real(dp), allocatable :: cache_grad_tors(:,:)
    real(dp), allocatable :: cache_grad_full(:,:)
    real(dp), allocatable :: cache_energy(:)
    real(dp), allocatable :: micro_radius(:)
    integer :: cache_capacity = 0
    integer :: cache_size = 0
    integer :: cache_next = 1
  end type hmtr_population

contains

#ifdef GPU
  subroutine hmtr_parse_device_env(str, values)
    character(len=*), intent(in) :: str
    integer, allocatable, intent(out) :: values(:)
    integer :: lenstr, i, start, count, ios, val
    integer, allocatable :: shrunk(:)
    character(len=:), allocatable :: buffer

    lenstr = len_trim(str)
    if (lenstr <= 0) then
      allocate(values(0))
      return
    end if

    allocate(character(len=lenstr) :: buffer)
    buffer = adjustl(str(1:lenstr))

    allocate(values(lenstr))
    count = 0
    start = 1

    do i = 1, lenstr
      select case (buffer(i:i))
      case (',', ' ', ';', ':')
        if (i > start) then
          read(buffer(start:i-1), *, iostat=ios) val
          if (ios == 0) then
            count = count + 1
            values(count) = val
          end if
        end if
        start = i + 1
      end select
    end do

    if (start <= lenstr) then
      read(buffer(start:lenstr), *, iostat=ios) val
      if (ios == 0) then
        count = count + 1
        values(count) = val
      end if
    end if

    if (count <= 0) then
      deallocate(values)
      allocate(values(0))
    else if (count < lenstr) then
      allocate(shrunk(count))
      shrunk = values(1:count)
      call move_alloc(shrunk, values)
    end if

    if (allocated(buffer)) deallocate(buffer)
  end subroutine hmtr_parse_device_env

  subroutine hmtr_init_device_map()
    use mod_vars_cuda, only : ngpus
    integer :: status, i, valid_count
    character(len=512) :: env
    integer, allocatable :: parsed(:)

    if (hmtr_device_map_ready) return
    if (ngpus <= 0) then
      hmtr_device_map_ready = .true.
      return
    end if

    env = ''
    status = 1
    call get_environment_variable('HMTR_GPU_MAP', env, status=status)
    if (status == 0) then
      call hmtr_parse_device_env(env, parsed)
    else
      allocate(parsed(0))
    end if

    valid_count = 0
    if (allocated(parsed)) then
      do i = 1, size(parsed)
        if (parsed(i) >= 0 .and. parsed(i) < ngpus) valid_count = valid_count + 1
      end do
    end if

    if (valid_count > 0) then
      if (allocated(hmtr_device_map)) deallocate(hmtr_device_map)
      allocate(hmtr_device_map(valid_count))
      valid_count = 0
      do i = 1, size(parsed)
        if (parsed(i) >= 0 .and. parsed(i) < ngpus) then
          valid_count = valid_count + 1
          hmtr_device_map(valid_count) = parsed(i)
        end if
      end do
    else
      if (allocated(hmtr_device_map)) deallocate(hmtr_device_map)
      allocate(hmtr_device_map(ngpus))
      do i = 1, ngpus
        hmtr_device_map(i) = i - 1
      end do
    end if

    if (allocated(parsed)) deallocate(parsed)

    if (allocated(hmtr_device_total)) deallocate(hmtr_device_total)
    if (allocated(hmtr_device_batch)) deallocate(hmtr_device_batch)
    allocate(hmtr_device_total(ngpus))
    allocate(hmtr_device_batch(ngpus))
    hmtr_device_total = 0
    hmtr_device_batch = 0
    hmtr_device_cursor = 0
    hmtr_device_map_ready = .true.
  end subroutine hmtr_init_device_map

  subroutine hmtr_ensure_thread_slots()
    integer :: needed, old_size
    integer, allocatable :: temp(:)
    type(c_ptr), allocatable :: tmp_stream(:)
#ifdef _OPENMP
    needed = omp_get_max_threads()
#else
    needed = 1
#endif
    if (needed <= 0) needed = 1

    if (.not. allocated(hmtr_thread_device)) then
      allocate(hmtr_thread_device(needed))
      hmtr_thread_device = -1
    else if (size(hmtr_thread_device) < needed) then
      old_size = size(hmtr_thread_device)
      allocate(temp(needed))
      temp = -1
      temp(1:old_size) = hmtr_thread_device
      call move_alloc(temp, hmtr_thread_device)
    end if

    if (.not. allocated(hmtr_thread_stream)) then
      allocate(hmtr_thread_stream(needed))
      hmtr_thread_stream = c_null_ptr
    else if (size(hmtr_thread_stream) < needed) then
      old_size = size(hmtr_thread_stream)
      allocate(tmp_stream(needed))
      tmp_stream = c_null_ptr
      tmp_stream(1:old_size) = hmtr_thread_stream
      call move_alloc(tmp_stream, hmtr_thread_stream)
    end if
  end subroutine hmtr_ensure_thread_slots

  integer function hmtr_assign_device()
    integer :: map_len, idx
    if (.not. hmtr_device_map_ready) call hmtr_init_device_map()
    if (.not. allocated(hmtr_device_map)) then
      hmtr_assign_device = -1
      return
    end if
    map_len = size(hmtr_device_map)
    if (map_len <= 0) then
      hmtr_assign_device = -1
      return
    end if
#ifdef _OPENMP
!$omp atomic capture
    hmtr_device_cursor = hmtr_device_cursor + 1
    idx = hmtr_device_cursor
!$omp end atomic
#else
    hmtr_device_cursor = hmtr_device_cursor + 1
    idx = hmtr_device_cursor
#endif
    hmtr_assign_device = hmtr_device_map(mod(idx - 1, map_len) + 1)
  end function hmtr_assign_device

  subroutine hmtr_note_device_use(device)
    integer, intent(in) :: device
    if (device < 0) return
    if (.not. allocated(hmtr_device_total)) return
    if (.not. allocated(hmtr_device_batch)) return
#ifdef _OPENMP
!$omp atomic
#endif
    hmtr_device_total(device + 1) = hmtr_device_total(device + 1) + 1
#ifdef _OPENMP
!$omp atomic
#endif
    hmtr_device_batch(device + 1) = hmtr_device_batch(device + 1) + 1
  end subroutine hmtr_note_device_use

  subroutine hmtr_prepare_batch(use_gpu)
    logical, intent(in) :: use_gpu
    if (.not. use_gpu) return
    call hmtr_init_device_map()
    call hmtr_ensure_thread_slots()
    if (allocated(hmtr_device_batch)) hmtr_device_batch = 0
  end subroutine hmtr_prepare_batch

  subroutine hmtr_log_batch_usage(use_gpu)
    logical, intent(in) :: use_gpu
    integer :: i
    logical :: any
    if (.not. use_gpu) return
    if (.not. allocated(hmtr_device_batch)) return
    any = .false.
    do i = 1, size(hmtr_device_batch)
      if (hmtr_device_batch(i) > 0) then
        if (.not. any) then
          write(iw,'(1x,"HMTR GPU batch usage:")')
          any = .true.
        end if
        write(iw,'(3x,"GPU",I0,":",I0)') i - 1, hmtr_device_batch(i)
      end if
    end do
    hmtr_device_batch = 0
  end subroutine hmtr_log_batch_usage

  subroutine hmtr_log_total_usage(use_gpu)
    logical, intent(in) :: use_gpu
    integer :: i
    logical :: any
    if (.not. use_gpu) return
    if (.not. allocated(hmtr_device_total)) return
    any = .false.
    do i = 1, size(hmtr_device_total)
      if (hmtr_device_total(i) > 0) then
        if (.not. any) then
          write(iw,'(1x,"HMTR GPU aggregate usage:")')
          any = .true.
        end if
        write(iw,'(3x,"GPU",I0,":",I0)') i - 1, hmtr_device_total(i)
      end if
    end do
    if (any) hmtr_device_total = 0
  end subroutine hmtr_log_total_usage
#endif

  subroutine hmtr_initialize(pop, torsion_init, params, use_gpu, torsion_idx, base_coords, ierr)
    type(hmtr_population), intent(inout) :: pop
    real(dp), intent(in) :: torsion_init(:,:)
    type(hmtr_params_type), intent(in), optional :: params
    logical, intent(in), optional :: use_gpu
    integer, intent(in) :: torsion_idx(:)
    real(dp), intent(in) :: base_coords(:)
    integer, intent(out), optional :: ierr
    integer :: stat
    logical :: want_gpu
#ifdef GPU
    type(mopac_hmtr_config) :: cfg
#endif

    stat = 0
    want_gpu = hmtr_gpu_available
    if (present(use_gpu)) want_gpu = use_gpu .and. hmtr_gpu_available

    pop%population = size(torsion_init, dim=1)
    pop%dim = size(torsion_init, dim=2)

    if (allocated(pop%torsions)) call hmtr_finalize(pop)

    allocate(pop%torsions(pop%population, pop%dim))
    allocate(pop%velocities(pop%population, pop%dim))
    allocate(pop%pbest(pop%population, pop%dim))
    allocate(pop%pbest_energy(pop%population))
    allocate(pop%pbest_grad(pop%population, pop%dim))
    allocate(pop%gbest(pop%dim))
    allocate(pop%gbest_grad(pop%dim))
    allocate(pop%rand1(pop%population, pop%dim))
    allocate(pop%rand2(pop%population, pop%dim))
    allocate(pop%torsion_idx(pop%dim))
    allocate(pop%base_coords(size(base_coords)))
    allocate(pop%pbest_grad_full(pop%population, size(base_coords)))
    allocate(pop%gbest_grad_full(size(base_coords)))
    allocate(pop%cache_torsions(pop%population, pop%dim))
    allocate(pop%cache_grad_tors(pop%population, pop%dim))
    allocate(pop%cache_grad_full(pop%population, size(base_coords)))
    allocate(pop%cache_energy(pop%population))
    allocate(pop%micro_radius(pop%population))

    pop%params = hmtr_params_type()
    if (present(params)) pop%params = params

    pop%torsions = torsion_init
    pop%velocities = 0.0_dp
    pop%pbest = torsion_init
    pop%pbest_energy = huge(1.0_dp)
    pop%gbest = torsion_init(1,:)
    pop%gbest_grad = 0.0_dp
    pop%pbest_grad = 0.0_dp
    pop%torsion_idx = torsion_idx
    pop%base_coords = base_coords
    pop%pbest_grad_full = 0.0_dp
    pop%gbest_grad_full = 0.0_dp
    pop%cache_capacity = pop%population
    pop%cache_size = 0
    pop%cache_next = 1
    pop%micro_radius = HMTR_MICRO_RADIUS
#ifdef GPU
    if (want_gpu) then
       call hmtr_init_device_map()
       call hmtr_ensure_thread_slots()
       cfg%torsion_dim = pop%dim
        cfg%population_size = pop%population
        cfg%wrap_angles = merge(1, 0, pop%params%use_wrap)
        cfg%inertia = pop%params%inertia
        cfg%cognitive = pop%params%cognitive
        cfg%social = pop%params%social
        cfg%max_velocity = pop%params%max_velocity
        stat = mopac_cuda_hmtr_configure(cfg)
        if (stat == 0) stat = mopac_cuda_hmtr_upload_population(pop%torsions, pop%velocities, pop%pbest)
        if (stat == 0) stat = mopac_cuda_hmtr_set_gbest(pop%gbest)
        pop%gpu_enabled = (stat == 0)
        if (.not. pop%gpu_enabled) then
          call mopac_cuda_hmtr_release()
        end if
    else
        pop%gpu_enabled = .false.
    end if
#else
    if (want_gpu) pop%gpu_enabled = .false.
#endif

    if (present(ierr)) ierr = stat
  end subroutine hmtr_initialize

  subroutine hmtr_finalize(pop)
    type(hmtr_population), intent(inout) :: pop

#ifdef GPU
    if (pop%gpu_enabled) call mopac_cuda_hmtr_release()
    call mopac_cuda_hmtr_clear_streams()
    if (allocated(hmtr_thread_device)) hmtr_thread_device = -1
    if (allocated(hmtr_thread_stream)) hmtr_thread_stream = c_null_ptr
#endif
    if (allocated(pop%torsions)) deallocate(pop%torsions)
    if (allocated(pop%velocities)) deallocate(pop%velocities)
    if (allocated(pop%pbest)) deallocate(pop%pbest)
    if (allocated(pop%pbest_energy)) deallocate(pop%pbest_energy)
    if (allocated(pop%pbest_grad)) deallocate(pop%pbest_grad)
    if (allocated(pop%gbest)) deallocate(pop%gbest)
    if (allocated(pop%gbest_grad)) deallocate(pop%gbest_grad)
    if (allocated(pop%rand1)) deallocate(pop%rand1)
    if (allocated(pop%rand2)) deallocate(pop%rand2)
    if (allocated(pop%torsion_idx)) deallocate(pop%torsion_idx)
    if (allocated(pop%base_coords)) deallocate(pop%base_coords)
    if (allocated(pop%pbest_grad_full)) deallocate(pop%pbest_grad_full)
    if (allocated(pop%gbest_grad_full)) deallocate(pop%gbest_grad_full)
    if (allocated(pop%cache_torsions)) deallocate(pop%cache_torsions)
    if (allocated(pop%cache_grad_tors)) deallocate(pop%cache_grad_tors)
    if (allocated(pop%cache_grad_full)) deallocate(pop%cache_grad_full)
    if (allocated(pop%cache_energy)) deallocate(pop%cache_energy)
    if (allocated(pop%micro_radius)) deallocate(pop%micro_radius)
    pop%population = 0
    pop%dim = 0
    pop%gpu_enabled = .false.
  end subroutine hmtr_finalize

  integer function hmtr_global_step(pop) result(stat)
    type(hmtr_population), intent(inout) :: pop
    stat = 0
    if (pop%population <= 0 .or. pop%dim <= 0) then
       stat = 1
       return
    end if

    call random_number(pop%rand1)
    call random_number(pop%rand2)

#ifdef GPU
    if (pop%gpu_enabled) then
       stat = mopac_cuda_hmtr_upload_population(pop%torsions, pop%velocities, pop%pbest)
       if (stat == 0) stat = mopac_cuda_hmtr_set_gbest(pop%gbest)
       if (stat == 0) stat = mopac_cuda_hmtr_pso_step(pop%rand1, pop%rand2)
       if (stat == 0) stat = mopac_cuda_hmtr_download_population(pop%torsions, pop%velocities)
       if (stat /= 0) then
          call mopac_cuda_hmtr_release()
          pop%gpu_enabled = .false.
          stat = 0
          call hmtr_cpu_update(pop)
       else
          call enforce_bounds(pop%torsions, pop%params%use_wrap)
       end if
    else
       call hmtr_cpu_update(pop)
    end if
#else
    call hmtr_cpu_update(pop)
#endif
  end function hmtr_global_step

  subroutine hmtr_update_best(pop, energies, gradients, gradients_full)
    type(hmtr_population), intent(inout) :: pop
    real(dp), intent(in) :: energies(:)
    real(dp), intent(in) :: gradients(:,:)
    real(dp), intent(in) :: gradients_full(:,:)
    integer :: i
    integer :: best_idx(1)

    if (size(energies) /= pop%population) return
    do i = 1, pop%population
       if (energies(i) < pop%pbest_energy(i)) then
          pop%pbest_energy(i) = energies(i)
          pop%pbest(i,:) = pop%torsions(i,:)
          pop%pbest_grad(i,:) = gradients(i,:)
          pop%pbest_grad_full(i,:) = gradients_full(i,:)
       end if
    end do
    best_idx = minloc(pop%pbest_energy)
    pop%gbest = pop%pbest(best_idx(1),:)
    pop%gbest_grad = pop%pbest_grad(best_idx(1),:)
    pop%gbest_grad_full = pop%pbest_grad_full(best_idx(1),:)
  end subroutine hmtr_update_best

  subroutine hmtr_cpu_update(pop)
    type(hmtr_population), intent(inout) :: pop
    integer :: i, j
    real(dp) :: delta_p, delta_g, new_vel

    do i = 1, pop%population
       do j = 1, pop%dim
          if (pop%params%use_wrap) then
             delta_p = wrap_delta(pop%pbest(i,j) - pop%torsions(i,j))
             delta_g = wrap_delta(pop%gbest(j) - pop%torsions(i,j))
          else
             delta_p = pop%pbest(i,j) - pop%torsions(i,j)
             delta_g = pop%gbest(j) - pop%torsions(i,j)
          end if
          new_vel = pop%params%inertia * pop%velocities(i,j) + &
                    pop%params%cognitive * pop%rand1(i,j) * delta_p + &
                    pop%params%social * pop%rand2(i,j) * delta_g
          if (new_vel > pop%params%max_velocity) new_vel = pop%params%max_velocity
          if (new_vel < -pop%params%max_velocity) new_vel = -pop%params%max_velocity
          pop%velocities(i,j) = new_vel
          if (pop%params%use_wrap) then
             pop%torsions(i,j) = wrap_angle(pop%torsions(i,j) + new_vel)
          else
             pop%torsions(i,j) = pop%torsions(i,j) + new_vel
          end if
       end do
    end do
    call enforce_bounds(pop%torsions, pop%params%use_wrap)
  end subroutine hmtr_cpu_update

  pure real(dp) function wrap_angle(x) result(val)
    real(dp), intent(in) :: x
    val = modulo(x, TWO_PI)
    if (val < 0.0_dp) val = val + TWO_PI
  end function wrap_angle

  pure real(dp) function wrap_delta(x) result(val)
    real(dp), intent(in) :: x
    val = modulo(x + PI, TWO_PI)
    if (val < 0.0_dp) val = val + TWO_PI
    val = val - PI
  end function wrap_delta

  subroutine enforce_bounds(tors, use_wrap)
    real(dp), intent(inout) :: tors(:,:)
    logical, intent(in) :: use_wrap
    integer :: i, j
    if (.not. use_wrap) return
    do i = 1, size(tors, dim=1)
       do j = 1, size(tors, dim=2)
          tors(i,j) = wrap_angle(tors(i,j))
       end do
    end do
  end subroutine enforce_bounds

  subroutine hmtr_optimize_torsions(xseed, evaluator, best_coords, best_energy, best_grad, &
       max_iters, pop_size, use_gpu, grad_tol, wrap_angles, ierr)
    use common_arrays_C, only : loc, lopt
    real(dp), intent(in) :: xseed(:)
    procedure(hmtr_evaluator) :: evaluator
    real(dp), intent(out) :: best_coords(:)
    real(dp), intent(out) :: best_energy
    real(dp), intent(out) :: best_grad(:)
    integer, intent(in), optional :: max_iters, pop_size
    logical, intent(in), optional :: use_gpu
    real(dp), intent(in), optional :: grad_tol
    logical, intent(in), optional :: wrap_angles
    integer, intent(out), optional :: ierr

    type(hmtr_population) :: pop
    type(hmtr_params_type) :: params
    integer :: iterations, population, nvar, member, ev_stat, status, iter_idx
    real(dp) :: tolerance, base_energy
    real(dp), allocatable :: init(:,:), energies(:), gradients(:,:), gradients_full(:,:)
    real(dp), allocatable :: candidate(:), base_grad(:)
    integer, allocatable :: torsion_idx(:)
    integer :: ntors, i

    nvar = size(xseed)
    hmtr_force_gpu_eval = .false.
    status = 0
    population = HMTR_DEFAULT_POPULATION
    if (present(pop_size)) population = max(1, pop_size)
    iterations = HMTR_DEFAULT_GLOBAL_ITERS
    if (present(max_iters)) iterations = max(1, max_iters)
    tolerance = 1.0e-3_dp
    if (present(grad_tol)) tolerance = grad_tol

    if (size(best_coords) /= nvar) then
       status = 3
       if (present(ierr)) ierr = status
       return
    end if
    if (size(best_grad) /= nvar) then
       status = 3
       if (present(ierr)) ierr = status
       return
    end if

    hmtr_rho_sum = 0.0_dp
    hmtr_rho_count = 0
    hmtr_rho_expand = 0
    hmtr_rho_shrink = 0

    allocate(base_grad(nvar))
    call evaluator(xseed, base_energy, base_grad, status)
    if (status /= 0) then
       best_coords = xseed
       best_energy = base_energy
       best_grad = base_grad
       if (present(ierr)) ierr = status
       deallocate(base_grad)
       return
    end if

    allocate(torsion_idx(nvar))
    ntors = 0
    do i = 1, nvar
       if (loc(2,i) == 3 .and. loc(1,i) > 0) then
          if (lopt(3,loc(1,i)) /= 0) then
             ntors = ntors + 1
             torsion_idx(ntors) = i
          end if
       end if
    end do
    if (ntors == 0) then
       best_coords = xseed
       best_energy = base_energy
       best_grad = base_grad
       if (present(ierr)) ierr = 0
       deallocate(base_grad, torsion_idx)
       return
    end if
    torsion_idx = torsion_idx(:ntors)

    params = hmtr_params_type()
    if (present(wrap_angles)) params%use_wrap = wrap_angles

    allocate(init(population, ntors))
    do i = 1, population
       init(i,:) = xseed(torsion_idx)
    end do
    if (population > 1) call perturb_initial(init, params%use_wrap)

    if (present(use_gpu)) then
       call hmtr_initialize(pop, init, params, use_gpu=use_gpu, torsion_idx=torsion_idx, base_coords=xseed, ierr=status)
    else
       call hmtr_initialize(pop, init, params, torsion_idx=torsion_idx, base_coords=xseed, ierr=status)
    end if
    if (status /= 0) then
       if (present(ierr)) ierr = status
       call hmtr_finalize(pop)
       if (allocated(init)) deallocate(init)
       if (allocated(torsion_idx)) deallocate(torsion_idx)
       if (allocated(base_grad)) deallocate(base_grad)
       return
    end if

    hmtr_force_gpu_eval = pop%gpu_enabled

    allocate(energies(population))
    allocate(gradients(population, ntors))
    allocate(gradients_full(population, nvar))
    allocate(candidate(nvar))

    status = hmtr_evaluate_batch(pop, 1, population, evaluator, energies, gradients, gradients_full)
    if (status /= 0) then
       call hmtr_finalize(pop)
       if (allocated(init)) deallocate(init)
       if (allocated(energies)) deallocate(energies)
       if (allocated(gradients)) deallocate(gradients)
       if (allocated(gradients_full)) deallocate(gradients_full)
       if (allocated(candidate)) deallocate(candidate)
       if (allocated(torsion_idx)) deallocate(torsion_idx)
       if (allocated(base_grad)) deallocate(base_grad)
       if (present(ierr)) ierr = status
       hmtr_force_gpu_eval = .false.
       return
    end if
    call hmtr_update_best(pop, energies, gradients, gradients_full)

    do iter_idx = 1, iterations
       status = hmtr_global_step(pop)
#ifdef GPU
       hmtr_force_gpu_eval = pop%gpu_enabled
#endif
       if (status /= 0) exit
       status = hmtr_evaluate_batch(pop, 1, population, evaluator, energies, gradients, gradients_full)
       if (status /= 0) exit
       call hmtr_update_best(pop, energies, gradients, gradients_full)
       if (maxval(abs(pop%gbest_grad)) < tolerance) exit
    end do

    if (status == 0) then
       best_coords = pop%base_coords
       best_coords(pop%torsion_idx) = pop%gbest
       best_energy = minval(pop%pbest_energy)
       best_grad = pop%gbest_grad_full
    else
       best_coords = xseed
       best_energy = base_energy
       best_grad = base_grad
    end if
    if (present(ierr)) ierr = status

    call hmtr_finalize(pop)
    hmtr_force_gpu_eval = .false.

    if (hmtr_rho_count > 0) then
       write(iw,'(1x,"HMTR micro-step avg rho=",F8.4," shrink=",I4," expand=",I4)') &
            hmtr_rho_sum/real(hmtr_rho_count,dp), hmtr_rho_shrink, hmtr_rho_expand
    end if

    hmtr_rho_sum = 0.0_dp
    hmtr_rho_count = 0
    hmtr_rho_expand = 0
    hmtr_rho_shrink = 0

#ifdef GPU
    call hmtr_log_total_usage(pop%gpu_enabled)
#endif
    if (allocated(init)) deallocate(init)
    if (allocated(energies)) deallocate(energies)
    if (allocated(gradients)) deallocate(gradients)
    if (allocated(gradients_full)) deallocate(gradients_full)
    if (allocated(candidate)) deallocate(candidate)
    if (allocated(torsion_idx)) deallocate(torsion_idx)
    if (allocated(base_grad)) deallocate(base_grad)
  end subroutine hmtr_optimize_torsions

  subroutine perturb_initial(arr, use_wrap)
    real(dp), intent(inout) :: arr(:,:)
    logical, intent(in) :: use_wrap
    integer :: i, j
    real(dp) :: noise
    do i = 2, size(arr, dim=1)
       do j = 1, size(arr, dim=2)
          call random_number(noise)
          if (use_wrap) then
             arr(i,j) = wrap_angle(arr(i,j) + 0.1_dp * (noise - 0.5_dp))
          else
             arr(i,j) = arr(i,j) + 0.1_dp * (noise - 0.5_dp)
          end if
       end do
    end do
  end subroutine perturb_initial

  subroutine hmtr_evaluate_member(pop, member, evaluator, energy, grad_tors, grad_full, scratch, status)
    type(hmtr_population), intent(inout) :: pop
    integer, intent(in) :: member
    procedure(hmtr_evaluator) :: evaluator
    real(dp), intent(inout) :: energy
    real(dp), intent(inout) :: grad_tors(:)
    real(dp), intent(inout) :: grad_full(:)
    real(dp), intent(inout) :: scratch(:)
    integer, intent(inout) :: status
#ifdef _OPENMP
    integer :: tid
#else
    integer, parameter :: tid = 0
#endif
#ifdef GPU
    integer :: device
    integer(c_int) :: device_c, tid_c, changed_flag
    type(c_ptr) :: stream_handle
    logical :: stream_bound
#endif

    if (hmtr_cache_lookup(pop, pop%torsions(member,:), energy, grad_tors, grad_full)) return
#ifdef _OPENMP
    tid = omp_get_thread_num()
#endif
#ifdef GPU
    stream_handle = c_null_ptr
    changed_flag = 0_c_int
    device = -1
    stream_bound = .false.
    if (pop%gpu_enabled .and. ngpus > 0) then
       if (.not. hmtr_device_map_ready) call hmtr_init_device_map()
       if (tid + 1 > size(hmtr_thread_device)) then
          pop%gpu_enabled = .false.
       else
          if (hmtr_thread_device(tid+1) < 0) then
             hmtr_thread_device(tid+1) = hmtr_assign_device()
          end if
          device = hmtr_thread_device(tid+1)
          if (device >= 0) then
             device_c = int(device, kind=c_int)
             tid_c = int(tid, kind=c_int)
             call mopac_cuda_hmtr_bind_thread(device_c, tid_c, stream_handle, changed_flag)
             hmtr_thread_stream(tid+1) = stream_handle
             if (c_associated(stream_handle)) then
                call mopac_cuda_set_active_stream(stream_handle)
                stream_bound = .true.
             end if
             call hmtr_note_device_use(device)
          else
             pop%gpu_enabled = .false.
          end if
       end if
    end if
#endif
    status = 0
    scratch = pop%base_coords
    scratch(pop%torsion_idx) = pop%torsions(member,:)
    call evaluator(scratch, energy, grad_full, status)
    if (status == 0) then
       grad_tors = grad_full(pop%torsion_idx)
       call hmtr_micro_refine(pop, member, evaluator, energy, grad_tors, grad_full, scratch, status)
       if (status == 0) call hmtr_cache_store(pop, pop%torsions(member,:), energy, grad_tors, grad_full)
    end if
#ifdef GPU
    if (stream_bound) call mopac_cuda_clear_active_stream()
#endif
  end subroutine hmtr_evaluate_member

  integer function hmtr_evaluate_batch(pop, first, last, evaluator, energies, gradients, gradients_full) result(stat)
    type(hmtr_population), intent(inout) :: pop
    integer, intent(in) :: first, last
    procedure(hmtr_evaluator) :: evaluator
    real(dp), intent(inout) :: energies(:)
    real(dp), intent(inout) :: gradients(:,:)
    real(dp), intent(inout) :: gradients_full(:,:)
    integer :: member, ierr
    real(dp), allocatable :: scratch_loc(:)

#ifdef GPU
    call hmtr_prepare_batch(pop%gpu_enabled)
#endif
    stat = 0
!$omp parallel default(shared) private(member, ierr, scratch_loc) if (.not. pop%gpu_enabled)
    allocate(scratch_loc(size(pop%base_coords)))
!$omp do schedule(dynamic)
    do member = first, last
       call hmtr_evaluate_member(pop, member, evaluator, energies(member), gradients(member,:), &
            gradients_full(member,:), scratch_loc, ierr)
       if (ierr /= 0) then
!$omp critical(hmtr_batch_err)
          if (stat == 0) stat = ierr
!$omp end critical(hmtr_batch_err)
       end if
    end do
!$omp end do nowait
    deallocate(scratch_loc)
!$omp end parallel
#ifdef GPU
    call hmtr_log_batch_usage(pop%gpu_enabled)
#endif
  end function hmtr_evaluate_batch

  logical function hmtr_cache_lookup(pop, tors, energy, grad_tors, grad_full)
    type(hmtr_population), intent(in) :: pop
    real(dp), intent(in) :: tors(:)
    real(dp), intent(out) :: energy
    real(dp), intent(out) :: grad_tors(:)
    real(dp), intent(out) :: grad_full(:)
    integer :: i

    hmtr_cache_lookup = .false.
    if (pop%cache_size <= 0) return
    do i = 1, pop%cache_size
       if (maxval(abs(pop%cache_torsions(i,:) - tors)) <= HMTR_CACHE_TOL) then
          energy = pop%cache_energy(i)
          grad_tors = pop%cache_grad_tors(i,:)
          grad_full = pop%cache_grad_full(i,:)
          hmtr_cache_lookup = .true.
          return
       end if
    end do
  end function hmtr_cache_lookup

  subroutine hmtr_cache_store(pop, tors, energy, grad_tors, grad_full)
    type(hmtr_population), intent(inout) :: pop
    real(dp), intent(in) :: tors(:)
    real(dp), intent(in) :: energy
    real(dp), intent(in) :: grad_tors(:)
    real(dp), intent(in) :: grad_full(:)
    integer :: idx

    if (pop%cache_capacity <= 0) return
!$omp critical(hmtr_cache_store)
    idx = pop%cache_next
    pop%cache_torsions(idx,:) = tors
    pop%cache_energy(idx) = energy
    pop%cache_grad_tors(idx,:) = grad_tors
    pop%cache_grad_full(idx,:) = grad_full
    pop%cache_next = pop%cache_next + 1
    if (pop%cache_next > pop%cache_capacity) pop%cache_next = 1
    if (pop%cache_size < pop%cache_capacity) pop%cache_size = pop%cache_size + 1
!$omp end critical(hmtr_cache_store)
  end subroutine hmtr_cache_store

  subroutine hmtr_micro_refine(pop, member, evaluator, energy, grad_tors, grad_full, scratch, status)
    type(hmtr_population), intent(inout) :: pop
    integer, intent(in) :: member
    procedure(hmtr_evaluator) :: evaluator
    real(dp), intent(inout) :: energy
    real(dp), intent(inout) :: grad_tors(:)
    real(dp), intent(inout) :: grad_full(:)
    real(dp), intent(inout) :: scratch(:)
    integer, intent(inout) :: status
    real(dp), allocatable :: new_grad(:), step(:), new_tors(:), grad_old(:)
    real(dp) :: step_norm, scale, new_energy, energy_old, pred, actual, rho, radius, radius_old
    integer :: i

    status = 0
    radius = pop%micro_radius(member)
    radius_old = radius
    if (radius <= HMTR_MICRO_RADIUS_MIN) return
    allocate(step(pop%dim))
    allocate(grad_old(pop%dim))
    grad_old = grad_tors
    step = -grad_old
    step_norm = sqrt(sum(step**2))
    if (step_norm <= HMTR_MICRO_TOL) then
       deallocate(step)
       deallocate(grad_old)
       return
    end if
    scale = min(1.0_dp, radius / step_norm)
    step = step * scale
    step_norm = sqrt(sum(step**2))
    allocate(new_tors(pop%dim))
    new_tors = pop%torsions(member,:) + step
    if (pop%params%use_wrap) then
       do i = 1, pop%dim
          new_tors(i) = wrap_angle(new_tors(i))
       end do
    end if

    scratch = pop%base_coords
    scratch(pop%torsion_idx) = new_tors
    allocate(new_grad(size(grad_full)))
    energy_old = energy
    call evaluator(scratch, new_energy, new_grad, status)
    if (status == 0 .and. new_energy < energy) then
       pred = -dot_product(grad_old, step)
       actual = energy_old - new_energy
       if (pred > HMTR_MICRO_TOL .and. actual > 0.0_dp) then
          rho = actual / pred
       else
          rho = 1.0_dp
       end if
       if (rho >= HMTR_MICRO_ETA1) then
          pop%torsions(member,:) = new_tors
          energy = new_energy
          grad_full = new_grad
          grad_tors = new_grad(pop%torsion_idx)
          if (rho > HMTR_MICRO_ETA2 .and. abs(step_norm - radius) < 1.0e-6_dp) then
             radius = min(radius * 2.0_dp, HMTR_MICRO_RADIUS_MAX)
          else if (rho < HMTR_MICRO_ETA1) then
             radius = max(radius * 0.5_dp, HMTR_MICRO_RADIUS_MIN)
          end if
       else
          radius = max(radius * 0.5_dp, HMTR_MICRO_RADIUS_MIN)
       end if
    else
       status = 0
       radius = max(radius * 0.5_dp, HMTR_MICRO_RADIUS_MIN)
       pred = 0.0_dp
       actual = 0.0_dp
       rho = 0.0_dp
    end if
    if (pred > HMTR_MICRO_TOL .and. actual > 0.0_dp) then
       hmtr_rho_sum = hmtr_rho_sum + rho
       hmtr_rho_count = hmtr_rho_count + 1
    end if
    if (radius > radius_old * (1.0_dp + 1.0e-6_dp)) hmtr_rho_expand = hmtr_rho_expand + 1
    if (radius < radius_old * (1.0_dp - 1.0e-6_dp)) hmtr_rho_shrink = hmtr_rho_shrink + 1
    deallocate(new_grad)
    deallocate(step)
    deallocate(new_tors)
    deallocate(grad_old)
    pop%micro_radius(member) = radius
  end subroutine hmtr_micro_refine

  subroutine hmtr_compfg_evaluator(coords, energy, grad, ierr)
    use molkst_C, only : nvar, moperr
#ifdef GPU
    use mod_vars_cuda, only : lgpu
#endif
    real(dp), intent(in) :: coords(:)
    real(dp), intent(out) :: energy
    real(dp), intent(out) :: grad(:)
    integer, intent(out) :: ierr
    real(dp), allocatable :: grad_local(:)
    logical :: want_grad
#ifdef GPU
    logical :: old_lgpu
#endif

    ierr = 0
    if (size(coords) /= nvar .or. size(grad) /= nvar) then
       ierr = 4
       return
    end if

    allocate(grad_local(nvar))
    want_grad = .true.
#ifdef GPU
    if (hmtr_force_gpu_eval) then
       old_lgpu = lgpu
       lgpu = .true.
    else
       old_lgpu = lgpu
    end if
#endif
    call compfg(coords, .true., energy, .true., grad_local, want_grad)
#ifdef GPU
    if (hmtr_force_gpu_eval) lgpu = old_lgpu
#endif
    grad = grad_local
    if (allocated(grad_local)) deallocate(grad_local)
    if (moperr) ierr = 1
  end subroutine hmtr_compfg_evaluator

end module hmtr_optimizer_mod
