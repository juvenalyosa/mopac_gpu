! Hierarchical memetic trust-region GPU optimizer (prototype)

module hmtr_optimizer_mod
  use iso_c_binding, only : c_double, c_int, c_bool
#ifdef GPU
  use gpu_hmtr_interfaces
#endif
  use chanel_C, only : iw
#ifdef _OPENMP
  use omp_lib
#endif
#ifdef GPU
  use settingGPUcard, only : setGPU
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
    logical(c_bool) :: gpu_set_ok
#endif

    if (hmtr_cache_lookup(pop, pop%torsions(member,:), energy, grad_tors, grad_full)) return
#ifdef _OPENMP
    tid = omp_get_thread_num()
#endif
#ifdef GPU
    if (pop%gpu_enabled .and. ngpus > 1) then
       device = mod(tid, ngpus)
       call setGPU(device, gpu_set_ok)
    end if
#endif
    status = 0
    scratch = pop%base_coords
    scratch(pop%torsion_idx) = pop%torsions(member,:)
    call evaluator(scratch, energy, grad_full, status)
    if (status /= 0) return
    grad_tors = grad_full(pop%torsion_idx)
    call hmtr_micro_refine(pop, member, evaluator, energy, grad_tors, grad_full, scratch, status)
    if (status == 0) call hmtr_cache_store(pop, pop%torsions(member,:), energy, grad_tors, grad_full)
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

    stat = 0
!$omp parallel default(shared) private(member, ierr, scratch_loc)
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
