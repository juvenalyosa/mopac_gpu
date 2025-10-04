program gpu_numeric_reference
  use iso_fortran_env, only : real64
  use molkst_C, only : norbs, mpack
#ifdef GPU
  use iso_c_binding, only : c_bool, c_int
  use gpu_fock_interfaces, only : mopac_cuda_fock2_scf
#endif
  implicit none
  integer :: failures
  failures = 0
#ifdef GPU
  if (.not. test_light_light()) failures = failures + 1
  if (.not. test_general_pair()) failures = failures + 1
  if (.not. test_heavy_light()) failures = failures + 1
#else
  print *, 'GPU support not enabled; skipping gpu_numeric_reference'
#endif
  if (failures /= 0) stop 1
contains
#ifdef GPU
  subroutine reset_dimensions(n)
    integer, intent(in) :: n
    norbs = n
    mpack = n * (n + 1) / 2
  end subroutine reset_dimensions

  logical function test_light_light()
    implicit none
    integer, parameter :: numat = 2
    integer(c_int) :: nfirst(numat), nlast(numat)
    integer :: local_norbs, local_mpack
    double precision :: ptot(3), p(3), w(1)
    double precision :: f_cpu(3), f_gpu(3)
    logical(c_bool) :: ok
    double precision :: diff, denom

    test_light_light = .false.

    local_norbs = 2
    call reset_dimensions(local_norbs)
    local_mpack = mpack

    nfirst = [1_c_int, 2_c_int]
    nlast  = [1_c_int, 2_c_int]

    ptot = (/1.20d0, 0.05d0, 0.90d0/)
    p    = (/1.18d0, 0.02d0, 0.88d0/)
    w(1) = 0.45d0
    f_cpu = 0.0d0
    f_gpu = 0.0d0

    call cpu_fock_light_light(ptot, p, w, f_cpu)

    ok = mopac_cuda_fock2_scf(local_norbs, local_mpack, numat, nfirst, nlast, ptot, p, w, w, w, 0_c_int, f_gpu)
    if (.not. ok) then
      print *, '[LIGHT-LIGHT] GPU path unavailable; skipping'
      test_light_light = .true.
      return
    end if

    diff = maxval(abs(f_cpu - f_gpu))
    denom = max(1.0d0, maxval(abs(f_cpu)))
    print '(a,1pe18.10)', '[LIGHT-LIGHT] max abs diff = ', diff
    print '(a,3(1pe14.6,1x))', '  CPU:', f_cpu
    print '(a,3(1pe14.6,1x))', '  GPU:', f_gpu
    if (diff > 1.0d-8 .and. diff/denom > 1.0d-8) then
      print *, '[LIGHT-LIGHT] CPU and GPU results differ beyond tolerance'
      return
    end if
    test_light_light = .true.
  end function test_light_light

  logical function test_general_pair()
    implicit none
    integer, parameter :: numat = 2
    integer(c_int) :: nfirst(numat), nlast(numat)
    integer :: local_norbs, local_mpack
    double precision, allocatable :: ptot(:), p(:), w(:)
    double precision, allocatable :: f_cpu(:), f_gpu(:)
    logical(c_bool) :: ok
    double precision :: diff, denom
    integer :: len_w, i

    test_general_pair = .false.

    local_norbs = 4
    call reset_dimensions(local_norbs)
    local_mpack = mpack

    allocate(ptot(local_mpack), p(local_mpack), f_cpu(local_mpack), f_gpu(local_mpack))
    do i = 1, local_mpack
      ptot(i) = 0.02d0 * dble(i)
      p(i)    = 0.015d0 * dble(i)
    end do
    f_cpu = 0.0d0
    f_gpu = 0.0d0

    nfirst = [1_c_int, 3_c_int]
    nlast  = [2_c_int, 4_c_int]

    len_w = pair_count(span_count(3, 4)) * pair_count(span_count(1, 2))
    if (len_w <= 0) then
      print *, '[GENERAL] no integrals to process'
      test_general_pair = .true.
      deallocate(ptot, p, f_cpu, f_gpu)
      return
    end if

    allocate(w(len_w))
    do i = 1, len_w
      w(i) = 1.0d-4 * dble(i)
    end do

    call cpu_fock_general(3, 4, 1, 2, ptot, p, w, f_cpu)

    ok = mopac_cuda_fock2_scf(local_norbs, local_mpack, numat, nfirst, nlast, ptot, p, w, w, w, 0_c_int, f_gpu)
    if (.not. ok) then
      print *, '[GENERAL] GPU path unavailable; skipping'
      test_general_pair = .true.
      deallocate(ptot, p, f_cpu, f_gpu, w)
      return
    end if

    diff = maxval(abs(f_cpu - f_gpu))
    denom = max(1.0d0, maxval(abs(f_cpu)))
    print '(a,1pe18.10)', '[GENERAL] max abs diff = ', diff
    print '(a)', '[GENERAL] CPU:'
    do i = 1, local_mpack
      write(*,'(1pe14.6,1x)', advance='no') f_cpu(i)
      if (mod(i,6) == 0 .or. i == local_mpack) write(*,*)
    end do
    print '(a)', '  GPU:'
    do i = 1, local_mpack
      write(*,'(1pe14.6,1x)', advance='no') f_gpu(i)
      if (mod(i,6) == 0 .or. i == local_mpack) write(*,*)
    end do
    if (diff > 1.0d-8 .and. diff/denom > 1.0d-8) then
      print *, '[GENERAL] CPU and GPU results differ beyond tolerance'
      deallocate(ptot, p, f_cpu, f_gpu, w)
      return
    end if
    test_general_pair = .true.
    deallocate(ptot, p, f_cpu, f_gpu, w)
  end function test_general_pair

  logical function test_heavy_light()
    implicit none
    integer, parameter :: numat = 2
    integer(c_int) :: nfirst(numat), nlast(numat)
    integer :: local_norbs, local_mpack
    double precision, allocatable :: ptot(:), p(:), w(:)
    double precision :: f_cpu(6), f_gpu(6)
    logical(c_bool) :: ok
    double precision :: diff, denom

    test_heavy_light = .false.

    local_norbs = 3
    call reset_dimensions(local_norbs)
    local_mpack = mpack

    allocate(ptot(local_mpack), p(local_mpack), w((span_count(1,2)*(span_count(1,2)+1))/2))
    ptot = (/1.05d0, 0.02d0, 0.01d0, 0.98d0, 0.03d0, 0.90d0/)
    p    = (/1.00d0, 0.01d0, 0.01d0, 0.95d0, 0.02d0, 0.88d0/)
    w    = (/0.10d0, 0.12d0, 0.14d0/)  ! Coulomb block entries

    nfirst = [1_c_int, 3_c_int]
    nlast  = [2_c_int, 3_c_int]

    call cpu_heavy_light_reference(1, 2, 3, ptot, p, w, f_cpu)

    f_gpu = 0.0d0
    ok = mopac_cuda_fock2_scf(local_norbs, local_mpack, numat, nfirst, nlast, ptot, p, w, w, w, 0_c_int, f_gpu)
    if (.not. ok) then
      print *, '[HEAVY-LIGHT] GPU path unavailable; skipping'
      test_heavy_light = .true.
      deallocate(ptot, p, w)
      return
    end if

    diff = maxval(abs(f_cpu - f_gpu))
    denom = max(1.0d0, maxval(abs(f_cpu)))
    print '(a,1pe18.10)', '[HEAVY-LIGHT] max abs diff = ', diff
    print '(a)', '[HEAVY-LIGHT] CPU:'
    write(*,'(6(1pe14.6,1x))') f_cpu
    print '(a)', '  GPU:'
    write(*,'(6(1pe14.6,1x))') f_gpu
    if (diff > 1.0d-8 .and. diff/denom > 1.0d-8) then
      print *, '[HEAVY-LIGHT] CPU and GPU results differ beyond tolerance'
      deallocate(ptot, p, w)
      return
    end if
    test_heavy_light = .true.
    deallocate(ptot, p, w)
  end function test_heavy_light

  subroutine cpu_fock_light_light(ptot, p, w, f)
    implicit none
```},{