! Simple GPU benchmark harness for GEMM, SYRK, DSYEVD, and accuracy checks
program gpu_bench
  use iso_c_binding
  use mopac_cublas_interfaces
  implicit none

  interface
    subroutine mopac_cuda_destroy_resources() bind(C, name='mopac_cuda_destroy_resources')
    end subroutine mopac_cuda_destroy_resources
    subroutine mopac_cuda_dsyevd(n, A, lda, W, info) bind(C, name='mopac_cuda_dsyevd')
      use iso_c_binding
      implicit none
      integer(c_int), value :: n
      integer(c_int), value :: lda
      real(c_double)        :: A(lda,*)
      real(c_double)        :: W(*)
      integer(c_int)        :: info
    end subroutine mopac_cuda_dsyevd
    subroutine call_rot_cuda_gpu(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny) &
      bind(C, name='call_rot_cuda_gpu')
      use iso_c_binding
      implicit none
      integer(c_int), value :: nocc, lumo, n
      real(c_double), value :: bigeps, tiny
      real(c_double)        :: fmo(*)
      real(c_double)        :: eig(*)
      real(c_double)        :: vector(n,*)
      real(c_double)        :: ci0(*)
      real(c_double)        :: ca0(*)
    end subroutine call_rot_cuda_gpu
    subroutine call_rot_cuda_2gpu_gpu(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny) &
      bind(C, name='call_rot_cuda_2gpu_gpu')
      use iso_c_binding
      implicit none
      integer(c_int), value :: nocc, lumo, n
      real(c_double), value :: bigeps, tiny
      real(c_double)        :: fmo(*)
      real(c_double)        :: eig(*)
      real(c_double)        :: vector(n,*)
      real(c_double)        :: ci0(*)
      real(c_double)        :: ca0(*)
    end subroutine call_rot_cuda_2gpu_gpu
  end interface

  ! Tunables with defaults (can be overridden via CLI flags)
  integer :: gemm_m=1024, gemm_n=1024, gemm_k=64, gemm_iters=20
  integer :: syrk_n=1024, syrk_k=64, syrk_iters=20
  integer :: dsy_n=512, dsy_iters=5
  integer :: rot1_n=1024, rot1_iters=10
  integer :: rot2_n=2048, rot2_iters=10
  integer :: acc_n=256, acc_k=128
  integer :: accuracy_failures = 0
  logical :: syrk_full = .false.
  logical :: accuracy_enabled = .false.
  logical :: accuracy_only = .false.
  real(c_double), parameter :: GEMM_MAX_ABS_TOL = 1.0d-10
  real(c_double), parameter :: GEMM_RMS_ABS_TOL = 1.0d-11
  real(c_double), parameter :: GEMM_REL_RMS_TOL = 1.0d-12
  real(c_double), parameter :: SYRK_MAX_ABS_TOL = 1.0d-10
  real(c_double), parameter :: SYRK_RMS_ABS_TOL = 1.0d-11
  real(c_double), parameter :: SYRK_REL_RMS_TOL = 1.0d-12
  real(c_double), parameter :: DSYEVD_RESIDUAL_TOL = 1.0d-10
  real(c_double), parameter :: DSYEVD_ORTHOGONALITY_TOL = 1.0d-10
  real(c_double), parameter :: ROT_MAX_ABS_TOL = 1.0d-10
  real(c_double), parameter :: ROT_RMS_ABS_TOL = 1.0d-11
  real(c_double), parameter :: ROT_REL_RMS_TOL = 1.0d-10

  call parse_args()
  if (.not. accuracy_only) then
    call bench_gemm()
    call bench_syrk()
    call bench_dsyevd()
    call bench_rot_single()
    call bench_rot_2gpu()
  end if
  if (accuracy_enabled) call bench_accuracy()
  call mopac_cuda_destroy_resources()
  if (accuracy_failures > 0) then
    write(*,'(a,i0)') 'ACCURACY_FAIL count=', accuracy_failures
    stop 1
  end if

contains

  subroutine bench_gemm()
    implicit none
    integer :: m,n,k, lda, ldb, ldc
    integer :: iters, i
    real(c_double), allocatable :: A(:,:), B(:,:), C(:,:)
    real(c_double) :: alpha, beta
    integer :: c0, c1, rate
    real(c_double) :: t_first, t_rest
    real(c_double) :: flops, gflops_first, gflops_rest

    m = gemm_m; n = gemm_n; k = gemm_k
    lda = m; ldb = k; ldc = m
    iters = gemm_iters
    allocate(A(lda,k), B(ldb,n), C(ldc,n))
    call random_seed()
    call random_number(A); call random_number(B); C = 0.0d0
    alpha = 1.0d0; beta = 0.0d0

    call system_clock(c0, rate)
    call gemm_cublas('N','N', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    call system_clock(c1)
    t_first = real(c1-c0,8)/real(rate,8)

    call system_clock(c0, rate)
    do i = 1, iters
      call gemm_cublas('N','N', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    end do
    call system_clock(c1)
    t_rest = real(c1-c0,8)/real(rate,8)/real(iters,8)
    if (matrix_has_nan(C)) then
      call record_bench_fail('GEMM produced non-finite output')
      deallocate(A,B,C)
      return
    end if

    flops = 2.0d0 * dble(m) * dble(n) * dble(k)
    gflops_first = (flops/1.0d9) / max(t_first,1.0d-12)
    gflops_rest  = (flops/1.0d9) / max(t_rest,1.0d-12)
    write(*,'(a,i6,a,i6,a,i6)') 'GEMM size m=',m,' n=',n,' k=',k
    write(*,'(a,f10.6,a,f10.3,a)') '  first call: ', t_first, ' s, ', gflops_first, ' GF/s'
    write(*,'(a,f10.6,a,f10.3,a)') '  avg (cached): ', t_rest, ' s, ', gflops_rest, ' GF/s'
    deallocate(A,B,C)
  end subroutine bench_gemm

  subroutine bench_syrk()
    implicit none
    integer :: n,k, lda, ldc
    integer :: iters, i
    real(c_double), allocatable :: A(:,:), C(:,:)
    real(c_double) :: alpha, beta
    integer :: c0, c1, rate
    real(c_double) :: t_first, t_rest
    real(c_double) :: flops, gflops_first, gflops_rest

    n = syrk_n; k = syrk_k
    lda = n; ldc = n
    iters = syrk_iters
    allocate(A(lda,k), C(ldc,n))
    call random_seed()
    call random_number(A); C = 0.0d0
    alpha = 1.0d0; beta = 0.0d0

    call system_clock(c0, rate)
    call syrk_cublas('U','N', n, k, alpha, A, lda, beta, C, ldc)
    call system_clock(c1)
    t_first = real(c1-c0,8)/real(rate,8)

    call system_clock(c0, rate)
    do i = 1, iters
      call syrk_cublas('U','N', n, k, alpha, A, lda, beta, C, ldc)
    end do
    call system_clock(c1)
    t_rest = real(c1-c0,8)/real(rate,8)/real(iters,8)
    if (matrix_has_nan(C)) then
      call record_bench_fail('SYRK produced non-finite output')
      deallocate(A,C)
      return
    end if

    ! Effective flops; many report n*n*k for triangular update
    if (syrk_full) then
      flops = 2.0d0 * dble(n) * dble(n) * dble(k)
    else
      flops = dble(n) * dble(n) * dble(k)
    end if
    gflops_first = (flops/1.0d9) / max(t_first,1.0d-12)
    gflops_rest  = (flops/1.0d9) / max(t_rest,1.0d-12)
    write(*,'(a,i6,a,i6)') 'SYRK size n=',n,' k=',k
    write(*,'(a,f10.6,a,f10.3,a)') '  first call: ', t_first, ' s, ', gflops_first, ' GF/s'
    write(*,'(a,f10.6,a,f10.3,a)') '  avg (cached): ', t_rest, ' s, ', gflops_rest, ' GF/s'
    deallocate(A,C)
  end subroutine bench_syrk

  subroutine bench_dsyevd()
    implicit none
    integer :: n, lda, iters, i
    real(c_double), allocatable :: A(:,:), W(:)
    integer(c_int) :: info
    integer :: c0, c1, rate
    real(c_double) :: t_first, t_rest
    real(c_double) :: flops, gflops_first, gflops_rest

    n = dsy_n; lda = n; iters = dsy_iters
    allocate(A(lda,n), W(n))
    call random_seed()
    call random_number(A)
    ! Symmetrize
    do i = 1, n
      A(i,i) = A(i,i) + 10.0d0
    end do
    A = 0.5d0*(A + transpose(A))

    call system_clock(c0, rate)
    call mopac_cuda_dsyevd(n, A, lda, W, info)
    call system_clock(c1)
    if (info /= 0_c_int) then
      call record_bench_fail_info('DSYEVD first', info)
      deallocate(A,W)
      return
    end if
    t_first = real(c1-c0,8)/real(rate,8)

    do i = 1, n
      A(i,i) = A(i,i) + 10.0d0
    end do
    A = 0.5d0*(A + transpose(A))

    call system_clock(c0, rate)
    do i = 1, iters
      call mopac_cuda_dsyevd(n, A, lda, W, info)
      if (info /= 0_c_int) then
        call record_bench_fail_info('DSYEVD cached', info)
        deallocate(A,W)
        return
      end if
    end do
    call system_clock(c1)
    t_rest = real(c1-c0,8)/real(rate,8)/real(iters,8)

    flops = 4.5d0 * dble(n) * dble(n) * dble(n)
    gflops_first = (flops/1.0d9) / max(t_first,1.0d-12)
    gflops_rest  = (flops/1.0d9) / max(t_rest,1.0d-12)
    write(*,'(a,i6)') 'DSYEVD size n=', n
    write(*,'(a,f10.6,a,f10.3,a)') '  first call: ', t_first, ' s, ', gflops_first, ' GF/s (approx)'
    write(*,'(a,f10.6,a,f10.3,a)') '  avg (cached): ', t_rest, ' s, ', gflops_rest, ' GF/s (approx)'
    deallocate(A,W)
  end subroutine bench_dsyevd

  subroutine print_help()
    implicit none
    write(*,*) 'Usage: mopac-gpu-bench [--gemm=m,n,k,iters]', &
               ' [--syrk=n,k,iters] [--syrk-full]', &
               ' [--dsyevd=n,iters] [--rot1=n,iters] [--rot2=n,iters]', &
               ' [--accuracy=n,k] [--accuracy-only]'
  end subroutine print_help

  subroutine parse_ints(str, a, b, c, d)
    character(len=*), intent(in) :: str
    integer, intent(inout) :: a
    integer, intent(inout), optional :: b, c, d
    character(len=len(str)) :: s
    integer :: i1
    s = str
    do i1 = 1, len_trim(s)
      if (s(i1:i1) == ',') s(i1:i1) = ' '
    end do
    if (present(d)) then
      read(s,*,err=99) a, b, c, d
    else if (present(c)) then
      read(s,*,err=99) a, b, c
    else if (present(b)) then
      read(s,*,err=99) a, b
    else
      read(s,*,err=99) a
    end if
99  continue
  end subroutine parse_ints

  subroutine parse_args()
    implicit none
    integer :: argc, i
    character(len=256) :: arg, val
    argc = command_argument_count()
    do i = 1, argc
      call get_command_argument(i, arg)
      if (index(arg, '--help') == 1) then
        call print_help()
        stop
      else if (index(arg, '--gemm=') == 1) then
        val = arg(8:)
        call parse_ints(val, gemm_m, gemm_n, gemm_k, gemm_iters)
      else if (index(arg, '--syrk=') == 1) then
        val = arg(8:)
        call parse_ints(val, syrk_n, syrk_k, syrk_iters)
      else if (index(arg, '--dsyevd=') == 1) then
        val = arg(10:)
        call parse_ints(val, dsy_n, dsy_iters)
      else if (index(arg, '--accuracy=') == 1) then
        val = arg(12:)
        accuracy_enabled = .true.
        call parse_ints(val, acc_n, acc_k)
      else if (trim(arg) == '--accuracy-only') then
        accuracy_enabled = .true.
        accuracy_only = .true.
      else if (trim(arg) == '--syrk-full') then
        syrk_full = .true.
      else if (index(arg, '--rot1=') == 1) then
        val = arg(8:)
        call parse_ints(val, rot1_n, rot1_iters)
      else if (index(arg, '--rot2=') == 1) then
        val = arg(8:)
        call parse_ints(val, rot2_n, rot2_iters)
      end if
    end do
  end subroutine parse_args

  subroutine bench_accuracy()
    implicit none
    integer :: n, k
    n = max(4, acc_n)
    k = max(1, acc_k)
    call bench_accuracy_gemm(n, k)
    call bench_accuracy_syrk(n, k)
    call bench_accuracy_dsyevd(n)
    call bench_accuracy_rot(n)
  end subroutine bench_accuracy

  subroutine bench_accuracy_gemm(n, k)
    implicit none
    integer, intent(in) :: n, k
    real(c_double), allocatable :: A(:,:), B(:,:), C_gpu(:,:), C_ref(:,:)
    real(c_double) :: max_abs, rms_abs, rel_rms

    allocate(A(n,k), B(k,n), C_gpu(n,n), C_ref(n,n))
    call random_seed()
    call random_number(A)
    call random_number(B)
    C_gpu = 0.0d0
    C_ref = matmul(A, B)
    call gemm_cublas('N','N', n, n, k, 1.0d0, A, n, B, k, 0.0d0, C_gpu, n)
    call error_stats_full(C_ref, C_gpu, max_abs, rms_abs, rel_rms)
    call print_accuracy3('GEMM', n, k, max_abs, rms_abs, rel_rms)
    call validate_accuracy3('GEMM', max_abs, rms_abs, rel_rms)
    deallocate(A, B, C_gpu, C_ref)
  end subroutine bench_accuracy_gemm

  subroutine bench_accuracy_syrk(n, k)
    implicit none
    integer, intent(in) :: n, k
    real(c_double), allocatable :: A(:,:), C_gpu(:,:), C_ref(:,:)
    real(c_double) :: max_abs, rms_abs, rel_rms

    allocate(A(n,k), C_gpu(n,n), C_ref(n,n))
    call random_seed()
    call random_number(A)
    C_gpu = 0.0d0
    C_ref = matmul(A, transpose(A))
    call syrk_cublas('U','N', n, k, 1.0d0, A, n, 0.0d0, C_gpu, n)
    call error_stats_upper(C_ref, C_gpu, max_abs, rms_abs, rel_rms)
    call print_accuracy3('SYRK', n, k, max_abs, rms_abs, rel_rms)
    call validate_accuracy3('SYRK', max_abs, rms_abs, rel_rms)
    deallocate(A, C_gpu, C_ref)
  end subroutine bench_accuracy_syrk

  subroutine bench_accuracy_dsyevd(n)
    implicit none
    integer, intent(in) :: n
    integer :: i
    integer(c_int) :: info
    real(c_double), allocatable :: A(:,:), A0(:,:), W(:), AV(:,:), VD(:,:), Gram(:,:)
    real(c_double) :: residual, orthogonality, anorm

    allocate(A(n,n), A0(n,n), W(n), AV(n,n), VD(n,n), Gram(n,n))
    call random_seed()
    call random_number(A0)
    do i = 1, n
      A0(i,i) = A0(i,i) + 10.0d0
    end do
    A0 = 0.5d0*(A0 + transpose(A0))
    A = A0
    W = 0.0d0
    call mopac_cuda_dsyevd(n, A, n, W, info)

    AV = matmul(A0, A)
    VD = A
    do i = 1, n
      VD(:,i) = VD(:,i) * W(i)
    end do
    anorm = max(fro_norm(A0), tiny(1.0d0))
    residual = fro_norm(AV - VD) / anorm

    Gram = matmul(transpose(A), A)
    do i = 1, n
      Gram(i,i) = Gram(i,i) - 1.0d0
    end do
    orthogonality = fro_norm(Gram) / dble(n)
    write(*,'(a,i6,a,1pe12.4,a,1pe12.4,a,i0)') 'ACCURACY DSYEVD n=', n, &
        ' residual=', residual, ' orthogonality=', orthogonality, ' info=', info
    call validate_accuracy_dsyevd(residual, orthogonality, info)
    deallocate(A, A0, W, AV, VD, Gram)
  end subroutine bench_accuracy_dsyevd

  subroutine bench_accuracy_rot(n_in)
    implicit none
    integer, intent(in) :: n_in
    integer :: n, nocc, lumo, i
    real(c_double), allocatable :: eig(:), vector0(:,:), vector_ref(:,:)
    real(c_double), allocatable :: vector_single(:,:), vector_2gpu(:,:), fmo(:)
    real(c_double), allocatable :: ci0(:), ca0(:)
    real(c_double) :: max_abs, rms_abs, rel_rms
    real(c_double) :: bigeps, tinyv

    n = n_in
    if (mod(n, 2) /= 0) n = n + 1
    nocc = n/2
    lumo = nocc + 1
    bigeps = 1.0d-5
    tinyv = 1.0d-12

    allocate(eig(n), vector0(n,n), vector_ref(n,n), vector_single(n,n), vector_2gpu(n,n))
    allocate(fmo(nocc * (n - nocc)))
    allocate(ci0(max(1, n * nocc)), ca0(max(1, n * (n - nocc))))
    call random_seed()
    call random_number(vector0)
    call random_number(fmo)
    do i = 1, n
      eig(i) = 0.25d0 * dble(i)
    end do
    fmo = 1.0d-3 * fmo
    ci0 = 0.0d0
    ca0 = 0.0d0
    vector_ref = vector0
    vector_single = vector0
    vector_2gpu = vector0

    call apply_rot_cpu(fmo, eig, vector_ref, nocc, lumo, n, bigeps, tinyv)
    call call_rot_cuda_gpu(fmo, eig, vector_single, ci0, ca0, nocc, lumo, n, bigeps, tinyv)
    call call_rot_cuda_2gpu_gpu(fmo, eig, vector_2gpu, ci0, ca0, nocc, lumo, n, bigeps, tinyv)
    call error_stats_two_full(vector_ref, vector_single, vector_2gpu, max_abs, rms_abs, rel_rms)
    call print_accuracy3('ROT_SINGLE_VS_2GPU', n, nocc, max_abs, rms_abs, rel_rms)
    call validate_accuracy3('ROT_SINGLE_VS_2GPU', max_abs, rms_abs, rel_rms)
    deallocate(eig, vector0, vector_ref, vector_single, vector_2gpu, fmo, ci0, ca0)
  end subroutine bench_accuracy_rot

  subroutine print_accuracy3(label, n, k, max_abs, rms_abs, rel_rms)
    implicit none
    character(len=*), intent(in) :: label
    integer, intent(in) :: n, k
    real(c_double), intent(in) :: max_abs, rms_abs, rel_rms
    write(*,'(a,a,a,i6,a,i6,a,1pe12.4,a,1pe12.4,a,1pe12.4)') 'ACCURACY ', &
        trim(label), ' n=', n, ' k=', k, ' max_abs=', max_abs, &
        ' rms_abs=', rms_abs, ' rel_rms=', rel_rms
  end subroutine print_accuracy3

  subroutine validate_accuracy3(label, max_abs, rms_abs, rel_rms)
    implicit none
    character(len=*), intent(in) :: label
    real(c_double), intent(in) :: max_abs, rms_abs, rel_rms

    call check_accuracy_metric(label, 'max_abs', max_abs, accuracy_tolerance(label, 'max_abs'))
    call check_accuracy_metric(label, 'rms_abs', rms_abs, accuracy_tolerance(label, 'rms_abs'))
    call check_accuracy_metric(label, 'rel_rms', rel_rms, accuracy_tolerance(label, 'rel_rms'))
  end subroutine validate_accuracy3

  subroutine validate_accuracy_dsyevd(residual, orthogonality, info)
    implicit none
    real(c_double), intent(in) :: residual, orthogonality
    integer(c_int), intent(in) :: info

    call check_accuracy_metric('DSYEVD', 'residual', residual, DSYEVD_RESIDUAL_TOL)
    call check_accuracy_metric('DSYEVD', 'orthogonality', orthogonality, DSYEVD_ORTHOGONALITY_TOL)
    if (info /= 0_c_int) then
      write(*,'(a,i0)') 'ACCURACY_FAIL DSYEVD metric=info value=', info
      accuracy_failures = accuracy_failures + 1
    end if
  end subroutine validate_accuracy_dsyevd

  subroutine record_bench_fail_info(label, info)
    implicit none
    character(len=*), intent(in) :: label
    integer(c_int), intent(in) :: info

    write(*,'(a,a,a,i0)') 'BENCH_FAIL ', trim(label), ' info=', info
    accuracy_failures = accuracy_failures + 1
  end subroutine record_bench_fail_info

  subroutine record_bench_fail(label)
    implicit none
    character(len=*), intent(in) :: label

    write(*,'(a,a)') 'BENCH_FAIL ', trim(label)
    accuracy_failures = accuracy_failures + 1
  end subroutine record_bench_fail

  logical function matrix_has_nan(values)
    implicit none
    real(c_double), intent(in) :: values(:,:)

    matrix_has_nan = any(values /= values)
  end function matrix_has_nan

  subroutine check_accuracy_metric(label, metric, value, tolerance)
    implicit none
    character(len=*), intent(in) :: label, metric
    real(c_double), intent(in) :: value, tolerance

    if (.not. metric_is_finite(value)) then
      write(*,'(a,a,a,a,a,1pe12.4)') 'ACCURACY_FAIL ', trim(label), &
          ' metric=', trim(metric), ' value=', value
      accuracy_failures = accuracy_failures + 1
    else if (abs(value) > tolerance) then
      write(*,'(a,a,a,a,a,1pe12.4,a,1pe12.4)') 'ACCURACY_FAIL ', trim(label), &
          ' metric=', trim(metric), ' value=', value, ' tol=', tolerance
      accuracy_failures = accuracy_failures + 1
    end if
  end subroutine check_accuracy_metric

  function metric_is_finite(value) result(ok)
    implicit none
    real(c_double), intent(in) :: value
    logical :: ok

    ok = (value == value) .and. (abs(value) <= huge(value))
  end function metric_is_finite

  function accuracy_tolerance(label, metric) result(tolerance)
    implicit none
    character(len=*), intent(in) :: label, metric
    real(c_double) :: tolerance

    tolerance = 0.0d0
    select case (trim(label))
    case ('GEMM')
      select case (trim(metric))
      case ('max_abs')
        tolerance = GEMM_MAX_ABS_TOL
      case ('rms_abs')
        tolerance = GEMM_RMS_ABS_TOL
      case ('rel_rms')
        tolerance = GEMM_REL_RMS_TOL
      end select
    case ('SYRK')
      select case (trim(metric))
      case ('max_abs')
        tolerance = SYRK_MAX_ABS_TOL
      case ('rms_abs')
        tolerance = SYRK_RMS_ABS_TOL
      case ('rel_rms')
        tolerance = SYRK_REL_RMS_TOL
      end select
    case ('ROT_SINGLE_VS_2GPU')
      select case (trim(metric))
      case ('max_abs')
        tolerance = ROT_MAX_ABS_TOL
      case ('rms_abs')
        tolerance = ROT_RMS_ABS_TOL
      case ('rel_rms')
        tolerance = ROT_REL_RMS_TOL
      end select
    end select
  end function accuracy_tolerance

  subroutine error_stats_full(ref, got, max_abs, rms_abs, rel_rms)
    implicit none
    real(c_double), intent(in) :: ref(:,:), got(:,:)
    real(c_double), intent(out) :: max_abs, rms_abs, rel_rms
    integer :: i, j, count
    real(c_double) :: err, sum_err2, sum_ref2

    max_abs = 0.0d0
    sum_err2 = 0.0d0
    sum_ref2 = 0.0d0
    count = 0
    do j = 1, size(ref, 2)
      do i = 1, size(ref, 1)
        err = got(i,j) - ref(i,j)
        max_abs = max(max_abs, abs(err))
        sum_err2 = sum_err2 + err*err
        sum_ref2 = sum_ref2 + ref(i,j)*ref(i,j)
        count = count + 1
      end do
    end do
    call finish_error_stats(sum_err2, sum_ref2, count, rms_abs, rel_rms)
  end subroutine error_stats_full

  subroutine error_stats_two_full(ref, got1, got2, max_abs, rms_abs, rel_rms)
    implicit none
    real(c_double), intent(in) :: ref(:,:), got1(:,:), got2(:,:)
    real(c_double), intent(out) :: max_abs, rms_abs, rel_rms
    integer :: i, j, count
    real(c_double) :: err1, err2, sum_err2, sum_ref2

    max_abs = 0.0d0
    sum_err2 = 0.0d0
    sum_ref2 = 0.0d0
    count = 0
    do j = 1, size(ref, 2)
      do i = 1, size(ref, 1)
        err1 = got1(i,j) - ref(i,j)
        err2 = got2(i,j) - ref(i,j)
        max_abs = max(max_abs, abs(err1), abs(err2))
        sum_err2 = sum_err2 + err1*err1 + err2*err2
        sum_ref2 = sum_ref2 + 2.0d0 * ref(i,j)*ref(i,j)
        count = count + 2
      end do
    end do
    call finish_error_stats(sum_err2, sum_ref2, count, rms_abs, rel_rms)
  end subroutine error_stats_two_full

  subroutine error_stats_upper(ref, got, max_abs, rms_abs, rel_rms)
    implicit none
    real(c_double), intent(in) :: ref(:,:), got(:,:)
    real(c_double), intent(out) :: max_abs, rms_abs, rel_rms
    integer :: i, j, count
    real(c_double) :: err, sum_err2, sum_ref2

    max_abs = 0.0d0
    sum_err2 = 0.0d0
    sum_ref2 = 0.0d0
    count = 0
    do j = 1, size(ref, 2)
      do i = 1, j
        err = got(i,j) - ref(i,j)
        max_abs = max(max_abs, abs(err))
        sum_err2 = sum_err2 + err*err
        sum_ref2 = sum_ref2 + ref(i,j)*ref(i,j)
        count = count + 1
      end do
    end do
    call finish_error_stats(sum_err2, sum_ref2, count, rms_abs, rel_rms)
  end subroutine error_stats_upper

  subroutine error_stats_pair(ref1, ref2, got1, got2, max_abs, rms_abs, rel_rms)
    implicit none
    real(c_double), intent(in) :: ref1(:), ref2(:), got1(:), got2(:)
    real(c_double), intent(out) :: max_abs, rms_abs, rel_rms
    integer :: i, count
    real(c_double) :: err, sum_err2, sum_ref2

    max_abs = 0.0d0
    sum_err2 = 0.0d0
    sum_ref2 = 0.0d0
    count = 0
    do i = 1, size(ref1)
      err = got1(i) - ref1(i)
      max_abs = max(max_abs, abs(err))
      sum_err2 = sum_err2 + err*err
      sum_ref2 = sum_ref2 + ref1(i)*ref1(i)
      err = got2(i) - ref2(i)
      max_abs = max(max_abs, abs(err))
      sum_err2 = sum_err2 + err*err
      sum_ref2 = sum_ref2 + ref2(i)*ref2(i)
      count = count + 2
    end do
    call finish_error_stats(sum_err2, sum_ref2, count, rms_abs, rel_rms)
  end subroutine error_stats_pair

  subroutine finish_error_stats(sum_err2, sum_ref2, count, rms_abs, rel_rms)
    implicit none
    real(c_double), intent(in) :: sum_err2, sum_ref2
    integer, intent(in) :: count
    real(c_double), intent(out) :: rms_abs, rel_rms
    real(c_double) :: ref_rms

    rms_abs = sqrt(sum_err2 / max(1, count))
    ref_rms = sqrt(sum_ref2 / max(1, count))
    rel_rms = rms_abs / max(ref_rms, tiny(1.0d0))
  end subroutine finish_error_stats

  function fro_norm(A) result(norm)
    implicit none
    real(c_double), intent(in) :: A(:,:)
    real(c_double) :: norm
    norm = sqrt(sum(A*A))
  end function fro_norm

  subroutine apply_rot_cpu(fmo, eig, vector, nocc, lumo, n, bigeps, tinyv)
    implicit none
    integer, intent(in) :: nocc, lumo, n
    real(c_double), intent(in) :: fmo(:), eig(:), bigeps, tinyv
    real(c_double), intent(inout) :: vector(:,:)
    integer :: i, j, ij, r
    real(c_double) :: x, a, b, d, e, alpha, beta, vi, vj

    ij = 0
    do i = 1, nocc
      do j = lumo, n
        ij = ij + 1
        x = fmo(ij)
        if (abs(x) < tinyv) cycle
        a = eig(i)
        b = eig(j)
        d = a - b
        if (d /= 0.0d0) then
          if (abs(x / d) < bigeps) cycle
        end if
        e = sign(sqrt(4.0d0*x*x + d*d), d)
        alpha = sqrt(0.5d0*(1.0d0 + d/e))
        beta = -sign(sqrt(1.0d0 - alpha*alpha), x)
        do r = 1, n
          vi = vector(r,i)
          vj = vector(r,j)
          vector(r,i) = alpha * vi + beta * vj
          vector(r,j) = alpha * vj - beta * vi
        end do
      end do
    end do
  end subroutine apply_rot_cpu

  subroutine bench_rot_single()
    use iso_c_binding
    implicit none
    integer :: n, nocc, lumo, iters, i
    integer :: c0, c1, rate
    real(c_double), allocatable :: eig(:), vector(:,:), fmo(:), ci0(:), ca0(:)
    real(c_double) :: t_first, t_rest
    real(c_double) :: bigeps, tiny

    n = rot1_n
    nocc = n/2
    lumo = nocc + 1
    iters = rot1_iters
    bigeps = 1.0d-5
    tiny   = 1.0d-12

    allocate(eig(n), vector(n,n), ci0(n), ca0(n))
    allocate(fmo(nocc * (n - nocc)))
    call random_seed()
    call random_number(eig)
    call random_number(vector)
    call random_number(fmo)
    fmo = 1.0d-3 * fmo
    ci0 = 0.0d0; ca0 = 0.0d0

    call system_clock(c0, rate)
    call call_rot_cuda_gpu(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny)
    call system_clock(c1)
    t_first = real(c1-c0,8)/real(rate,8)

    call system_clock(c0, rate)
    do i = 1, iters
      call call_rot_cuda_gpu(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny)
    end do
    call system_clock(c1)
    t_rest = real(c1-c0,8)/real(rate,8)/real(iters,8)
    if (matrix_has_nan(vector)) then
      call record_bench_fail('ROT single produced non-finite output')
      deallocate(eig, vector, fmo, ci0, ca0)
      return
    end if

    write(*,'(a,i6,a,i6)') 'ROT single n=', n, ' nocc=', nocc
    write(*,'(a,f10.6,a)') '  first call: ', t_first, ' s'
    write(*,'(a,f10.6,a)') '  avg (cached): ', t_rest, ' s'

    deallocate(eig, vector, fmo, ci0, ca0)
  end subroutine bench_rot_single

  subroutine bench_rot_2gpu()
    use iso_c_binding
    implicit none
    integer :: n, nocc, lumo, iters, i
    integer :: c0, c1, rate
    real(c_double), allocatable :: eig(:), vector(:,:), fmo(:), ci0(:), ca0(:)
    real(c_double) :: t_first, t_rest
    real(c_double) :: bigeps, tiny

    n = rot2_n
    nocc = n/2
    lumo = nocc + 1
    iters = rot2_iters
    bigeps = 1.0d-5
    tiny   = 1.0d-12

    allocate(eig(n), vector(n,n), ci0(n), ca0(n))
    allocate(fmo(nocc * (n - nocc)))
    call random_seed()
    call random_number(eig)
    call random_number(vector)
    call random_number(fmo)
    fmo = 1.0d-3 * fmo
    ci0 = 0.0d0; ca0 = 0.0d0

    call system_clock(c0, rate)
    call call_rot_cuda_2gpu_gpu(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny)
    call system_clock(c1)
    t_first = real(c1-c0,8)/real(rate,8)

    call system_clock(c0, rate)
    do i = 1, iters
      call call_rot_cuda_2gpu_gpu(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny)
    end do
    call system_clock(c1)
    t_rest = real(c1-c0,8)/real(rate,8)/real(iters,8)
    if (matrix_has_nan(vector)) then
      call record_bench_fail('ROT 2-GPU produced non-finite output')
      deallocate(eig, vector, fmo, ci0, ca0)
      return
    end if

    write(*,'(a,i6,a,i6)') 'ROT 2-GPU n=', n, ' nocc=', nocc
    write(*,'(a,f10.6,a)') '  first call: ', t_first, ' s'
    write(*,'(a,f10.6,a)') '  avg (cached): ', t_rest, ' s'

    deallocate(eig, vector, fmo, ci0, ca0)
  end subroutine bench_rot_2gpu

end program gpu_bench
