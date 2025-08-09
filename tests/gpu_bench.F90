! Simple GPU benchmark harness for GEMM, SYRK, and DSYEVD wrappers
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

  integer :: i
  ! Tunables with defaults (can be overridden via CLI flags)
  integer :: gemm_m=1024, gemm_n=1024, gemm_k=64, gemm_iters=20
  integer :: syrk_n=1024, syrk_k=64, syrk_iters=20
  integer :: dsy_n=512, dsy_iters=5
  integer :: rot1_n=1024, rot1_iters=10
  integer :: rot2_n=2048, rot2_iters=10
  logical :: syrk_full = .false.

  call parse_args()
  call bench_gemm()
  call bench_syrk()
  call bench_dsyevd()
  call bench_rot_single()
  call bench_rot_2gpu()
  call mopac_cuda_destroy_resources()

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
    t_first = real(c1-c0,8)/real(rate,8)

    do i = 1, n
      A(i,i) = A(i,i) + 10.0d0
    end do
    A = 0.5d0*(A + transpose(A))

    call system_clock(c0, rate)
    do i = 1, iters
      call mopac_cuda_dsyevd(n, A, lda, W, info)
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

  subroutine parse_args()
  implicit none
  integer :: argc, i, p
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
  contains
  subroutine print_help()
    implicit none
    write(*,*) 'Usage: mopac-gpu-bench [--gemm=m,n,k,iters] [--syrk=n,k,iters] [--syrk-full]', &
               ' [--dsyevd=n,iters] [--rot1=n,iters] [--rot2=n,iters]'
  end subroutine print_help
  subroutine parse_ints(str, a, b, c, d)
    character(len=*), intent(in) :: str
    integer, intent(inout) :: a
    integer, intent(inout), optional :: b, c, d
    character(len=len(str)) :: s
    integer :: i1, i2, i3
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
  end subroutine parse_args

end program gpu_bench

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

  write(*,'(a,i6,a,i6)') 'ROT 2-GPU n=', n, ' nocc=', nocc
  write(*,'(a,f10.6,a)') '  first call: ', t_first, ' s'
  write(*,'(a,f10.6,a)') '  avg (cached): ', t_rest, ' s'

  deallocate(eig, vector, fmo, ci0, ca0)
end subroutine bench_rot_2gpu
