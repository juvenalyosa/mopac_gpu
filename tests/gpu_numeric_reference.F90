program gpu_numeric_reference
  use iso_fortran_env, only : real64
  use molkst_C, only : norbs, mpack
  use chanel_C, only : iw
#ifdef GPU
  use iso_c_binding, only : c_bool, c_int
  use gpu_fock_interfaces, only : mopac_cuda_fock2_scf
#endif
  implicit none
  integer :: failures
  failures = 0
#ifdef GPU
  call init_console()
  if (.not. test_light_light()) failures = failures + 1
  if (.not. test_general_pair()) failures = failures + 1
#else
  print *, 'GPU support not enabled; skipping gpu_numeric_reference'
#endif
  if (failures /= 0) stop 1
contains
#ifdef GPU
  subroutine init_console()
    implicit none
    open(unit=iw, file='gpu_numeric_reference.log', status='replace', action='write')
  end subroutine init_console

  subroutine reset_dimensions(n)
    integer, intent(in) :: n
    call setup_indices(n)
  end subroutine reset_dimensions

  subroutine setup_indices(n)
    integer, intent(in) :: n
    norbs = n
    mpack = n * (n + 1) / 2
  end subroutine setup_indices

  logical function test_light_light()
    implicit none
    integer, parameter :: numat = 2
    integer(c_int) :: nfirst(numat), nlast(numat)
    integer :: local_norbs, local_mpack
    double precision :: ptot(3), p(3), w(1), wj(1), wk(1)
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
    wj = 0.0d0
    wk = 0.0d0
    f_cpu = 0.0d0
    f_gpu = 0.0d0

    call cpu_fock_light_light(ptot, p, w, f_cpu)

    ok = mopac_cuda_fock2_scf(local_norbs, local_mpack, numat, nfirst, nlast, ptot, p, w, wj, wk, 0_c_int, f_gpu)
    if (.not. ok) then
      write(iw, '(a)') '[LIGHT-LIGHT] GPU path unavailable; skipping'
      test_light_light = .true.
      return
    end if

    diff = maxval(abs(f_cpu - f_gpu))
    denom = max(1.0d0, maxval(abs(f_cpu)))
    write(iw,'(a,1pe18.10)') '[LIGHT-LIGHT] max abs diff = ', diff
    write(iw,'(a,3(1pe14.6,1x))') '  CPU:', f_cpu
    write(iw,'(a,3(1pe14.6,1x))') '  GPU:', f_gpu
    if (diff > 1.0d-8 .and. diff/denom > 1.0d-8) then
      write(iw,'(a)') '[LIGHT-LIGHT] CPU and GPU results differ beyond tolerance'
      return
    end if
    test_light_light = .true.
  end function test_light_light

  logical function test_general_pair()
    implicit none
    integer, parameter :: numat = 2
    integer(c_int) :: nfirst(numat), nlast(numat)
    integer :: local_norbs, local_mpack
    double precision, allocatable :: ptot(:), p(:), w(:), wj(:), wk(:)
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

    len_w = span_count(3,4)
    len_w = pair_count(len_w)
    len_w = len_w * pair_count(span_count(1,2))
    if (len_w <= 0) then
      write(iw,'(a)') '[GENERAL] no integrals to process'
      test_general_pair = .true.
      deallocate(ptot, p, f_cpu, f_gpu)
      return
    end if
    allocate(w(len_w))
    do i = 1, len_w
      w(i) = 1.0d-4 * dble(i)
    end do
    allocate(wj(1), wk(1))
    wj = 0.0d0
    wk = 0.0d0

    call cpu_fock_general(3, 4, 1, 2, ptot, p, w, f_cpu)

    ok = mopac_cuda_fock2_scf(local_norbs, local_mpack, numat, nfirst, nlast, ptot, p, w, wj, wk, 0_c_int, f_gpu)
    if (.not. ok) then
      write(iw,'(a)') '[GENERAL] GPU path unavailable; skipping'
      test_general_pair = .true.
      deallocate(ptot, p, f_cpu, f_gpu, w, wj, wk)
      return
    end if

    diff = maxval(abs(f_cpu - f_gpu))
    denom = max(1.0d0, maxval(abs(f_cpu)))
    write(iw,'(a,1pe18.10)') '[GENERAL] max abs diff = ', diff
    write(iw,'(a)') '  CPU:'
    do i = 1, local_mpack
      write(iw,'(1pe14.6,1x)', advance='no') f_cpu(i)
      if (mod(i,6) == 0 .or. i == local_mpack) write(iw,*)
    end do
    write(iw,'(a)') '  GPU:'
    do i = 1, local_mpack
      write(iw,'(1pe14.6,1x)', advance='no') f_gpu(i)
      if (mod(i,6) == 0 .or. i == local_mpack) write(iw,*)
    end do
    if (diff > 1.0d-8 .and. diff/denom > 1.0d-8) then
      write(iw,'(a)') '[GENERAL] CPU and GPU results differ beyond tolerance'
      deallocate(ptot, p, f_cpu, f_gpu, w, wj, wk)
      return
    end if
    test_general_pair = .true.
    deallocate(ptot, p, f_cpu, f_gpu, w, wj, wk)
  end function test_general_pair

  subroutine cpu_fock_light_light(ptot, p, w, f)
    implicit none
    double precision, intent(in) :: ptot(:), p(:), w(:)
    double precision, intent(inout) :: f(:)
    integer :: ii, jj, ij
    double precision :: val

    f = 0.0d0
    val = w(1)
    ii = packed_index(1, 1)
    jj = packed_index(2, 2)
    ij = packed_index(2, 1)
    f(ii) = f(ii) + val * ptot(jj)
    f(jj) = f(jj) + val * ptot(ii)
    f(ij) = f(ij) - val * p(ij)
  end subroutine cpu_fock_light_light

  subroutine cpu_fock_general(ia, ib, ja, jb, ptot, p, w, f)
    implicit none
    integer, intent(in) :: ia, ib, ja, jb
    double precision, intent(in) :: ptot(:), p(:), w(:)
    double precision, intent(inout) :: f(:)
    integer :: i, j, k, l, kr
    double precision :: aa, bb, a, exch
    integer :: ij, kl, ik, il, jk, jl
    logical :: have_ik, have_il, have_jk, have_jl

    f = 0.0d0
    kr = 0
    do i = ia, ib
      do j = ia, i
        aa = 2.0d0
        if (i == j) aa = 1.0d0
        ij = packed_index(i, j)
        do k = ja, jb
          do l = ja, k
            bb = 2.0d0
            if (k == l) bb = 1.0d0
            kl = packed_index(k, l)
            have_ik = (i >= k)
            have_il = (i >= l)
            have_jk = (j >= k)
            have_jl = (j >= l)
            kr = kr + 1
            if (kr > size(w)) then
              write(iw,'(a)') '[cpu_fock_general] insufficient integrals provided'
              stop 1
            end if
            a = w(kr)
            f(ij) = f(ij) + bb * a * ptot(kl)
            f(kl) = f(kl) + aa * a * ptot(ij)
            exch = a * aa * bb * 0.25d0
            if (have_ik .and. have_jl) then
              ik = packed_index(i, k)
              jl = packed_index(j, l)
              f(ik) = f(ik) - exch * p(jl)
            end if
            if (have_il .and. have_jk) then
              il = packed_index(i, l)
              jk = packed_index(j, k)
              f(il) = f(il) - exch * p(jk)
              f(jk) = f(jk) - exch * p(il)
            end if
            if (have_jl .and. have_ik) then
              jl = packed_index(j, l)
              ik = packed_index(i, k)
              f(jl) = f(jl) - exch * p(ik)
            end if
          end do
        end do
      end do
    end do
  end subroutine cpu_fock_general

  integer function packed_index(i, j)
    implicit none
    integer, intent(in) :: i, j
    integer :: ii, jj
    ii = i
    jj = j
    if (ii >= jj) then
      packed_index = (ii * (ii - 1)) / 2 + jj
    else
      packed_index = (jj * (jj - 1)) / 2 + ii
    end if
  end function packed_index

  integer function span_count(first, last)
    implicit none
    integer, intent(in) :: first, last
    if (last >= first) then
      span_count = last - first + 1
    else
      span_count = 0
    end if
  end function span_count

  integer function pair_count(span)
    implicit none
    integer, intent(in) :: span
    if (span > 0) then
      pair_count = span * (span + 1) / 2
    else
      pair_count = 0
    end if
  end function pair_count
#endif
end program gpu_numeric_reference
