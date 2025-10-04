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

    len_w = pair_count(span_count(3, 4))
    len_w = len_w * pair_count(span_count(1, 2))
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
    allocate(wj(1), wk(1))
    wj = 0.0d0
    wk = 0.0d0

    call cpu_fock_general(3, 4, 1, 2, ptot, p, w, f_cpu)

    ok = mopac_cuda_fock2_scf(local_norbs, local_mpack, numat, nfirst, nlast, ptot, p, w, wj, wk, 0_c_int, f_gpu)
    if (.not. ok) then
      print *, '[GENERAL] GPU path unavailable; skipping'
      test_general_pair = .true.
      deallocate(ptot, p, f_cpu, f_gpu, w, wj, wk)
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
      deallocate(ptot, p, f_cpu, f_gpu, w, wj, wk)
      return
    end if
    test_general_pair = .true.
    deallocate(ptot, p, f_cpu, f_gpu, w, wj, wk)
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

    allocate(ptot(local_mpack), p(local_mpack), w(6))
    ptot = (/1.05d0, 0.02d0, 0.01d0, 0.98d0, 0.03d0, 0.90d0/)
    p    = (/1.00d0, 0.01d0, 0.01d0, 0.95d0, 0.02d0, 0.88d0/)
    w    = (/0.10d0, 0.12d0, 0.14d0, 0.16d0, 0.18d0, 0.20d0/)

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

    diff = maxval(abs(f_cpu - f_gpu(1:local_mpack)))
    denom = max(1.0d0, maxval(abs(f_cpu)))
    print '(a,1pe18.10)', '[HEAVY-LIGHT] max abs diff = ', diff
    print '(a)', '[HEAVY-LIGHT] CPU:'
    write(*,'(6(1pe14.6,1x))') f_cpu(1:local_mpack)
    print '(a)', '  GPU:'
    write(*,'(6(1pe14.6,1x))') f_gpu(1:local_mpack)
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
    integer :: ij, kl

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
            kr = kr + 1
            if (kr > size(w)) then
              print *, '[cpu_fock_general] insufficient integrals provided'
              stop 1
            end if
            a = w(kr)
            f(ij) = f(ij) + bb * a * ptot(kl)
            f(kl) = f(kl) + aa * a * ptot(ij)
            exch = a * aa * bb * 0.25d0
            if (i >= k .and. j >= l) f(packed_index(i, k)) = f(packed_index(i, k)) - exch * p(packed_index(j, l))
            if (i >= l .and. j >= k) then
              f(packed_index(i, l)) = f(packed_index(i, l)) - exch * p(packed_index(j, k))
              f(packed_index(j, k)) = f(packed_index(j, k)) - exch * p(packed_index(i, l))
            end if
            if (j >= l .and. i >= k) f(packed_index(j, l)) = f(packed_index(j, l)) - exch * p(packed_index(i, k))
          end do
        end do
      end do
    end do
  end subroutine cpu_fock_general

 subroutine cpu_heavy_light_reference(heavy_start, heavy_end, light_atom, ptot, p, w, f)
    implicit none
    integer, intent(in) :: heavy_start, heavy_end, light_atom
    double precision, intent(in) :: ptot(:), p(:), w(:)
    double precision, intent(inout) :: f(:)
    integer :: span, coulomb_len, offset
    integer :: rel, relj, orb_i, orb_j, map
    double precision :: ptot_ll, sumdia, sumoff, val, acc

    f = 0.0d0
    span = heavy_end - heavy_start + 1
    if (span <= 0) return
    coulomb_len = span * (span + 1) / 2
    ptot_ll = ptot(packed_index(light_atom, light_atom))
    sumdia = 0.0d0
    sumoff = 0.0d0
    offset = 0
    do rel = 0, span - 1
      orb_i = heavy_start + rel
      if (rel > 0) then
        do relj = 0, rel - 1
          orb_j = heavy_start + relj
          if (offset < coulomb_len) then
            val = w(offset + 1)
          else
            val = 0.0d0
          end if
          offset = offset + 1
          f(packed_index(orb_i, orb_j)) = f(packed_index(orb_i, orb_j)) + ptot_ll * val
          sumoff = sumoff + ptot(packed_index(orb_i, orb_j)) * val
        end do
      end if
      if (offset < coulomb_len) then
        val = w(offset + 1)
      else
        val = 0.0d0
      end if
      offset = offset + 1
      f(packed_index(orb_i, orb_i)) = f(packed_index(orb_i, orb_i)) + ptot_ll * val
      sumdia = sumdia + ptot(packed_index(orb_i, orb_i)) * val
    end do
    f(packed_index(light_atom, light_atom)) = f(packed_index(light_atom, light_atom)) + sumoff * 2.0d0 + sumdia

    do rel = 0, span - 1
      orb_i = heavy_start + rel
      acc = 0.0d0
      do relj = 0, span - 1
        map = jindex_lookup(rel + 1, span, relj + 1)
        if (map <= 0 .or. map > coulomb_len) cycle
        acc = acc + p(packed_index(heavy_start + relj, light_atom)) * w(map)
      end do
      f(packed_index(orb_i, light_atom)) = f(packed_index(orb_i, light_atom)) - acc
    end do
  end subroutine cpu_heavy_light_reference

  integer function jindex_lookup(row, span, col)
    implicit none
    integer, intent(in) :: row, span, col
    integer :: offset
    if (span <= 0) then
      jindex_lookup = 0
      return
    end if
    offset = (row - 1) * span + col
    if (offset < 1 .or. offset > 256) then
      jindex_lookup = 0
    else
      jindex_lookup = jindex_table(offset)
    end if
  end function jindex_lookup

  integer function jindex_table(idx)
    implicit none
    integer, intent(in) :: idx
    integer, save :: table(256)
    logical, save :: initialized = .false.
    integer :: i, j, k, l, m, ij, ji, ik, kl, lk

    if (.not. initialized) then
      m = 0
      do i = 1, 4
        do j = 1, 4
          ij = min(i, j)
          ji = i + j - ij
          do k = 1, 4
            do l = 1, 4
              m = m + 1
              ik = min(i, k)
              lk = k + l - min(k, l)
              kl = min(k, l)
              table(m) = (ifact_from(ji) + ij) * 10 + ifact_from(lk) + kl - 10
            end do
          end do
        end do
      end do
      initialized = .true.
    end if

    if (idx < 1 .or. idx > 256) then
      jindex_table = 0
    else
      jindex_table = table(idx)
    end if
  end function jindex_table

  integer function ifact_from(n)
    implicit none
    integer, intent(in) :: n
    ifact_from = n * (n - 1) / 2
  end function ifact_from

  integer function packed_index(i, j)
    implicit none
    integer, intent(in) :: i, j
    if (i >= j) then
      packed_index = (i * (i - 1)) / 2 + j
    else
      packed_index = (j * (j - 1)) / 2 + i
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
