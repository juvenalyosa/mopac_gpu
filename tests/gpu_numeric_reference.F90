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
  if (.not. test_heavy_heavy()) failures = failures + 1
  if (.not. test_periodic_pair()) failures = failures + 1
#else
  print *, 'GPU support not enabled; skipping gpu_numeric_reference'
#endif
  if (failures /= 0) stop 1
contains
#ifdef GPU
  subroutine set_dimensions(n)
    integer, intent(in) :: n
    norbs = n
    mpack = n * (n + 1) / 2
  end subroutine set_dimensions

  logical function test_light_light()
    implicit none
    integer, parameter :: numat = 2
    integer(c_int) :: nfirst(numat), nlast(numat)
    double precision :: ptot(3), p(3), w(1)
    double precision :: f_cpu(3), f_gpu(3)
    logical(c_bool) :: ok
    double precision :: diff, denom

    test_light_light = .false.
    call set_dimensions(2)

    nfirst = [1_c_int, 2_c_int]
    nlast  = [1_c_int, 2_c_int]

    ptot = (/1.20d0, 0.05d0, 0.90d0/)
    p    = (/1.18d0, 0.02d0, 0.88d0/)
    w(1) = 0.45d0

    call cpu_fock_light_light(ptot, p, w, f_cpu)

    ok = mopac_cuda_fock2_scf(2, 3, numat, nfirst, nlast, ptot, p, w, w, w, 0_c_int, f_gpu)
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
    if (diff > 1.0d-8 .and. diff/denom > 1.0d-8) return
    test_light_light = .true.
  end function test_light_light

  logical function test_general_pair()
    implicit none
    integer, parameter :: numat = 2
    integer(c_int) :: nfirst(numat), nlast(numat)
    double precision, allocatable :: ptot(:), p(:), w(:)
    double precision, allocatable :: f_cpu(:), f_gpu(:)
    logical(c_bool) :: ok
    double precision :: diff, denom
    integer :: len_w, i

    test_general_pair = .false.
    call set_dimensions(4)

    allocate(ptot(mpack), p(mpack), f_cpu(mpack), f_gpu(mpack))
    do i = 1, mpack
      ptot(i) = 0.02d0 * dble(i)
      p(i)    = 0.015d0 * dble(i)
    end do

    nfirst = [1_c_int, 3_c_int]
    nlast  = [2_c_int, 4_c_int]

    len_w = pair_count(span_count(3, 4)) * pair_count(span_count(1, 2))
    allocate(w(len_w))
    do i = 1, len_w
      w(i) = 1.0d-4 * dble(i)
    end do

    call cpu_fock_general(3, 4, 1, 2, ptot, p, w, f_cpu)

    ok = mopac_cuda_fock2_scf(4, mpack, numat, nfirst, nlast, ptot, p, w, w, w, 0_c_int, f_gpu)
    if (.not. ok) then
      print *, '[GENERAL] GPU path unavailable; skipping'
      test_general_pair = .true.
      deallocate(ptot, p, f_cpu, f_gpu, w)
      return
    end if

    diff = maxval(abs(f_cpu - f_gpu))
    denom = max(1.0d0, maxval(abs(f_cpu)))
    print '(a,1pe18.10)', '[GENERAL] max abs diff = ', diff
    call print_vector('[GENERAL] CPU:', f_cpu)
    call print_vector('  GPU:', f_gpu)
    if (diff > 1.0d-8 .and. diff/denom > 1.0d-8) then
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
    double precision, allocatable :: ptot(:), p(:), w(:)
    double precision, allocatable :: f_cpu(:), f_gpu(:)
    logical(c_bool) :: ok
    double precision :: diff, denom
    integer :: i

    test_heavy_light = .false.
    call set_dimensions(5)

    allocate(ptot(mpack), p(mpack), f_cpu(mpack), f_gpu(mpack))
    allocate(w(10))

    do i = 1, mpack
      ptot(i) = 0.01d0 * dble(i)
      p(i)    = 0.008d0 * dble(i)
    end do

    do i = 1, 10
      w(i) = 0.10d0 + 0.01d0 * dble(i - 1)
    end do

    nfirst = [1_c_int, 5_c_int]
    nlast  = [4_c_int, 5_c_int]

    call cpu_heavy_light_reference(1, 4, 5, ptot, p, w, f_cpu)

    ok = mopac_cuda_fock2_scf(5, mpack, numat, nfirst, nlast, ptot, p, w, w, w, 0_c_int, f_gpu)
    if (.not. ok) then
      print *, '[HEAVY-LIGHT] GPU path unavailable; skipping'
      test_heavy_light = .true.
      deallocate(ptot, p, w, f_cpu, f_gpu)
      return
    end if

    diff = maxval(abs(f_cpu - f_gpu))
    denom = max(1.0d0, maxval(abs(f_cpu)))
    print '(a,1pe18.10)', '[HEAVY-LIGHT] max abs diff = ', diff
    call print_vector('[HEAVY-LIGHT] CPU:', f_cpu)
    call print_vector('  GPU:', f_gpu)
    if (diff > 1.0d-8 .and. diff/denom > 1.0d-8) then
      deallocate(ptot, p, w, f_cpu, f_gpu)
      return
    end if

    test_heavy_light = .true.
    deallocate(ptot, p, w, f_cpu, f_gpu)
  end function test_heavy_light

  logical function test_heavy_heavy()
    implicit none
    integer, parameter :: numat = 2
    integer(c_int) :: nfirst(numat), nlast(numat)
    double precision, allocatable :: ptot(:), p(:), w(:)
    double precision, allocatable :: f_cpu(:), f_gpu(:)
    logical(c_bool) :: ok
    double precision :: diff, denom
    integer :: len_w, i

    test_heavy_heavy = .false.
    call set_dimensions(8)

    allocate(ptot(mpack), p(mpack), f_cpu(mpack), f_gpu(mpack))
    len_w = pair_count(span_count(1, 4)) * pair_count(span_count(5, 8))
    allocate(w(len_w))

    do i = 1, mpack
      ptot(i) = 0.02d0 * dble(i)
      p(i)    = 0.015d0 * dble(i)
    end do

    do i = 1, len_w
      w(i) = 1.0d-3 * dble(i)
    end do

    nfirst = [1_c_int, 5_c_int]
    nlast  = [4_c_int, 8_c_int]

    call cpu_fock_general(1, 4, 5, 8, ptot, p, w, f_cpu)

    ok = mopac_cuda_fock2_scf(8, mpack, numat, nfirst, nlast, ptot, p, w, w, w, 0_c_int, f_gpu)
    if (.not. ok) then
      print *, '[HEAVY-HEAVY] GPU path unavailable; skipping'
      test_heavy_heavy = .true.
      deallocate(ptot, p, w, f_cpu, f_gpu)
      return
    end if

    diff = maxval(abs(f_cpu - f_gpu))
    denom = max(1.0d0, maxval(abs(f_cpu)))
    print '(a,1pe18.10)', '[HEAVY-HEAVY] max abs diff = ', diff
    call print_vector('[HEAVY-HEAVY] CPU:', f_cpu)
    call print_vector('  GPU:', f_gpu)
    if (diff > 1.0d-8 .and. diff/denom > 1.0d-8) then
      deallocate(ptot, p, w, f_cpu, f_gpu)
      return
    end if

    test_heavy_heavy = .true.
    deallocate(ptot, p, w, f_cpu, f_gpu)
  end function test_heavy_heavy

  logical function test_periodic_pair()
    implicit none
    integer, parameter :: numat = 2
    integer(c_int) :: nfirst(numat), nlast(numat)
    double precision, allocatable :: ptot(:), p(:), w(:), wj(:), wk(:)
    double precision, allocatable :: f_cpu(:), f_gpu(:)
    logical(c_bool) :: ok
    double precision :: diff, denom
    integer :: len_block, i

    test_periodic_pair = .false.
    call set_dimensions(4)

    allocate(ptot(mpack), p(mpack), f_cpu(mpack), f_gpu(mpack))
    len_block = pair_count(span_count(1, 2)) * pair_count(span_count(3, 4))
    allocate(w(len_block), wj(len_block), wk(len_block))

    do i = 1, mpack
      ptot(i) = 0.03d0 * dble(i)
      p(i)    = 0.02d0 * dble(i)
    end do

    do i = 1, len_block
      w(i)  = 0.5d0 * dble(i)
      wj(i) = 1.0d-3 * dble(i)
      wk(i) = 2.0d-3 * dble(i)
    end do

    nfirst = [3_c_int, 1_c_int]
    nlast  = [4_c_int, 2_c_int]

    call cpu_fock_periodic(3, 4, 1, 2, ptot, p, wj, wk, f_cpu)

    ok = mopac_cuda_fock2_scf(4, mpack, numat, nfirst, nlast, ptot, p, w, wj, wk, 1_c_int, f_gpu)
    if (.not. ok) then
      print *, '[PERIODIC] GPU path unavailable; skipping'
      test_periodic_pair = .true.
      deallocate(ptot, p, w, wj, wk, f_cpu, f_gpu)
      return
    end if

    diff = maxval(abs(f_cpu - f_gpu))
    denom = max(1.0d0, maxval(abs(f_cpu)))
    print '(a,1pe18.10)', '[PERIODIC] max abs diff = ', diff
    call print_vector('[PERIODIC] CPU:', f_cpu)
    call print_vector('  GPU:', f_gpu)
    if (diff > 1.0d-8 .and. diff/denom > 1.0d-8) then
      deallocate(ptot, p, w, wj, wk, f_cpu, f_gpu)
      return
    end if

    test_periodic_pair = .true.
    deallocate(ptot, p, w, wj, wk, f_cpu, f_gpu)
  end function test_periodic_pair

  subroutine print_vector(label, vec)
    character(*), intent(in) :: label
    double precision, intent(in) :: vec(:)
    integer :: i
    print '(a)', label
    do i = 1, size(vec)
      write(*,'(1pe14.6,1x)', advance='no') vec(i)
      if (mod(i,6) == 0 .or. i == size(vec)) write(*,*)
    end do
  end subroutine print_vector

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

    f = 0.0d0
    kr = 0
    do i = ia, ib
      do j = ia, i
        aa = 2.0d0
        if (i == j) aa = 1.0d0
        do k = ja, jb
          do l = ja, k
            bb = 2.0d0
            if (k == l) bb = 1.0d0
            kr = kr + 1
            if (kr > size(w)) stop '[cpu_fock_general] not enough integrals'
            a = w(kr)
            f(packed_index(i, j)) = f(packed_index(i, j)) + bb * a * ptot(packed_index(k, l))
            f(packed_index(k, l)) = f(packed_index(k, l)) + aa * a * ptot(packed_index(i, j))
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
          val = 0.0d0
          if (offset + 1 <= size(w)) val = w(offset + 1)
          offset = offset + 1
          f(packed_index(orb_i, orb_j)) = f(packed_index(orb_i, orb_j)) + ptot_ll * val
          sumoff = sumoff + ptot(packed_index(orb_i, orb_j)) * val
        end do
      end if
      val = 0.0d0
      if (offset + 1 <= size(w)) val = w(offset + 1)
      offset = offset + 1
      f(packed_index(orb_i, orb_i)) = f(packed_index(orb_i, orb_i)) + ptot_ll * val
      sumdia = sumdia + ptot(packed_index(orb_i, orb_i)) * val
    end do
    f(packed_index(light_atom, light_atom)) = f(packed_index(light_atom, light_atom)) + 2.0d0 * sumoff + sumdia

    do rel = 0, span - 1
      orb_i = heavy_start + rel
      acc = 0.0d0
      do relj = 0, span - 1
        map = jindex_lookup(rel + 1, span, relj + 1)
        if (map <= 0 .or. map > size(w)) cycle
        acc = acc + p(packed_index(heavy_start + relj, light_atom)) * w(map)
      end do
      f(packed_index(orb_i, light_atom)) = f(packed_index(orb_i, light_atom)) - acc
    end do
  end subroutine cpu_heavy_light_reference

  subroutine cpu_fock_periodic(ia, ib, ja, jb, ptot, p, wj_block, wk_block, f)
    implicit none
    integer, intent(in) :: ia, ib, ja, jb
    double precision, intent(in) :: ptot(:), p(:)
    double precision, intent(in) :: wj_block(:), wk_block(:)
    double precision, intent(inout) :: f(:)
    integer :: i, j, k, l
    integer :: kl, ij
    integer :: idx
    double precision :: aa, bb, aj, ak, exch

    f = 0.0d0
    idx = 0
    do i = ia, ib
      do j = ia, i
        aa = 2.0d0
        if (i == j) aa = 1.0d0
        ij = packed_index(i, j)
        do k = ja, jb
          do l = ja, k
            bb = 2.0d0
            if (k == l) bb = 1.0d0
            idx = idx + 1
            aj = wj_block(idx)
            ak = wk_block(idx)
            kl = packed_index(k, l)
            if (kl > ij) cycle
            if (i == k .and. (aa + bb) < 2.1d0) then
              f(ij) = f(ij) + aj * ptot(kl)
            else
              f(ij) = f(ij) + bb * aj * ptot(kl)
              f(kl) = f(kl) + aa * aj * ptot(ij)
              exch = ak * aa * bb * 0.25d0
              if (i >= k .and. j >= l) then
                f(packed_index(i, k)) = f(packed_index(i, k)) - exch * p(packed_index(j, l))
              end if
              if (i >= l .and. j >= k) then
                f(packed_index(i, l)) = f(packed_index(i, l)) - exch * p(packed_index(j, k))
                f(packed_index(j, k)) = f(packed_index(j, k)) - exch * p(packed_index(i, l))
              end if
              if (j >= l .and. i >= k) then
                f(packed_index(j, l)) = f(packed_index(j, l)) - exch * p(packed_index(i, k))
              end if
            end if
          end do
        end do
      end do
    end do
  end subroutine cpu_fock_periodic

  integer function jindex_lookup(row, span, col)
    implicit none
    integer, intent(in) :: row, span, col
    integer, save :: table(256)
    logical, save :: initialized = .false.
    integer :: idx

    if (.not. initialized) then
      call build_jindex_table(table)
      initialized = .true.
    end if

    idx = (row - 1) * span + col
    if (idx < 1 .or. idx > 256) then
      jindex_lookup = 0
    else
      jindex_lookup = table(idx)
    end if
  end function jindex_lookup

  subroutine build_jindex_table(table)
    implicit none
    integer, intent(out) :: table(256)
    integer :: i, j, k, l, m, ij, ji, lk
    m = 0
    do i = 1, 4
      do j = 1, 4
        ij = min(i, j)
        ji = i + j - ij
        do k = 1, 4
          do l = 1, 4
            m = m + 1
            lk = k + l - min(k, l)
            table(m) = (ifact_from(ji) + ij) * 10 + ifact_from(lk) + min(k, l) - 10
          end do
        end do
      end do
    end do
  end subroutine build_jindex_table

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

  integer function ifact_from(n)
    implicit none
    integer, intent(in) :: n
    ifact_from = n * (n - 1) / 2
  end function ifact_from
#else
  ! Placeholder to avoid unused-module warnings when GPU is disabled
  logical function test_light_light()
    test_light_light = .true.
  end function test_light_light
  logical function test_general_pair()
    test_general_pair = .true.
  end function test_general_pair
  logical function test_heavy_light()
    test_heavy_light = .true.
  end function test_heavy_light
#endif
end program gpu_numeric_reference
