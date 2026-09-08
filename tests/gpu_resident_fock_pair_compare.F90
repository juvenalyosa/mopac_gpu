program gpu_resident_fock_pair_compare
#ifdef GPU
  use iso_c_binding, only : c_int
  use gpu_fock_interfaces, only : mopac_cuda_mozyme_sparse_fock_setup, mopac_cuda_mozyme_sparse_fock_run
#endif
  implicit none
  integer :: failures

  failures = 0
#ifdef GPU
  if (.not. run_pair4x1_case(.true., '4x1 resident')) failures = failures + 1
  if (.not. run_pair4x1_case(.false., '1x4 resident')) failures = failures + 1
  if (.not. run_case(1, 1, '1x1')) failures = failures + 1
  if (.not. run_case(1, 4, '1x4 generic')) failures = failures + 1
  if (.not. run_case(4, 1, '4x1 generic')) failures = failures + 1
  if (.not. run_case(4, 4, '4x4')) failures = failures + 1
  if (.not. run_case(1, 9, '1x9')) failures = failures + 1
  if (.not. run_case(9, 1, '9x1')) failures = failures + 1
  if (.not. run_case(4, 9, '4x9')) failures = failures + 1
  if (.not. run_case(9, 4, '9x4')) failures = failures + 1
  if (.not. run_case(9, 9, '9x9')) failures = failures + 1
  if (.not. run_diag_case(4, '4x4 diagonal')) failures = failures + 1
  if (.not. run_diag_case(9, '9x9 diagonal')) failures = failures + 1
  if (.not. run_point_case(1, 4, -2, 'point 1x4 dipole')) failures = failures + 1
  if (.not. run_point_case(4, 1, -2, 'point 4x1 dipole')) failures = failures + 1
  if (.not. run_point_case(4, 4, -2, 'point 4x4 dipole')) failures = failures + 1
  if (.not. run_point_case(1, 9, -2, 'point 1x9 dipole')) failures = failures + 1
  if (.not. run_point_case(9, 1, -2, 'point 9x1 dipole')) failures = failures + 1
  if (.not. run_point_case(9, 4, -2, 'point 9x4 dipole')) failures = failures + 1
  if (.not. run_point_case(9, 9, -2, 'point 9x9 dipole')) failures = failures + 1
  if (.not. run_point_case(9, 9, -1, 'point 9x9 monopole')) failures = failures + 1
#else
  print *, 'GPU support not enabled; gpu_resident_fock_pair_compare requires a GPU build'
  failures = failures + 1
#endif
  if (failures /= 0) stop 1

#ifdef GPU
contains
  logical function run_case(iab, jba, label)
    implicit none
    integer, intent(in) :: iab, jba
    character(*), intent(in) :: label
    integer :: ni, nj, cross_count, total, mpack, i
    integer(c_int) :: code
    integer(c_int) :: dummy_i(1), one_iabs(1), one_ilims(1)
    integer(c_int) :: pair_iabs(1), pair_jbas(1), pair_i_offsets(1), pair_j_offsets(1)
    integer(c_int) :: pair_cross_offsets(1), pair_diag_flags(1), pair_w_offsets(1)
    integer(c_int) :: point_iabs(1), point_jbas(1), point_i_atoms(1), point_j_atoms(1)
    integer(c_int) :: point_i_offsets(1), point_j_offsets(1), point_addr_flags(1)
    double precision :: dummy_d(1), point_w_values(7), qe(2)
    double precision, allocatable :: ptot(:), f_start(:), f_cpu(:), f_gpu(:), wj(:), wk(:)

    interface
      subroutine focd2z(iab_ref, jba_ref, fii, fjj, fij, pii, pjj, pij, &
          wj_ref, wk_ref, diagonal, kr_ref)
        integer, intent(in) :: iab_ref, jba_ref
        integer, intent(inout) :: kr_ref
        logical, intent(in) :: diagonal
        double precision, intent(inout) :: fii(*), fjj(*), fij(*)
        double precision, intent(in) :: pii(*), pjj(*), pij(*), wj_ref(*), wk_ref(*)
      end subroutine focd2z
    end interface

    run_case = .false.
    ni = tri(iab)
    nj = tri(jba)
    cross_count = iab * jba
    total = ni * nj
    mpack = ni + nj + cross_count

    allocate(ptot(mpack), f_start(mpack), f_cpu(mpack), f_gpu(mpack), wj(total), wk(total))
    do i = 1, mpack
      ptot(i) = 0.001d0 * dble(mod(37 * i, 29) + 1)
      f_start(i) = 0.0001d0 * dble(mod(19 * i, 17) + 1)
    end do
    f_cpu = f_start
    f_gpu = f_start
    do i = 1, total
      wj(i) = 0.00001d0 * dble(mod(23 * i, 31) + 1)
      wk(i) = 0.00002d0 * dble(mod(29 * i, 37) + 1)
    end do

    i = 0
    call focd2z(iab, jba, f_cpu(1), f_cpu(ni + 1), f_cpu(ni + nj + 1), &
      ptot(1), ptot(ni + 1), ptot(ni + nj + 1), wj, wk, .false., i)
    if (i /= total) then
      print *, '[RESIDENT FOCK ', trim(label), '] focd2z consumed ', i, &
        ' weights but expected ', total
      deallocate(ptot, f_start, f_cpu, f_gpu, wj, wk)
      return
    end if

    dummy_i = 1_c_int
    one_iabs = 1_c_int
    one_ilims = 1_c_int
    dummy_d = 0.0d0
    point_w_values = 0.0d0
    qe = 0.0d0

    pair_iabs(1) = int(iab, c_int)
    pair_jbas(1) = int(jba, c_int)
    pair_i_offsets(1) = 1_c_int
    pair_j_offsets(1) = int(ni + 1, c_int)
    pair_cross_offsets(1) = int(ni + nj + 1, c_int)
    pair_diag_flags(1) = 0_c_int
    pair_w_offsets(1) = 1_c_int
    point_iabs = 1_c_int
    point_jbas = 1_c_int
    point_i_atoms = 1_c_int
    point_j_atoms = 1_c_int
    point_i_offsets = 1_c_int
    point_j_offsets = 1_c_int
    point_addr_flags = 0_c_int

    code = mopac_cuda_mozyme_sparse_fock_setup(int(mpack, c_int), 2_c_int, 0_c_int, &
      dummy_i, dummy_i, one_iabs, one_ilims, 0_c_int, dummy_d, 1_c_int, &
      pair_iabs, pair_jbas, pair_i_offsets, pair_j_offsets, pair_cross_offsets, &
      pair_diag_flags, pair_w_offsets, int(total, c_int), wj, wk, 0_c_int, &
      dummy_i, dummy_i, dummy_i, dummy_d, dummy_d, 0_c_int, point_iabs, point_jbas, &
      point_i_atoms, point_j_atoms, point_i_offsets, point_j_offsets, point_addr_flags, point_w_values)
    if (code /= 0) then
      print *, '[RESIDENT FOCK ', trim(label), '] setup failed code=', code
      deallocate(ptot, f_start, f_cpu, f_gpu, wj, wk)
      return
    end if

    code = mopac_cuda_mozyme_sparse_fock_run(int(mpack, c_int), ptot, qe, f_gpu)
    if (code /= 0) then
      print *, '[RESIDENT FOCK ', trim(label), '] run failed code=', code
      deallocate(ptot, f_start, f_cpu, f_gpu, wj, wk)
      return
    end if

    if (.not. compare_result(label, 'add', f_cpu, f_gpu)) then
      deallocate(ptot, f_start, f_cpu, f_gpu, wj, wk)
      return
    end if

    f_cpu = -f_start
    i = 0
    call focd2z(iab, jba, f_cpu(1), f_cpu(ni + 1), f_cpu(ni + nj + 1), &
      ptot(1), ptot(ni + 1), ptot(ni + nj + 1), wj, wk, .false., i)
    f_cpu = -f_cpu
    if (i /= total) then
      print *, '[RESIDENT FOCK ', trim(label), ' subtract] focd2z consumed ', i, &
        ' weights but expected ', total
      deallocate(ptot, f_start, f_cpu, f_gpu, wj, wk)
      return
    end if

    f_gpu = -f_start
    code = mopac_cuda_mozyme_sparse_fock_run(int(mpack, c_int), ptot, qe, f_gpu)
    if (code /= 0) then
      print *, '[RESIDENT FOCK ', trim(label), ' subtract] run failed code=', code
      deallocate(ptot, f_start, f_cpu, f_gpu, wj, wk)
      return
    end if
    f_gpu = -f_gpu

    if (.not. compare_result(label, 'subtract', f_cpu, f_gpu)) then
      deallocate(ptot, f_start, f_cpu, f_gpu, wj, wk)
      return
    end if

    run_case = .true.
    deallocate(ptot, f_start, f_cpu, f_gpu, wj, wk)
  end function run_case

  logical function run_diag_case(iab, label)
    implicit none
    integer, intent(in) :: iab
    character(*), intent(in) :: label
    integer :: ni, total, mpack, i
    integer(c_int) :: code
    integer(c_int) :: dummy_i(1), one_iabs(1), one_ilims(1)
    integer(c_int) :: pair_iabs(1), pair_jbas(1), pair_i_offsets(1), pair_j_offsets(1)
    integer(c_int) :: pair_cross_offsets(1), pair_diag_flags(1), pair_w_offsets(1)
    integer(c_int) :: point_iabs(1), point_jbas(1), point_i_atoms(1), point_j_atoms(1)
    integer(c_int) :: point_i_offsets(1), point_j_offsets(1), point_addr_flags(1)
    double precision :: dummy_d(1), point_w_values(7), qe(1)
    double precision, allocatable :: ptot(:), f_start(:), f_cpu(:), f_gpu(:), wj(:), wk(:)

    interface
      subroutine focd2z(iab_ref, jba_ref, fii, fjj, fij, pii, pjj, pij, &
          wj_ref, wk_ref, diagonal, kr_ref)
        integer, intent(in) :: iab_ref, jba_ref
        integer, intent(inout) :: kr_ref
        logical, intent(in) :: diagonal
        double precision, intent(inout) :: fii(*), fjj(*), fij(*)
        double precision, intent(in) :: pii(*), pjj(*), pij(*), wj_ref(*), wk_ref(*)
      end subroutine focd2z
    end interface

    run_diag_case = .false.
    ni = tri(iab)
    total = ni * ni
    mpack = ni

    allocate(ptot(mpack), f_start(mpack), f_cpu(mpack), f_gpu(mpack), wj(total), wk(total))
    do i = 1, mpack
      ptot(i) = 0.0017d0 * dble(mod(43 * i, 31) + 1)
      f_start(i) = 0.00013d0 * dble(mod(23 * i, 19) + 1)
    end do
    do i = 1, total
      wj(i) = 0.000011d0 * dble(mod(17 * i, 37) + 1)
      wk(i) = 0.000019d0 * dble(mod(31 * i, 41) + 1)
    end do

    f_cpu = f_start
    f_gpu = f_start
    i = 0
    call focd2z(iab, iab, f_cpu(1), f_cpu(1), f_cpu(1), &
      ptot(1), ptot(1), ptot(1), wj, wk, .true., i)
    if (i /= total) then
      print *, '[RESIDENT FOCK ', trim(label), '] focd2z consumed ', i, &
        ' weights but expected ', total
      deallocate(ptot, f_start, f_cpu, f_gpu, wj, wk)
      return
    end if

    dummy_i = 1_c_int
    one_iabs = 1_c_int
    one_ilims = 1_c_int
    dummy_d = 0.0d0
    point_w_values = 0.0d0
    qe = 0.0d0
    pair_iabs(1) = int(iab, c_int)
    pair_jbas(1) = int(iab, c_int)
    pair_i_offsets(1) = 1_c_int
    pair_j_offsets(1) = 1_c_int
    pair_cross_offsets(1) = 1_c_int
    pair_diag_flags(1) = 1_c_int
    pair_w_offsets(1) = 1_c_int
    point_iabs = 1_c_int
    point_jbas = 1_c_int
    point_i_atoms = 1_c_int
    point_j_atoms = 1_c_int
    point_i_offsets = 1_c_int
    point_j_offsets = 1_c_int
    point_addr_flags = 0_c_int

    code = mopac_cuda_mozyme_sparse_fock_setup(int(mpack, c_int), 1_c_int, 0_c_int, &
      dummy_i, dummy_i, one_iabs, one_ilims, 0_c_int, dummy_d, 1_c_int, &
      pair_iabs, pair_jbas, pair_i_offsets, pair_j_offsets, pair_cross_offsets, &
      pair_diag_flags, pair_w_offsets, int(total, c_int), wj, wk, 0_c_int, &
      dummy_i, dummy_i, dummy_i, dummy_d, dummy_d, 0_c_int, point_iabs, point_jbas, &
      point_i_atoms, point_j_atoms, point_i_offsets, point_j_offsets, point_addr_flags, point_w_values)
    if (code /= 0) then
      print *, '[RESIDENT FOCK ', trim(label), '] setup failed code=', code
      deallocate(ptot, f_start, f_cpu, f_gpu, wj, wk)
      return
    end if

    code = mopac_cuda_mozyme_sparse_fock_run(int(mpack, c_int), ptot, qe, f_gpu)
    if (code /= 0) then
      print *, '[RESIDENT FOCK ', trim(label), '] run failed code=', code
      deallocate(ptot, f_start, f_cpu, f_gpu, wj, wk)
      return
    end if

    if (.not. compare_result(label, 'add', f_cpu, f_gpu)) then
      deallocate(ptot, f_start, f_cpu, f_gpu, wj, wk)
      return
    end if

    run_diag_case = .true.
    deallocate(ptot, f_start, f_cpu, f_gpu, wj, wk)
  end function run_diag_case

  logical function run_pair4x1_case(heavy_first, label)
    implicit none
    logical, intent(in) :: heavy_first
    character(*), intent(in) :: label
    integer :: heavy_offset, light_offset, cross_offset, i, mpack
    integer(c_int) :: code
    integer(c_int) :: dummy_i(1), one_iabs(1), one_ilims(1)
    integer(c_int) :: pair_iabs(1), pair_jbas(1), pair_i_offsets(1), pair_j_offsets(1)
    integer(c_int) :: pair_cross_offsets(1), pair_diag_flags(1), pair_w_offsets(1)
    integer(c_int) :: pair4_heavy_offsets(1), pair4_light_offsets(1), pair4_cross_offsets(1)
    integer(c_int) :: point_iabs(1), point_jbas(1), point_i_atoms(1), point_j_atoms(1)
    integer(c_int) :: point_i_offsets(1), point_j_offsets(1), point_addr_flags(1)
    double precision :: dummy_d(1), pair4_wj_values(10), pair4_wk_values(16)
    double precision :: point_w_values(7), qe(2)
    double precision, allocatable :: ptot(:), f_start(:), f_cpu(:), f_gpu(:)

    run_pair4x1_case = .false.
    mpack = 15
    if (heavy_first) then
      heavy_offset = 1
      light_offset = 11
    else
      light_offset = 1
      heavy_offset = 2
    end if
    cross_offset = 12

    allocate(ptot(mpack), f_start(mpack), f_cpu(mpack), f_gpu(mpack))
    do i = 1, mpack
      ptot(i) = 0.0015d0 * dble(mod(31 * i, 23) + 1)
      f_start(i) = 0.0002d0 * dble(mod(17 * i, 13) + 1)
    end do
    do i = 1, 10
      pair4_wj_values(i) = 0.00003d0 * dble(mod(11 * i, 19) + 1)
    end do
    do i = 1, 16
      pair4_wk_values(i) = 0.00004d0 * dble(mod(13 * i, 29) + 1)
    end do

    f_cpu = f_start
    f_gpu = f_start
    call apply_pair4x1_cpu(f_cpu, ptot, heavy_offset, light_offset, &
      cross_offset, pair4_wj_values, pair4_wk_values)

    dummy_i = 1_c_int
    one_iabs = 1_c_int
    one_ilims = 1_c_int
    dummy_d = 0.0d0
    point_w_values = 0.0d0
    qe = 0.0d0
    pair_iabs = 1_c_int
    pair_jbas = 1_c_int
    pair_i_offsets = 1_c_int
    pair_j_offsets = 1_c_int
    pair_cross_offsets = 1_c_int
    pair_diag_flags = 0_c_int
    pair_w_offsets = 1_c_int
    pair4_heavy_offsets(1) = int(heavy_offset, c_int)
    pair4_light_offsets(1) = int(light_offset, c_int)
    pair4_cross_offsets(1) = int(cross_offset, c_int)
    point_iabs = 1_c_int
    point_jbas = 1_c_int
    point_i_atoms = 1_c_int
    point_j_atoms = 1_c_int
    point_i_offsets = 1_c_int
    point_j_offsets = 1_c_int
    point_addr_flags = 0_c_int

    code = mopac_cuda_mozyme_sparse_fock_setup(int(mpack, c_int), 2_c_int, 0_c_int, &
      dummy_i, dummy_i, one_iabs, one_ilims, 0_c_int, dummy_d, 0_c_int, &
      pair_iabs, pair_jbas, pair_i_offsets, pair_j_offsets, pair_cross_offsets, &
      pair_diag_flags, pair_w_offsets, 0_c_int, dummy_d, dummy_d, 1_c_int, &
      pair4_heavy_offsets, pair4_light_offsets, pair4_cross_offsets, &
      pair4_wj_values, pair4_wk_values, 0_c_int, point_iabs, point_jbas, &
      point_i_atoms, point_j_atoms, point_i_offsets, point_j_offsets, &
      point_addr_flags, point_w_values)
    if (code /= 0) then
      print *, '[RESIDENT FOCK ', trim(label), '] setup failed code=', code
      deallocate(ptot, f_start, f_cpu, f_gpu)
      return
    end if

    code = mopac_cuda_mozyme_sparse_fock_run(int(mpack, c_int), ptot, qe, f_gpu)
    if (code /= 0) then
      print *, '[RESIDENT FOCK ', trim(label), '] run failed code=', code
      deallocate(ptot, f_start, f_cpu, f_gpu)
      return
    end if

    if (.not. compare_result(label, 'add', f_cpu, f_gpu)) then
      deallocate(ptot, f_start, f_cpu, f_gpu)
      return
    end if

    f_cpu = -f_start
    f_gpu = -f_start
    call apply_pair4x1_cpu(f_cpu, ptot, heavy_offset, light_offset, &
      cross_offset, pair4_wj_values, pair4_wk_values)
    f_cpu = -f_cpu
    code = mopac_cuda_mozyme_sparse_fock_run(int(mpack, c_int), ptot, qe, f_gpu)
    if (code /= 0) then
      print *, '[RESIDENT FOCK ', trim(label), ' subtract] run failed code=', code
      deallocate(ptot, f_start, f_cpu, f_gpu)
      return
    end if
    f_gpu = -f_gpu

    if (.not. compare_result(label, 'subtract', f_cpu, f_gpu)) then
      deallocate(ptot, f_start, f_cpu, f_gpu)
      return
    end if

    run_pair4x1_case = .true.
    deallocate(ptot, f_start, f_cpu, f_gpu)
  end function run_pair4x1_case

  ! resident sparse point-charge/dipole 9-orbital comparison
  logical function run_point_case(iab, jba, addr_flag, label)
    implicit none
    integer, intent(in) :: iab, jba, addr_flag
    character(*), intent(in) :: label
    integer :: ni, nj, mpack, i
    integer(c_int) :: code
    integer(c_int) :: dummy_i(1), one_iabs(1), one_ilims(1)
    integer(c_int) :: pair_iabs(1), pair_jbas(1), pair_i_offsets(1), pair_j_offsets(1)
    integer(c_int) :: pair_cross_offsets(1), pair_diag_flags(1), pair_w_offsets(1)
    integer(c_int) :: point_iabs(1), point_jbas(1), point_i_atoms(1), point_j_atoms(1)
    integer(c_int) :: point_i_offsets(1), point_j_offsets(1), point_addr_flags(1)
    double precision :: dummy_d(1), point_w_values(7), qe(2)
    double precision, allocatable :: ptot(:), f_start(:), f_cpu(:), f_gpu(:)

    run_point_case = .false.
    ni = tri(iab)
    nj = tri(jba)
    mpack = ni + nj

    allocate(ptot(mpack), f_start(mpack), f_cpu(mpack), f_gpu(mpack))
    do i = 1, mpack
      ptot(i) = 0.002d0 * dble(mod(41 * i, 31) + 1)
      f_start(i) = 0.0003d0 * dble(mod(13 * i, 19) + 1)
    end do
    point_w_values = (/ 0.031d0, -0.007d0, 0.011d0, -0.013d0, 0.017d0, -0.019d0, 0.023d0 /)
    qe = (/ -0.37d0, 0.61d0 /)

    f_cpu = f_start
    f_gpu = f_start
    call apply_point_cpu(iab, jba, addr_flag, 1, ni + 1, 1, 2, ptot, qe, point_w_values, f_cpu)

    dummy_i = 1_c_int
    one_iabs = 1_c_int
    one_ilims = 1_c_int
    dummy_d = 0.0d0
    pair_iabs = 1_c_int
    pair_jbas = 1_c_int
    pair_i_offsets = 1_c_int
    pair_j_offsets = 1_c_int
    pair_cross_offsets = 1_c_int
    pair_diag_flags = 0_c_int
    pair_w_offsets = 1_c_int
    point_iabs(1) = int(iab, c_int)
    point_jbas(1) = int(jba, c_int)
    point_i_atoms(1) = 1_c_int
    point_j_atoms(1) = 2_c_int
    point_i_offsets(1) = 1_c_int
    point_j_offsets(1) = int(ni + 1, c_int)
    point_addr_flags(1) = int(addr_flag, c_int)

    code = mopac_cuda_mozyme_sparse_fock_setup(int(mpack, c_int), 2_c_int, 0_c_int, &
      dummy_i, dummy_i, one_iabs, one_ilims, 0_c_int, dummy_d, 0_c_int, &
      pair_iabs, pair_jbas, pair_i_offsets, pair_j_offsets, pair_cross_offsets, &
      pair_diag_flags, pair_w_offsets, 0_c_int, dummy_d, dummy_d, 0_c_int, &
      dummy_i, dummy_i, dummy_i, dummy_d, dummy_d, 1_c_int, point_iabs, point_jbas, &
      point_i_atoms, point_j_atoms, point_i_offsets, point_j_offsets, point_addr_flags, point_w_values)
    if (code /= 0) then
      print *, '[RESIDENT FOCK ', trim(label), '] setup failed code=', code
      deallocate(ptot, f_start, f_cpu, f_gpu)
      return
    end if

    code = mopac_cuda_mozyme_sparse_fock_run(int(mpack, c_int), ptot, qe, f_gpu)
    if (code /= 0) then
      print *, '[RESIDENT FOCK ', trim(label), '] run failed code=', code
      deallocate(ptot, f_start, f_cpu, f_gpu)
      return
    end if

    if (.not. compare_result(label, 'add', f_cpu, f_gpu)) then
      deallocate(ptot, f_start, f_cpu, f_gpu)
      return
    end if

    f_cpu = -f_start
    f_gpu = -f_start
    call apply_point_cpu(iab, jba, addr_flag, 1, ni + 1, 1, 2, ptot, qe, point_w_values, f_cpu)
    f_cpu = -f_cpu
    code = mopac_cuda_mozyme_sparse_fock_run(int(mpack, c_int), ptot, qe, f_gpu)
    if (code /= 0) then
      print *, '[RESIDENT FOCK ', trim(label), ' subtract] run failed code=', code
      deallocate(ptot, f_start, f_cpu, f_gpu)
      return
    end if
    f_gpu = -f_gpu

    if (.not. compare_result(label, 'subtract', f_cpu, f_gpu)) then
      deallocate(ptot, f_start, f_cpu, f_gpu)
      return
    end if

    run_point_case = .true.
    deallocate(ptot, f_start, f_cpu, f_gpu)
  end function run_point_case

  subroutine apply_pair4x1_cpu(f, ptot, heavy_offset, light_offset, cross_offset, &
      wj_values, wk_values)
    implicit none
    double precision, intent(inout) :: f(:)
    double precision, intent(in) :: ptot(:), wj_values(10), wk_values(16)
    integer, intent(in) :: heavy_offset, light_offset, cross_offset
    integer :: i, j, k, li
    double precision :: sum, sumdia, sumoff

    sumdia = 0.0d0
    sumoff = 0.0d0
    li = heavy_offset - 1
    k = 0
    do i = 1, 4
      do j = 1, i - 1
        li = li + 1
        k = k + 1
        f(li) = f(li) + ptot(light_offset) * wj_values(k)
        sumoff = sumoff + ptot(li) * wj_values(k)
      end do
      li = li + 1
      k = k + 1
      f(li) = f(li) + ptot(light_offset) * wj_values(k)
      sumdia = sumdia + ptot(li) * wj_values(k)
    end do
    f(light_offset) = f(light_offset) + sumoff * 2.0d0 + sumdia

    k = 0
    do i = 1, 4
      sum = 0.0d0
      do j = 1, 4
        k = k + 1
        sum = sum + ptot(cross_offset + j - 1) * wk_values(k)
      end do
      f(cross_offset + i - 1) = f(cross_offset + i - 1) - sum * 0.5d0
    end do
  end subroutine apply_pair4x1_cpu

  subroutine apply_point_cpu(iab, jba, addr_flag, i_offset, j_offset, i_atom, j_atom, ptot, qe, w, f)
    implicit none
    integer, intent(in) :: iab, jba, addr_flag, i_offset, j_offset, i_atom, j_atom
    double precision, intent(in) :: ptot(:), qe(:), w(7)
    double precision, intent(inout) :: f(:)
    integer :: orb, i_base, j_base, k, ndiag
    integer :: diag_offsets(4)
    double precision :: w1, w2, w3, w4, w5, w6, w7, sum

    i_base = i_offset - 1
    j_base = j_offset - 1
    w1 = w(1)
    do orb = 1, iab
      f(i_base + pack_lower(orb, orb)) = f(i_base + pack_lower(orb, orb)) + qe(j_atom) * w1
    end do
    do orb = 1, jba
      f(j_base + pack_lower(orb, orb)) = f(j_base + pack_lower(orb, orb)) + qe(i_atom) * w1
    end do

    if (addr_flag /= -2) return

    w2 = w(2)
    w3 = w(3)
    w4 = w(4)
    w5 = w(5)
    w6 = w(6)
    w7 = w(7)
    diag_offsets = (/ 1, 3, 6, 10 /)

    if (iab > 1) then
      f(i_base + 2) = f(i_base + 2) + qe(j_atom) * w5
      f(i_base + 4) = f(i_base + 4) + qe(j_atom) * w6
      f(i_base + 7) = f(i_base + 7) + qe(j_atom) * w7
    end if
    if (jba > 1) then
      f(j_base + 2) = f(j_base + 2) + qe(i_atom) * w2
      f(j_base + 4) = f(j_base + 4) + qe(i_atom) * w3
      f(j_base + 7) = f(j_base + 7) + qe(i_atom) * w4
    end if
    if (iab > 1) then
      ndiag = 1
      if (jba > 1) ndiag = 4
      sum = 2.0d0 * (ptot(i_base + 2) * w5 + ptot(i_base + 4) * w6 + ptot(i_base + 7) * w7)
      do k = 1, ndiag
        f(j_base + diag_offsets(k)) = f(j_base + diag_offsets(k)) + sum
      end do
    end if
    if (jba > 1) then
      ndiag = 1
      if (iab > 1) ndiag = 4
      sum = 2.0d0 * (ptot(j_base + 2) * w2 + ptot(j_base + 4) * w3 + ptot(j_base + 7) * w4)
      do k = 1, ndiag
        f(i_base + diag_offsets(k)) = f(i_base + diag_offsets(k)) + sum
      end do
    end if
  end subroutine apply_point_cpu

  logical function compare_result(label, mode_label, f_cpu, f_gpu)
    implicit none
    character(*), intent(in) :: label, mode_label
    double precision, intent(in) :: f_cpu(:), f_gpu(:)
    double precision :: diff, denom

    diff = maxval(abs(f_cpu - f_gpu))
    denom = max(1.0d0, maxval(abs(f_cpu)))
    print '(a,a,1x,a,a,1pe18.10)', '[RESIDENT FOCK ', trim(label), trim(mode_label), '] max abs diff = ', diff
    if (diff > 1.0d-8 .and. diff / denom > 1.0d-8) then
      print *, '[RESIDENT FOCK ', trim(label), ' ', trim(mode_label), '] CPU/GPU mismatch denom=', denom
      compare_result = .false.
      return
    end if

    compare_result = .true.
  end function compare_result

  integer function tri(n)
    implicit none
    integer, intent(in) :: n
    tri = (n * (n + 1)) / 2
  end function tri

  integer function pack_lower(i, j)
    implicit none
    integer, intent(in) :: i, j
    integer :: hi, lo
    hi = max(i, j)
    lo = min(i, j)
    pack_lower = (hi * (hi - 1)) / 2 + lo
  end function pack_lower
#endif
end program gpu_resident_fock_pair_compare
