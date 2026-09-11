! Molecular Orbital PACkage (MOPAC)
! Copyright 2021 Virginia Polytechnic Institute and State University
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!    http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

! GPU evaluation of the MOZYME per-atom-pair work: the Cartesian gradient
! (pair loop of dcart_build_scf_gradient_cpu) and the block pairs of
! hcore_for_MOZYME.  The device code lives in src/gpu/mozyme_pair_gradient.cu;
! this module builds the pair list from the MOZYME block index (nijbo or the
! compact iij/ijall/iijj lists), hands the parameter tables over and adds the
! results on to the Fortran arrays.
module mozyme_gpu_gradient
  use iso_c_binding, only : c_int, c_double, c_ptr
  use mozyme_section_timers, only : mozyme_section_timer_begin, mozyme_section_timer_end
  implicit none
  private
  public :: mozyme_gpu_gradient_enabled
  public :: mozyme_gpu_gradient_check_enabled
  public :: mozyme_gpu_gradient_run
  public :: mozyme_gpu_hcore_enabled
  public :: mozyme_gpu_hcore_check_enabled
  public :: mozyme_gpu_hcore_run
  public :: mozyme_gpu_sp_pair
  public :: mozyme_gpu_device_pair
  public :: mozyme_gpu_d_pairs_enabled
  public :: mozyme_gpu_disp_enabled
  public :: mozyme_gpu_disp_check_enabled

  ! Mirror of MozymePairTablesC in mozyme_pair_gradient.cu.
  type, bind(C) :: mozyme_pair_tables_c
    type(c_ptr) :: natorb, npq, iod
    type(c_ptr) :: zs, zp, zd, betas, betap, betad, tore, alp, am, ad, aq, dd, qq
    type(c_ptr) :: po, ddp, guess1, guess2, guess3, alpb, xfac, v_par
    real(c_double) :: a0, ev, cutofs, cutof1, trunc_1, trunc_2
    integer(c_int) :: method_flags(7)
  end type mozyme_pair_tables_c

#ifdef GPU
  ! Interoperable copies of the parameter tables (the parameters_C arrays do
  ! not all carry the TARGET attribute).
  integer(c_int), save, target :: t_natorb(107), t_npq(107*3), t_iod(107)
  real(c_double), save, target :: t_zs(107), t_zp(107), t_zd(107), t_betas(107), t_betap(107), t_betad(107)
  real(c_double), save, target :: t_tore(107), t_alp(107), t_am(107), t_ad(107), t_aq(107), t_dd(107), t_qq(107)
  real(c_double), save, target :: t_po(9*107), t_ddp(6*107), t_guess1(107*4), t_guess2(107*4), t_guess3(107*4)
  real(c_double), save, target :: t_alpb(100*100), t_xfac(100*100), t_v_par(60)

  interface
    function mopac_cuda_mozyme_pair_gradient(numat_c, mpack_c, npairs_c, pair_i_c, pair_j_c, &
        pair_off_c, row_start_c, diag_off_c, iorbs_c, nat_c, coord_c, p_c, distance_gate_c, &
        d_on_device_c, cutof2_c, cutofp_c, chnge_c, cnst_c, fpc_9_c, force_c, tables_c, dxyz_c, ms_c, &
        d_pairs_c) bind(C, name='mopac_cuda_mozyme_pair_gradient') result(rc)
      import :: c_int, c_double, mozyme_pair_tables_c
      integer(c_int), value :: numat_c, mpack_c, npairs_c, distance_gate_c, d_on_device_c, force_c
      integer(c_int) :: pair_i_c(*), pair_j_c(*), pair_off_c(*), row_start_c(*), diag_off_c(*)
      integer(c_int) :: iorbs_c(*), nat_c(*)
      real(c_double) :: coord_c(*), p_c(*), dxyz_c(*)
      real(c_double), value :: cutof2_c, cutofp_c, chnge_c, cnst_c, fpc_9_c
      type(mozyme_pair_tables_c) :: tables_c
      real(c_double) :: ms_c
      integer(c_int) :: d_pairs_c
      integer(c_int) :: rc
    end function mopac_cuda_mozyme_pair_gradient

    function mopac_cuda_mozyme_hcore_pairs(numat_c, mpack_c, npairs_c, pair_i_c, pair_j_c, &
        pair_off_c, row_start_c, diag_off_c, iorbs_c, nat_c, coord_c, distance_gate_c, &
        d_on_device_c, cutof2_c, tables_c, nijbo_c, have_nijbo_c, point_c, h_c, enuc_c, ms_c, d_pairs_c) &
        bind(C, name='mopac_cuda_mozyme_hcore_pairs') result(rc)
      import :: c_int, c_double, mozyme_pair_tables_c
      integer(c_int), value :: numat_c, mpack_c, npairs_c, distance_gate_c, d_on_device_c
      integer(c_int), value :: have_nijbo_c, point_c
      integer(c_int) :: pair_i_c(*), pair_j_c(*), pair_off_c(*), row_start_c(*), diag_off_c(*)
      integer(c_int) :: iorbs_c(*), nat_c(*), nijbo_c(*)
      real(c_double) :: coord_c(*), h_c(*)
      real(c_double), value :: cutof2_c
      type(mozyme_pair_tables_c) :: tables_c
      real(c_double) :: enuc_c, ms_c
      integer(c_int) :: d_pairs_c
      integer(c_int) :: rc
    end function mopac_cuda_mozyme_hcore_pairs
  end interface
#endif

contains

  logical function env_truthy(name)
    implicit none
    character(len=*), intent(in) :: name
    character(len=16) :: value
    integer :: status, length
    env_truthy = .false.
    value = ' '
    call get_environment_variable(name, value, length=length, status=status)
    if (status /= 0 .or. length <= 0) return
    value = adjustl(value)
    select case (trim(value))
    case ('1', 'T', 't', 'TRUE', 'true', 'True', 'Y', 'y', 'YES', 'yes', 'ON', 'on')
      env_truthy = .true.
    end select
  end function env_truthy

  ! Explicit off switch: 0 / off / no / false.
  logical function env_falsy(name)
    implicit none
    character(len=*), intent(in) :: name
    character(len=16) :: value
    integer :: status, length
    env_falsy = .false.
    value = ' '
    call get_environment_variable(name, value, length=length, status=status)
    if (status /= 0 .or. length <= 0) return
    value = adjustl(value)
    select case (trim(value))
    case ('0', 'F', 'f', 'FALSE', 'false', 'False', 'N', 'n', 'NO', 'no', 'OFF', 'off')
      env_falsy = .true.
    end select
  end function env_falsy

  logical function gpu_allowed()
    implicit none
#ifdef GPU
    gpu_allowed = .not. env_truthy('MOPAC_NOGPU') .and. .not. env_truthy('MOZYME_GPU_OFF')
#else
    gpu_allowed = .false.
#endif
  end function gpu_allowed

  ! MOPAC_MOZYME_GRAD_GPU=1 (set by the MOZYME GPU defaults) enables the
  ! gradient path; MOPAC_NOGPU / MOZYME_GPU_OFF always win.
  logical function mozyme_gpu_gradient_enabled()
    implicit none
    mozyme_gpu_gradient_enabled = gpu_allowed() .and. env_truthy('MOPAC_MOZYME_GRAD_GPU')
  end function mozyme_gpu_gradient_enabled

  ! MOPAC_GPU_GRAD_CHECK=1: evaluate both CPU and GPU gradients and report the
  ! difference (the CPU result is kept).
  logical function mozyme_gpu_gradient_check_enabled()
    implicit none
    mozyme_gpu_gradient_check_enabled = env_truthy('MOPAC_GPU_GRAD_CHECK')
  end function mozyme_gpu_gradient_check_enabled

  ! MOPAC_MOZYME_HCORE_GPU=1 (set by the defaults) enables the hcore block-pair path.
  logical function mozyme_gpu_hcore_enabled()
    implicit none
    mozyme_gpu_hcore_enabled = gpu_allowed() .and. env_truthy('MOPAC_MOZYME_HCORE_GPU')
  end function mozyme_gpu_hcore_enabled

  ! MOPAC_GPU_HCORE_CHECK=1: build h on the CPU as well and report the difference.
  logical function mozyme_gpu_hcore_check_enabled()
    implicit none
    mozyme_gpu_hcore_check_enabled = env_truthy('MOPAC_GPU_HCORE_CHECK')
  end function mozyme_gpu_hcore_check_enabled

  ! MOPAC_DH_DISP_GPU=1 (set by the defaults): PM6-DH/PM7 dispersion energy and
  ! analytic gradient on the device (src/gpu/dh_dispersion.cu).
  logical function mozyme_gpu_disp_enabled()
    implicit none
    mozyme_gpu_disp_enabled = gpu_allowed() .and. env_truthy('MOPAC_DH_DISP_GPU')
  end function mozyme_gpu_disp_enabled

  ! MOPAC_GPU_DISP_CHECK=1: compare with the CPU energy / numerical gradient.
  logical function mozyme_gpu_disp_check_enabled()
    implicit none
    mozyme_gpu_disp_check_enabled = env_truthy('MOPAC_GPU_DISP_CHECK')
  end function mozyme_gpu_disp_check_enabled

  ! True when the sp device kernels handle the pair: both atoms carry 1 or 4 orbitals.
  logical function mozyme_gpu_sp_pair(norb_i, norb_j)
    implicit none
    integer, intent(in) :: norb_i, norb_j
    mozyme_gpu_sp_pair = (norb_i == 1 .or. norb_i == 4) .and. (norb_j == 1 .or. norb_j == 4)
  end function mozyme_gpu_sp_pair

  ! Pairs with a d-orbital atom go to the device unless MOPAC_MOZYME_DPAIRS_GPU=0.
  logical function mozyme_gpu_d_pairs_enabled()
    implicit none
    mozyme_gpu_d_pairs_enabled = gpu_allowed() .and. .not. env_falsy('MOPAC_MOZYME_DPAIRS_GPU')
  end function mozyme_gpu_d_pairs_enabled

  ! True when some device kernel handles the pair (sp-sp, or {1,4,9} x {1,4,9}
  ! with the d path enabled).  Everything else stays on the CPU.
  logical function mozyme_gpu_device_pair(norb_i, norb_j)
    implicit none
    integer, intent(in) :: norb_i, norb_j
    mozyme_gpu_device_pair = mozyme_gpu_sp_pair(norb_i, norb_j)
    if (.not. mozyme_gpu_device_pair .and. mozyme_gpu_d_pairs_enabled()) then
      mozyme_gpu_device_pair = (norb_i == 1 .or. norb_i == 4 .or. norb_i == 9) .and. &
        (norb_j == 1 .or. norb_j == 4 .or. norb_j == 9)
    end if
  end function mozyme_gpu_device_pair

  ! Adds the pair contributions of the MOZYME gradient on to dxyz(3, numat).
  ! code: 0 success, 1 unsupported configuration (nothing done), 2 device
  ! failure (nothing done), 3 device evaluation failed for some pair (nothing done).
  ! d_pairs_out: interacting pairs involving a d-orbital atom, which the device
  ! skips; the caller must add them with dcart_build_scf_gradient_cpu(d_pairs_only).
  subroutine mozyme_gpu_gradient_run(numat_in, coord, dxyz, force, chnge, const, code, ms, npairs_out, &
      d_pairs_out)
    implicit none
    integer, intent(in) :: numat_in
    double precision, intent(in) :: coord(3, numat_in)
    double precision, intent(inout) :: dxyz(3, numat_in)
    logical, intent(in) :: force
    double precision, intent(in) :: chnge, const
    integer, intent(out) :: code
    double precision, intent(out) :: ms
    integer, intent(out) :: npairs_out, d_pairs_out
#ifdef GPU
    call mozyme_gpu_gradient_run_gpu(numat_in, coord, dxyz, force, chnge, const, code, ms, npairs_out, &
      d_pairs_out)
#else
    code = 1
    ms = 0.d0
    npairs_out = 0
    d_pairs_out = 0
    if (numat_in < 0 .or. force .or. chnge < 0.d0 .or. const < 0.d0) code = 1
    if (coord(1, 1) /= coord(1, 1)) dxyz(1, 1) = dxyz(1, 1)
#endif
  end subroutine mozyme_gpu_gradient_run

  ! Adds the block-pair (ijbo >= 0, sp-sp) contributions of hcore_for_MOZYME on
  ! to h(mpack) and returns their core-core repulsion in enuc_add.  Codes as above.
  subroutine mozyme_gpu_hcore_run(numat_in, coord, h, enuc_add, point_pairs, code, ms, npairs_out, d_pairs_out)
    implicit none
    integer, intent(in) :: numat_in
    double precision, intent(in) :: coord(3, numat_in)
    double precision, intent(inout) :: h(*)
    double precision, intent(out) :: enuc_add
    logical, intent(in) :: point_pairs
    integer, intent(out) :: code
    double precision, intent(out) :: ms
    integer, intent(out) :: npairs_out, d_pairs_out
#ifdef GPU
    call mozyme_gpu_hcore_run_gpu(numat_in, coord, h, enuc_add, point_pairs, code, ms, npairs_out, d_pairs_out)
#else
    code = 1
    ms = 0.d0
    enuc_add = 0.d0
    npairs_out = 0
    d_pairs_out = 0
    if (numat_in < 0 .or. point_pairs) code = 1
    if (coord(1, 1) /= coord(1, 1)) h(1) = h(1)
#endif
  end subroutine mozyme_gpu_hcore_run

#ifdef GPU
  ! Pair list: for every atom ii, its partners jj < ii with a stored density
  ! block, sorted by jj (the CSR row order is what the point-charge kernel
  ! binary-searches).  Offsets are 0-based.  ok = .false. when the MOZYME
  ! index arrays are not available.
  subroutine build_pair_list(pair_i, pair_j, pair_off, row_start, diag_off, iorbs_c, nat_c, &
      npairs, distance_gate, ok)
    use common_arrays_C, only : nat
    use molkst_C, only : numat
    use MOZYME_C, only : iorbs, lijbo, nijbo, iij, numij, ijall, iijj
    implicit none
    integer(c_int), allocatable, intent(out) :: pair_i(:), pair_j(:), pair_off(:), row_start(:), diag_off(:)
    integer(c_int), allocatable, intent(out) :: iorbs_c(:), nat_c(:)
    integer, intent(out) :: npairs, distance_gate
    logical, intent(out) :: ok
    integer :: ii, jj, ix

    ok = .false.
    npairs = 0
    distance_gate = 1
    if (.not. allocated(iorbs)) return
    if (lijbo) then
      if (.not. allocated(nijbo)) return
      distance_gate = 0
    else
      if (.not. allocated(iij) .or. .not. allocated(numij) .or. .not. allocated(ijall) &
          .or. .not. allocated(iijj)) return
    end if
    ! nijbo is symmetric; index it as nijbo(jj, ii) so the inner loop walks a
    ! contiguous column (the strided form cost ~0.3 s per call for 7000 atoms).
    if (lijbo) then
      do ii = 2, numat
        do jj = 1, ii - 1
          if (nijbo(jj, ii) >= 0) npairs = npairs + 1
        end do
      end do
    else
      do ii = 2, numat
        do ix = iij(ii), numij(ii)
          if (ijall(ix) < ii) npairs = npairs + 1
        end do
      end do
    end if
    allocate(pair_i(max(1, npairs)), pair_j(max(1, npairs)), pair_off(max(1, npairs)))
    allocate(row_start(numat + 1), diag_off(numat), iorbs_c(numat), nat_c(numat))
    npairs = 0
    row_start(1) = 0
    do ii = 1, numat
      if (lijbo) then
        diag_off(ii) = int(nijbo(ii, ii), kind=c_int)
        do jj = 1, ii - 1
          if (nijbo(jj, ii) >= 0) then
            npairs = npairs + 1
            pair_i(npairs) = int(ii, kind=c_int)
            pair_j(npairs) = int(jj, kind=c_int)
            pair_off(npairs) = int(nijbo(jj, ii), kind=c_int)
          end if
        end do
      else
        diag_off(ii) = -1
        do ix = iij(ii), numij(ii)
          jj = ijall(ix)
          if (jj == ii) then
            diag_off(ii) = int(iijj(ix), kind=c_int)
          else if (jj < ii) then
            npairs = npairs + 1
            pair_i(npairs) = int(ii, kind=c_int)
            pair_j(npairs) = int(jj, kind=c_int)
            pair_off(npairs) = int(iijj(ix), kind=c_int)
          end if
        end do
      end if
      if (diag_off(ii) < 0) return
      row_start(ii + 1) = int(npairs, kind=c_int)
      iorbs_c(ii) = int(iorbs(ii), kind=c_int)
      nat_c(ii) = int(nat(ii), kind=c_int)
    end do
    ok = .true.
  end subroutine build_pair_list

  subroutine fill_tables(tables)
    use iso_c_binding, only : c_loc
    use molkst_C, only : trunc_1, trunc_2, l_feather, &
      method_PM7, method_PM6, method_PM8, method_pm6_org, method_AM1, method_MNDOD
    use MOZYME_C, only : cutofs
    use overlaps_C, only : cutof1
    use funcon_C, only : ev, a0
    use parameters_C, only : natorb, npq, zs, zp, zd, betas, betap, betad, tore, iod, alp, &
      am, ad, aq, dd, qq, po, ddp, guess1, guess2, guess3, alpb, xfac, v_par
    implicit none
    type(mozyme_pair_tables_c), intent(out) :: tables

    t_natorb = int(natorb, kind=c_int)
    t_npq = int(reshape(npq, [107*3]), kind=c_int)
    t_iod = int(iod, kind=c_int)
    t_zs = zs
    t_zp = zp
    t_zd = zd
    t_betas = betas
    t_betap = betap
    t_betad = betad
    t_tore = tore
    t_alp = alp
    t_am = am
    t_ad = ad
    t_aq = aq
    t_dd = dd
    t_qq = qq
    t_po = reshape(po, [9*107])
    t_ddp = reshape(ddp, [6*107])
    t_guess1 = reshape(guess1, [107*4])
    t_guess2 = reshape(guess2, [107*4])
    t_guess3 = reshape(guess3, [107*4])
    t_alpb = reshape(alpb, [100*100])
    t_xfac = reshape(xfac, [100*100])
    t_v_par = v_par
    tables%natorb = c_loc(t_natorb)
    tables%npq = c_loc(t_npq)
    tables%iod = c_loc(t_iod)
    tables%zs = c_loc(t_zs)
    tables%zp = c_loc(t_zp)
    tables%zd = c_loc(t_zd)
    tables%betas = c_loc(t_betas)
    tables%betap = c_loc(t_betap)
    tables%betad = c_loc(t_betad)
    tables%tore = c_loc(t_tore)
    tables%alp = c_loc(t_alp)
    tables%am = c_loc(t_am)
    tables%ad = c_loc(t_ad)
    tables%aq = c_loc(t_aq)
    tables%dd = c_loc(t_dd)
    tables%qq = c_loc(t_qq)
    tables%po = c_loc(t_po)
    tables%ddp = c_loc(t_ddp)
    tables%guess1 = c_loc(t_guess1)
    tables%guess2 = c_loc(t_guess2)
    tables%guess3 = c_loc(t_guess3)
    tables%alpb = c_loc(t_alpb)
    tables%xfac = c_loc(t_xfac)
    tables%v_par = c_loc(t_v_par)
    tables%a0 = real(a0, kind=c_double)
    tables%ev = real(ev, kind=c_double)
    tables%cutofs = real(cutofs, kind=c_double)
    tables%cutof1 = real(cutof1, kind=c_double)
    tables%trunc_1 = real(trunc_1, kind=c_double)
    tables%trunc_2 = real(trunc_2, kind=c_double)
    tables%method_flags = 0_c_int
    if (method_PM7) tables%method_flags(1) = 1_c_int
    if (method_PM6) tables%method_flags(2) = 1_c_int
    if (method_PM8) tables%method_flags(3) = 1_c_int
    if (method_pm6_org) tables%method_flags(4) = 1_c_int
    if (method_AM1) tables%method_flags(5) = 1_c_int
    if (method_MNDOD) tables%method_flags(6) = 1_c_int
    if (l_feather) tables%method_flags(7) = 1_c_int
  end subroutine fill_tables

  subroutine mozyme_gpu_gradient_run_gpu(numat_in, coord, dxyz, force, chnge, const, code, ms, npairs_out, &
      d_pairs_out)
    use common_arrays_C, only : p
    use molkst_C, only : mpack, cutofp, id, uhf, numat
    use MOZYME_C, only : mode
    use overlaps_C, only : cutof2
    use funcon_C, only : fpc_9
    implicit none
    integer, intent(in) :: numat_in
    double precision, intent(in) :: coord(3, numat_in)
    double precision, intent(inout) :: dxyz(3, numat_in)
    logical, intent(in) :: force
    double precision, intent(in) :: chnge, const
    integer, intent(out) :: code
    double precision, intent(out) :: ms
    integer, intent(out) :: npairs_out, d_pairs_out
    integer(c_int), allocatable :: pair_i(:), pair_j(:), pair_off(:), row_start(:), diag_off(:)
    integer(c_int), allocatable :: iorbs_c(:), nat_c(:)
    type(mozyme_pair_tables_c) :: tables
    integer(c_int) :: d_pairs_c, rc
    real(c_double) :: ms_c
    integer :: npairs, distance_gate
    logical :: ok

    code = 1
    ms = 0.d0
    npairs_out = 0
    d_pairs_out = 0
    if (id /= 0 .or. uhf .or. mode /= 0 .or. numat_in /= numat .or. mpack <= 0) return
    if (.not. allocated(p)) return
    call build_pair_list(pair_i, pair_j, pair_off, row_start, diag_off, iorbs_c, nat_c, &
      npairs, distance_gate, ok)
    if (.not. ok) return
    call fill_tables(tables)
    d_pairs_c = 0_c_int
    ms_c = 0.d0
    rc = mopac_cuda_mozyme_pair_gradient(int(numat, kind=c_int), int(mpack, kind=c_int), &
        int(npairs, kind=c_int), pair_i, pair_j, pair_off, row_start, diag_off, iorbs_c, nat_c, &
        coord, p, int(distance_gate, kind=c_int), merge(1_c_int, 0_c_int, mozyme_gpu_d_pairs_enabled()), &
        real(cutof2, kind=c_double), &
        real(cutofp, kind=c_double), real(chnge, kind=c_double), real(const, kind=c_double), &
        real(fpc_9, kind=c_double), merge(1_c_int, 0_c_int, force), tables, dxyz, ms_c, d_pairs_c)
    code = int(rc)
    ms = ms_c
    npairs_out = npairs
    d_pairs_out = int(d_pairs_c)
  end subroutine mozyme_gpu_gradient_run_gpu

  subroutine mozyme_gpu_hcore_run_gpu(numat_in, coord, h, enuc_add, point_pairs, code, ms, npairs_out, d_pairs_out)
    use molkst_C, only : mpack, id, numat
    use MOZYME_C, only : mode, lijbo, nijbo
    use overlaps_C, only : cutof2
    implicit none
    integer, intent(in) :: numat_in
    double precision, intent(in) :: coord(3, numat_in)
    double precision, intent(inout) :: h(*)
    double precision, intent(out) :: enuc_add
    logical, intent(in) :: point_pairs
    integer, intent(out) :: code
    double precision, intent(out) :: ms
    integer, intent(out) :: npairs_out, d_pairs_out
    integer(c_int), allocatable :: pair_i(:), pair_j(:), pair_off(:), row_start(:), diag_off(:)
    integer(c_int), allocatable :: iorbs_c(:), nat_c(:)
    integer(c_int) :: nijbo_dummy(1)
    type(mozyme_pair_tables_c) :: tables
    integer(c_int) :: d_pairs_c, rc
    real(c_double) :: ms_c, enuc_c
    integer :: npairs, distance_gate
    logical :: ok
    double precision :: t_section

    code = 1
    ms = 0.d0
    enuc_add = 0.d0
    npairs_out = 0
    d_pairs_out = 0
    if (id /= 0 .or. mode /= 0 .or. numat_in /= numat .or. mpack <= 0) return
    call mozyme_section_timer_begin('hcore_gpu_pairlist', t_section)
    call build_pair_list(pair_i, pair_j, pair_off, row_start, diag_off, iorbs_c, nat_c, &
      npairs, distance_gate, ok)
    call mozyme_section_timer_end('hcore_gpu_pairlist', t_section)
    if (.not. ok) return
    call fill_tables(tables)
    call mozyme_section_timer_begin('hcore_gpu_call', t_section)
    d_pairs_c = 0_c_int
    ms_c = 0.d0
    enuc_c = 0.d0
    nijbo_dummy = 0_c_int
    if (point_pairs .and. lijbo .and. allocated(nijbo)) then
      rc = mopac_cuda_mozyme_hcore_pairs(int(numat, kind=c_int), int(mpack, kind=c_int), &
          int(npairs, kind=c_int), pair_i, pair_j, pair_off, row_start, diag_off, iorbs_c, nat_c, &
          coord, int(distance_gate, kind=c_int), merge(1_c_int, 0_c_int, mozyme_gpu_d_pairs_enabled()), &
          real(cutof2, kind=c_double), tables, nijbo, 1_c_int, 1_c_int, h, enuc_c, ms_c, d_pairs_c)
    else
      rc = mopac_cuda_mozyme_hcore_pairs(int(numat, kind=c_int), int(mpack, kind=c_int), &
          int(npairs, kind=c_int), pair_i, pair_j, pair_off, row_start, diag_off, iorbs_c, nat_c, &
          coord, int(distance_gate, kind=c_int), merge(1_c_int, 0_c_int, mozyme_gpu_d_pairs_enabled()), &
          real(cutof2, kind=c_double), tables, nijbo_dummy, 0_c_int, merge(1_c_int, 0_c_int, point_pairs), &
          h, enuc_c, ms_c, d_pairs_c)
    end if
    call mozyme_section_timer_end('hcore_gpu_call', t_section)
    code = int(rc)
    ms = ms_c
    enuc_add = enuc_c
    npairs_out = npairs
    d_pairs_out = int(d_pairs_c)
  end subroutine mozyme_gpu_hcore_run_gpu
#endif

end module mozyme_gpu_gradient
