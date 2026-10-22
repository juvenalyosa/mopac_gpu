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
  use iso_c_binding, only : c_int, c_long_long, c_double, c_ptr
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
  public :: conv_gpu_gradient_enabled
  public :: conv_gpu_gradient_run
  public :: conv_gpu_hcore_enabled
  public :: conv_gpu_solrot_run
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

    function mopac_cuda_pair_gradient_ext(nent_c, np_c, nitems_c, item_i_c, item_j_c, item_w_c, &
        first_c, iorbs_c, nat_c, coord_c, p_c, periodic_c, clower_c, cupper_c, trunk_c_c, trunk_cr_c, &
        trunk_cr2_c, trunk_clim_c, wlim_c, chnge_c, cnst_c, fpc_9_c, force_c, tables_c, dxyz_c, ms_c, &
        cpu_pairs_c) bind(C, name='mopac_cuda_pair_gradient_ext') result(rc)
      import :: c_int, c_long_long, c_double, mozyme_pair_tables_c
      integer(c_int), value :: nent_c, nitems_c, periodic_c, force_c
      integer(c_long_long), value :: np_c
      integer(c_int) :: item_i_c(*), item_j_c(*), first_c(*), iorbs_c(*), nat_c(*)
      real(c_double) :: item_w_c(*), coord_c(*), p_c(*), dxyz_c(*)
      real(c_double), value :: clower_c, cupper_c, trunk_c_c, trunk_cr_c, trunk_cr2_c, trunk_clim_c
      real(c_double), value :: wlim_c, chnge_c, cnst_c, fpc_9_c
      type(mozyme_pair_tables_c) :: tables_c
      real(c_double) :: ms_c
      integer(c_int) :: cpu_pairs_c
      integer(c_int) :: rc
    end function mopac_cuda_pair_gradient_ext

    function mopac_cuda_solrot_pairs(npairs_c, pair_ni_c, pair_nj_c, pair_woff_c, pair_kb_c, pair_first_c, &
        pair_self_c, pair_xi_c, pbest_w1_c, pbest_img_c, nitems_c, item_pair_c, item_img_c, item_xj_c, &
        item_xjuc_c, tables_c, w_c, wk_c, di_c, e1b_c, e2a_c, enuc_c, win_c, ms_c) &
        bind(C, name='mopac_cuda_solrot_pairs') result(rc)
      import :: c_int, c_long_long, c_double, mozyme_pair_tables_c
      integer(c_int), value :: npairs_c, nitems_c
      integer(c_int) :: pair_ni_c(*), pair_nj_c(*), pair_kb_c(*), pair_first_c(*), pair_self_c(*)
      integer(c_int) :: pbest_img_c(*), item_pair_c(*), item_img_c(*), win_c(*)
      integer(c_long_long) :: pair_woff_c(*)
      real(c_double) :: pair_xi_c(*), pbest_w1_c(*), item_xj_c(*), item_xjuc_c(*)
      real(c_double) :: w_c(*), wk_c(*), di_c(*), e1b_c(*), e2a_c(*), enuc_c(*)
      type(mozyme_pair_tables_c) :: tables_c
      real(c_double) :: ms_c
      integer(c_int) :: rc
    end function mopac_cuda_solrot_pairs
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
#ifdef GPU
    use mod_vars_cuda, only : lgpu, mozyme_gpu
#endif
    implicit none
#ifdef GPU
    ! One kill switch for every MOZYME GPU helper (hcore, gradient, dispersion,
    ! H-bond correction): the NOGPU keyword (lgpu / mozyme_gpu cleared in
    ! run_mopac) or MOPAC_NOGPU / MOZYME_GPU_OFF in the environment.
    gpu_allowed = lgpu .and. mozyme_gpu .and. &
      .not. env_truthy('MOPAC_NOGPU') .and. .not. env_truthy('MOZYME_GPU_OFF')
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

  ! Conventional (dense density) SCF, molecules and solids: the pair loop of
  ! dcart_build_scf_gradient_cpu on the device unless MOPAC_GPU_CONV_GRAD=0
  ! (T4, 2026-10-20: max |CPU - GPU| <= 4.5e-7 kcal/mol/A on Si, quartz, urea).
  ! Not for UHF (the device energy takes pa = pb = p/2), MOZYME (its own
  ! route), or id = 3 with methods other than PM7 (rotate's nddo_to_point).
  logical function conv_gpu_gradient_enabled()
#ifdef GPU
    use mod_vars_cuda, only : lgpu
    use molkst_C, only : mozyme, uhf, id, method_pm7
#endif
    implicit none
#ifdef GPU
    conv_gpu_gradient_enabled = lgpu .and. .not. env_truthy('MOPAC_NOGPU') .and. &
      .not. env_falsy('MOPAC_GPU_CONV_GRAD') .and. .not. mozyme .and. .not. uhf .and. &
      .not. (id == 3 .and. .not. method_pm7)
#else
    conv_gpu_gradient_enabled = .false.
#endif
  end function conv_gpu_gradient_enabled

  ! Adds the pair (and, in solids, the distant point-charge) contributions of
  ! dcart_build_scf_gradient_cpu on to dxyz(3, numat*l123), q = tore - charge
  ! per atom.  code: 0 success, 1 unsupported (nothing done), 2 device failure
  ! (nothing done), 3 some pair failed on the device (nothing done).
  subroutine conv_gpu_gradient_run(numat_in, l123_in, coord, dxyz, q, force, chnge, const, code, ms, &
      nitems_out)
    implicit none
    integer, intent(in) :: numat_in, l123_in
    double precision, intent(in) :: coord(3, numat_in), q(numat_in)
    double precision, intent(inout) :: dxyz(3, numat_in*l123_in)
    logical, intent(in) :: force
    double precision, intent(in) :: chnge, const
    integer, intent(out) :: code, nitems_out
    double precision, intent(out) :: ms
#ifdef GPU
    call conv_gpu_gradient_run_gpu(numat_in, l123_in, coord, dxyz, q, force, chnge, const, code, ms, &
      nitems_out)
#else
    code = 1
    ms = 0.d0
    nitems_out = 0
    if (numat_in < 0 .or. force .or. chnge < 0.d0 .or. const < 0.d0 .or. q(1) /= q(1)) code = 1
    if (coord(1, 1) /= coord(1, 1)) dxyz(1, 1) = dxyz(1, 1)
#endif
  end subroutine conv_gpu_gradient_run

  ! Solids (id /= 0), conventional SCF: the image sums of hcore / solrot for
  ! the images closer than clower on the device unless MOPAC_GPU_CONV_HCORE=0
  ! (T4, 2026-10-20: h, w, e1b, e2a, enuc within 3e-11 of the CPU on Si, quartz,
  ! rutile, urea; wk can take another of several equidistant images, a choice
  ! the CPU makes on the last bit of w(1)).
  logical function conv_gpu_hcore_enabled()
#ifdef GPU
    use mod_vars_cuda, only : lgpu
    use molkst_C, only : mozyme, id, method_pm7
#endif
    implicit none
#ifdef GPU
    conv_gpu_hcore_enabled = lgpu .and. .not. env_truthy('MOPAC_NOGPU') .and. &
      .not. env_falsy('MOPAC_GPU_CONV_HCORE') .and. .not. mozyme .and. id /= 0 .and. &
      .not. (id == 3 .and. .not. method_pm7)
#else
    conv_gpu_hcore_enabled = .false.
#endif
  end function conv_gpu_hcore_enabled

  ! For every pair (i, j <= i) of a solid, in hcore's order, the sums over the
  ! images of solrot and of hcore's h1elec loop: w and wk (common_arrays_C) get
  ! the pair's J and K integrals at their place, gdi(81, pair) the summed
  ! h1elec block (9 x 9, column-major), ge1b / ge2a / genuc the summed
  ! rotate / point terms (genuc times solrot's one) and gkb the number of
  ! integrals of the pair.  code: 0 success, otherwise nothing usable.
  subroutine conv_gpu_solrot_run(gdi, ge1b, ge2a, genuc, gkb, code, ms, nitems_out)
    implicit none
    double precision, allocatable, intent(out) :: gdi(:,:), ge1b(:,:), ge2a(:,:), genuc(:)
    integer, allocatable, intent(out) :: gkb(:)
    integer, intent(out) :: code, nitems_out
    double precision, intent(out) :: ms
#ifdef GPU
    call conv_gpu_solrot_run_gpu(gdi, ge1b, ge2a, genuc, gkb, code, ms, nitems_out)
#else
    allocate(gdi(81, 1), ge1b(45, 1), ge2a(45, 1), genuc(1), gkb(1))
    code = 1
    ms = 0.d0
    nitems_out = 0
#endif
  end subroutine conv_gpu_solrot_run

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
    use molkst_C, only : numat, mpack
    use MOZYME_C, only : iorbs, lijbo, nijbo, iij, numij, ijall, iijj
    use fillij_changes_C, only : nijbo_fill_count, pl_i, pl_j, pl_off, pl_n, pl_fill
    implicit none
    integer(c_int), allocatable, intent(out) :: pair_i(:), pair_j(:), pair_off(:), row_start(:), diag_off(:)
    integer(c_int), allocatable, intent(out) :: iorbs_c(:), nat_c(:)
    integer, intent(out) :: npairs, distance_gate
    logical, intent(out) :: ok
    integer :: ii, jj, ix
    ! hcore and the gradient build the same list from the same nijbo within a
    ! geometry step (55 ms each at 7000 atoms): keep the last one, keyed by
    ! the fillij fill pass, numat and mpack.
    integer(c_int), allocatable, save :: c_pair_i(:), c_pair_j(:), c_pair_off(:), c_row_start(:), c_diag_off(:)
    integer(c_int), allocatable, save :: c_iorbs(:), c_nat(:)
    integer, save :: c_fill = -1, c_numat = -1, c_mpack = -1, c_npairs = 0, c_gate = 1
    logical, save :: c_lijbo = .false.

    ok = .false.
    npairs = 0
    distance_gate = 1
    if (.not. allocated(iorbs)) return
    ! (.eqv. binds looser than .and.: keep it parenthesised)
    if (c_fill >= 0 .and. c_fill == nijbo_fill_count .and. c_numat == numat .and. &
        c_mpack == mpack .and. (c_lijbo .eqv. lijbo) .and. allocated(c_pair_i)) then
      pair_i = c_pair_i
      pair_j = c_pair_j
      pair_off = c_pair_off
      row_start = c_row_start
      diag_off = c_diag_off
      iorbs_c = c_iorbs
      nat_c = c_nat
      npairs = c_npairs
      distance_gate = c_gate
      ok = .true.
      return
    end if
    if (lijbo) then
      if (.not. allocated(nijbo)) return
      distance_gate = 0
    else
      if (.not. allocated(iij) .or. .not. allocated(numij) .or. .not. allocated(ijall) &
          .or. .not. allocated(iijj)) return
    end if
    ! The fillij grid pass keeps the sorted (ii, jj < ii) block-pair list with
    ! its offsets: take it and only derive the row starts (O(numat)).
    if (lijbo .and. pl_fill == nijbo_fill_count .and. pl_n > 0 .and. allocated(pl_i)) then
      npairs = pl_n
      allocate(pair_i(npairs), pair_j(npairs), pair_off(npairs))
      allocate(row_start(numat + 1), diag_off(numat), iorbs_c(numat), nat_c(numat))
      pair_i(1:npairs) = int(pl_i(1:npairs), kind=c_int)
      pair_j(1:npairs) = int(pl_j(1:npairs), kind=c_int)
      pair_off(1:npairs) = int(pl_off(1:npairs), kind=c_int)
      row_start = 0_c_int
      do ix = 1, npairs
        row_start(pl_i(ix) + 1) = row_start(pl_i(ix) + 1) + 1_c_int
      end do
      do ii = 1, numat
        row_start(ii + 1) = row_start(ii + 1) + row_start(ii)
        diag_off(ii) = int(nijbo(ii, ii), kind=c_int)
        if (diag_off(ii) < 0) return
        iorbs_c(ii) = int(iorbs(ii), kind=c_int)
        nat_c(ii) = int(nat(ii), kind=c_int)
      end do
      ok = .true.
      c_pair_i = pair_i
      c_pair_j = pair_j
      c_pair_off = pair_off
      c_row_start = row_start
      c_diag_off = diag_off
      c_iorbs = iorbs_c
      c_nat = nat_c
      c_npairs = npairs
      c_gate = distance_gate
      c_fill = nijbo_fill_count
      c_numat = numat
      c_mpack = mpack
      c_lijbo = lijbo
      return
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
    c_pair_i = pair_i
    c_pair_j = pair_j
    c_pair_off = pair_off
    c_row_start = row_start
    c_diag_off = diag_off
    c_iorbs = iorbs_c
    c_nat = nat_c
    c_npairs = npairs
    c_gate = distance_gate
    c_fill = nijbo_fill_count
    c_numat = numat
    c_mpack = mpack
    c_lijbo = lijbo
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

  subroutine conv_gpu_gradient_run_gpu(numat_in, l123_in, coord, dxyz, q, force, chnge, const, code, ms, &
      nitems_out)
    use common_arrays_C, only : p, nfirst, nlast, nat, tvec, wk_image
    use molkst_C, only : id, mpack, cutofp, clower, cupper, l1u, l2u, l3u
    use funcon_C, only : fpc_9, a0, ev
    implicit none
    integer, intent(in) :: numat_in, l123_in
    double precision, intent(in) :: coord(3, numat_in), q(numat_in)
    double precision, intent(inout) :: dxyz(3, numat_in*l123_in)
    logical, intent(in) :: force
    double precision, intent(in) :: chnge, const
    integer, intent(out) :: code, nitems_out
    double precision, intent(out) :: ms
    double precision, external :: derp
    integer(c_int), allocatable :: item_i(:), item_j(:), first_e(:), iorbs_e(:), nat_e(:)
    double precision, allocatable :: item_w(:), coord_e(:,:), dfar(:,:)
    type(mozyme_pair_tables_c) :: tables
    integer(c_int) :: rc, cpu_pairs_c
    real(c_double) :: ms_c
    integer :: ii, jj, ik, jk, kl, kkkk, iii, jjj, k, icuc, ione, nent, nitems, pass, n, knear
    double precision :: near2, rij, ee, deriv, half, d(3), t_section, rnear
    double precision :: bound1, bound2, range, cupper_l, c, cr, cr2, clim, wlim

    code = 1
    ms = 0.d0
    nitems_out = 0
    if (.not. allocated(p) .or. mpack <= 0) return
    do ii = 1, numat_in
      n = nlast(ii) - nfirst(ii) + 1
      if (n /= 1 .and. n /= 4 .and. n /= 9) return
    end do
    icuc = (l123_in + 1)/2
    ione = 1
    if (id /= 0) ione = 0
    nent = numat_in*l123_in
    allocate(coord_e(3, nent), first_e(nent), iorbs_e(nent), nat_e(nent), dfar(3, nent))
    dfar = 0.d0
    do jj = 1, numat_in
      kkkk = 0
      do ik = -l1u, l1u
        do jk = -l2u, l2u
          do kl = -l3u, l3u
            kkkk = kkkk + 1
            k = l123_in*(jj - 1) + kkkk
            coord_e(:,k) = coord(:,jj) + tvec(:,1)*ik + tvec(:,2)*jk + tvec(:,3)*kl
            first_e(k) = int(nfirst(jj), kind=c_int)
            iorbs_e(k) = int(nlast(jj) - nfirst(jj) + 1, kind=c_int)
            nat_e(k) = int(nat(jj), kind=c_int)
          end do
        end do
      end do
    end do
    near2 = (2.d0/3.d0*cutofp)**2
    call mozyme_section_timer_begin('conv_grad_pairlist', t_section)
!
!  Pass 1 counts the device pairs, pass 2 stores them.  Pairs beyond 2/3 of
!  cutofp in a solid are point charges (dcart's derp term), done here.
!
    nitems = 0
    do pass = 1, 2
      if (pass == 2) then
        allocate(item_i(max(1, nitems)), item_j(max(1, nitems)), item_w(max(1, nitems)))
        nitems = 0
      end if
      do ii = 1, numat_in
        iii = l123_in*(ii - 1)
        do jj = 1, ii - ione
          half = 1.d0
          if (ii == jj) half = 0.5d0
          jjj = l123_in*(jj - 1)
          ! the image whose exchange integrals the energy has (dcart's dhc_exchange): as
          ! hcore recorded it, otherwise the first nearest one; none for ii = jj
          knear = 0
          if (id /= 0 .and. allocated(wk_image)) then
            knear = wk_image((ii*(ii - 1))/2 + jj)
          else if (id /= 0 .and. pass == 2 .and. ii /= jj) then
            rnear = huge(1.d0)
            do kkkk = 1, l123_in
              d = coord_e(:, jjj + kkkk) - coord(:, ii)
              rij = d(1)**2 + d(2)**2 + d(3)**2
              if (rij < rnear) then
                rnear = rij
                knear = kkkk
              end if
            end do
          end if
          do kkkk = 1, l123_in
            if (ii == jj .and. kkkk == icuc) cycle   ! the atom with itself: the two terms cancel
            if (id /= 0) then
              d = coord_e(:, jjj + kkkk) - coord(:, ii)
              rij = d(1)**2 + d(2)**2 + d(3)**2
              if (rij > near2) then
                rij = sqrt(rij)
                ! derp is zero beyond clower and cupper: nothing to add
                if (pass == 2 .and. .not. (rij > clower .and. rij > cupper)) then
                  ee = q(ii)*q(jj)*fpc_9*ev*a0*derp(rij)
                  do k = 1, 3
                    deriv = half*ee*d(k)/rij
                    dfar(k, iii + icuc) = dfar(k, iii + icuc) - deriv
                    dfar(k, jjj + kkkk) = dfar(k, jjj + kkkk) + deriv
                  end do
                end if
                cycle
              end if
            end if
            nitems = nitems + 1
            if (pass == 2) then
              item_i(nitems) = int(iii + icuc, kind=c_int)
              item_j(nitems) = int(jjj + kkkk, kind=c_int)
              item_w(nitems) = half
              ! periodic: a negative weight tells the kernel to leave out the exchange integrals
              if (id /= 0 .and. kkkk /= knear) item_w(nitems) = -half
            end if
          end do
        end do
      end do
    end do
!
!  trunk() and dhc()'s constants for solids
!
    bound1 = 0.d0
    cupper_l = 0.d0
    c = 0.d0
    cr = 0.d0
    cr2 = 0.d0
    clim = 0.d0
    wlim = 0.d0
    if (id /= 0) then
      bound1 = clower / cutofp
      cupper_l = cutofp + 0
      bound2 = cupper_l / cutofp
      range = bound2 - bound1
      c = -0.5d0 * bound1 ** 2 * cutofp / range
      cr = 1.0d0 + bound1 / range
      cr2 = -1 / (cutofp*2*range)
      clim = c + cr * cupper_l + cr2 * cupper_l ** 2
      wlim = 4.d0
    end if
    call mozyme_section_timer_end('conv_grad_pairlist', t_section)
    call fill_tables(tables)
    cpu_pairs_c = 0_c_int
    ms_c = 0.d0
    call mozyme_section_timer_begin('conv_grad_device', t_section)
    rc = mopac_cuda_pair_gradient_ext(int(nent, kind=c_int), int(mpack, kind=c_long_long), &
        int(nitems, kind=c_int), item_i, item_j, item_w, first_e, iorbs_e, nat_e, coord_e, p, &
        merge(1_c_int, 0_c_int, id /= 0), real(clower, kind=c_double), real(cupper_l, kind=c_double), &
        real(c, kind=c_double), real(cr, kind=c_double), real(cr2, kind=c_double), real(clim, kind=c_double), &
        real(wlim, kind=c_double), real(chnge, kind=c_double), real(const, kind=c_double), &
        real(fpc_9, kind=c_double), merge(1_c_int, 0_c_int, force), tables, dxyz, ms_c, cpu_pairs_c)
    call mozyme_section_timer_end('conv_grad_device', t_section)
    code = int(rc)
    if (code == 0) dxyz = dxyz + dfar
    ms = ms_c
    nitems_out = nitems
  end subroutine conv_gpu_gradient_run_gpu

  subroutine conv_gpu_solrot_run_gpu(gdi, ge1b, ge2a, genuc, gkb, code, ms, nitems_out)
    use common_arrays_C, only : coord, nat, tvec, w, wk, wk_image
    use molkst_C, only : numat, l1u, l2u, l3u, clower
    use parameters_C, only : natorb, tore
    use funcon_C, only : ev, a0
    use overlaps_C, only : cutof1
    implicit none
    double precision, allocatable, intent(out) :: gdi(:,:), ge1b(:,:), ge2a(:,:), genuc(:)
    integer, allocatable, intent(out) :: gkb(:)
    integer, intent(out) :: code, nitems_out
    double precision, intent(out) :: ms
    double precision, external :: trunk
    integer(c_int), allocatable :: pair_ni(:), pair_nj(:), pair_kb(:), pair_first(:), pair_self(:), pbest_img(:)
    integer(c_int), allocatable :: win(:)
    integer(c_int), allocatable :: item_pair(:), item_img(:)
    integer(c_long_long), allocatable :: pair_woff(:)
    double precision, allocatable :: pair_xi(:,:), pbest_w1(:), item_xj(:,:), item_xjuc(:,:)
    double precision, allocatable :: sw(:), se1(:), se2(:), sen(:)
    type(mozyme_pair_tables_c) :: tables
    integer(c_int) :: rc
    real(c_double) :: ms_c
    integer :: i, j, ii, jj, k, kkkk, p, npairs, nitems, pass, kr, ni, nj, li, lj, nii, njj, l
    double precision :: xj(3), xjuc(3), r, cutof2, one, ee, ee1, ee2, t_section

    code = 1
    ms = 0.d0
    nitems_out = 0
    ! the device sees only the images within clower: overlaps must vanish beyond it
    if (clower**2 < cutof1) return
    do i = 1, numat
      l = natorb(nat(i))
      if (l /= 1 .and. l /= 4 .and. l /= 9) return
    end do
    npairs = (numat*(numat + 1))/2
    allocate(pair_ni(npairs), pair_nj(npairs), pair_kb(npairs), pair_first(npairs + 1), pair_self(npairs))
    allocate(pbest_img(npairs), win(npairs), pair_woff(npairs), pair_xi(3, npairs), pbest_w1(npairs))
    allocate(sw(npairs), se1(npairs), se2(npairs), sen(npairs))
    allocate(gdi(81, npairs), ge1b(45, npairs), ge2a(45, npairs), genuc(npairs), gkb(npairs))
    cutof2 = clower**2
    nitems = 0
    call mozyme_section_timer_begin('conv_hcore_pairlist', t_section)
    do pass = 1, 2
      if (pass == 2) then
        allocate(item_pair(max(1, nitems)), item_img(max(1, nitems)), item_xj(3, max(1, nitems)), &
          item_xjuc(3, max(1, nitems)))
        nitems = 0
      end if
      kr = 1
      p = 0
      do i = 1, numat
        ni = nat(i)
        li = natorb(ni)
        nii = (li*(li + 1))/2
        do j = 1, i
          nj = nat(j)
          lj = natorb(nj)
          njj = (lj*(lj + 1))/2
          p = p + 1
          if (pass == 1) then
            pair_ni(p) = int(ni, kind=c_int)
            pair_nj(p) = int(nj, kind=c_int)
            pair_kb(p) = int(nii*njj, kind=c_int)
            pair_woff(p) = int(kr - 1, kind=c_long_long)
            pair_xi(:, p) = coord(:, i)
            one = 1.d0
            if (Abs(coord(1,i) - coord(1,j)) <1.d-20 .and. Abs(coord(2,i) - coord(2,j)) <1.d-20 .and. &
              Abs(coord(3,i) - coord(3,j)) <1.d-20) one = 0.5D0
            pair_self(p) = merge(1_c_int, 0_c_int, one < 0.9d0)
            pbest_w1(p) = 0.d0
            pbest_img(p) = 0_c_int
            sw(p) = 0.d0
            se1(p) = 0.d0
            se2(p) = 0.d0
            sen(p) = 0.d0
          else
            pair_first(p) = int(nitems, kind=c_int)
            one = merge(0.5d0, 1.d0, pair_self(p) == 1_c_int)
          end if
          kr = kr + nii*njj
          kkkk = 0
          do ii = -l1u, l1u
            do jj = -l2u, l2u
              do k = -l3u, l3u
                kkkk = kkkk + 1
                xj = coord(:,j) + tvec(:,1)*ii + tvec(:,2)*jj + tvec(:,3)*k
                xjuc = xj - coord(:,i)
                r = xjuc(1)**2 + xjuc(2)**2 + xjuc(3)**2
                if (r > cutof2) then
                  if (pass == 1) then
                    ! solrot's point(): summed here
                    r = sqrt(r)
                    r = trunk(r)
                    ee = ev*a0/r
                    ee1 = -ee*tore(nj)
                    ee2 = -ee*tore(ni)
                    sw(p) = sw(p) + ee
                    se1(p) = se1(p) + ee1
                    se2(p) = se2(p) + ee2
                    sen(p) = sen(p) + (-ee1*tore(ni))*one
                    if (pbest_w1(p) < ee) then
                      pbest_w1(p) = ee
                      pbest_img(p) = int(kkkk, kind=c_int)
                    end if
                  end if
                  cycle
                end if
                if (r < 0.00002D0) cycle   ! rotate and h1elec give zero
                nitems = nitems + 1
                if (pass == 2) then
                  item_pair(nitems) = int(p - 1, kind=c_int)
                  item_img(nitems) = int(kkkk, kind=c_int)
                  item_xj(:, nitems) = xj
                  item_xjuc(:, nitems) = xjuc
                end if
              end do
            end do
          end do
        end do
        kr = kr + nii*nii
      end do
    end do
    pair_first(npairs + 1) = int(nitems, kind=c_int)
    gkb = pair_kb
    call mozyme_section_timer_end('conv_hcore_pairlist', t_section)
    call fill_tables(tables)
    ms_c = 0.d0
    call mozyme_section_timer_begin('conv_hcore_device', t_section)
    rc = mopac_cuda_solrot_pairs(int(npairs, kind=c_int), pair_ni, pair_nj, pair_woff, pair_kb, pair_first, &
      pair_self, pair_xi, pbest_w1, pbest_img, int(nitems, kind=c_int), item_pair, item_img, item_xj, &
      item_xjuc, tables, w, wk, gdi, ge1b, ge2a, genuc, win, ms_c)
    call mozyme_section_timer_end('conv_hcore_device', t_section)
    code = int(rc)
    ms = ms_c
    nitems_out = nitems
    if (code /= 0) return
    wk_image(1:npairs) = int(win(1:npairs))
!
!  Add the point-charge images
!
    do p = 1, npairs
      li = natorb(pair_ni(p))
      lj = natorb(pair_nj(p))
      njj = (lj*(lj + 1))/2
      do ii = 1, li
        do jj = 1, lj
          k = int(pair_woff(p)) + ((ii*(ii + 1))/2 - 1)*njj + (jj*(jj + 1))/2
          w(k) = w(k) + sw(p)
        end do
        ge1b((ii*(ii + 1))/2, p) = ge1b((ii*(ii + 1))/2, p) + se1(p)
      end do
      do jj = 1, lj
        ge2a((jj*(jj + 1))/2, p) = ge2a((jj*(jj + 1))/2, p) + se2(p)
      end do
      genuc(p) = genuc(p) + sen(p)
    end do
  end subroutine conv_gpu_solrot_run_gpu

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
