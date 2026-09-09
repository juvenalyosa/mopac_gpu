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

! GPU evaluation of the MOZYME Cartesian gradient (the pair loop of
! dcart_build_scf_gradient_cpu).  The device code lives in
! src/gpu/mozyme_pair_gradient.cu; this module builds the pair list from the
! MOZYME block index (nijbo or the compact iij/ijall/iijj lists), hands the
! parameter tables over and adds the result on to dxyz.
module mozyme_gpu_gradient
  implicit none
  private
  public :: mozyme_gpu_gradient_enabled
  public :: mozyme_gpu_gradient_check_enabled
  public :: mozyme_gpu_gradient_run

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

  ! MOPAC_MOZYME_GRAD_GPU=1 (set by the MOZYME GPU defaults) enables the path;
  ! MOPAC_NOGPU / MOZYME_GPU_OFF always win.
  logical function mozyme_gpu_gradient_enabled()
    implicit none
#ifdef GPU
    mozyme_gpu_gradient_enabled = env_truthy('MOPAC_MOZYME_GRAD_GPU') .and. &
      .not. env_truthy('MOPAC_NOGPU') .and. .not. env_truthy('MOZYME_GPU_OFF')
#else
    mozyme_gpu_gradient_enabled = .false.
#endif
  end function mozyme_gpu_gradient_enabled

  ! MOPAC_GPU_GRAD_CHECK=1: evaluate both CPU and GPU gradients and report the
  ! difference (the CPU result is kept).
  logical function mozyme_gpu_gradient_check_enabled()
    implicit none
    mozyme_gpu_gradient_check_enabled = env_truthy('MOPAC_GPU_GRAD_CHECK')
  end function mozyme_gpu_gradient_check_enabled

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

#ifdef GPU
  subroutine mozyme_gpu_gradient_run_gpu(numat_in, coord, dxyz, force, chnge, const, code, ms, npairs_out, &
      d_pairs_out)
    use iso_c_binding, only : c_int, c_double
    use common_arrays_C, only : nat, p
    use molkst_C, only : mpack, cutofp, id, uhf, numat, trunc_1, trunc_2, l_feather, &
      method_PM7, method_PM6, method_PM8, method_pm6_org, method_AM1, method_MNDOD
    use MOZYME_C, only : iorbs, lijbo, nijbo, iij, numij, ijall, iijj, cutofs, mode
    use overlaps_C, only : cutof1, cutof2
    use funcon_C, only : fpc_9, ev, a0
    use parameters_C, only : natorb, npq, zs, zp, zd, betas, betap, betad, tore, iod, alp, &
      am, ad, aq, dd, qq, po, ddp, guess1, guess2, guess3, alpb, xfac, v_par
    implicit none
    integer, intent(in) :: numat_in
    double precision, intent(in) :: coord(3, numat_in)
    double precision, intent(inout) :: dxyz(3, numat_in)
    logical, intent(in) :: force
    double precision, intent(in) :: chnge, const
    integer, intent(out) :: code
    double precision, intent(out) :: ms
    integer, intent(out) :: npairs_out, d_pairs_out
    interface
      function mopac_cuda_mozyme_pair_gradient(numat_c, mpack_c, npairs_c, pair_i_c, pair_j_c, &
          pair_off_c, row_start_c, diag_off_c, iorbs_c, nat_c, coord_c, p_c, distance_gate_c, &
          cutof1_c, cutof2_c, cutofp_c, cutofs_c, chnge_c, cnst_c, fpc_9_c, ev_c, a0_c, &
          trunc_1_c, trunc_2_c, force_c, method_flags_c, &
          natorb_c, npq_c, zs_c, zp_c, zd_c, betas_c, betap_c, betad_c, &
          iod_c, tore_c, alp_c, am_c, ad_c, aq_c, dd_c, qq_c, po_c, ddp_c, guess1_c, guess2_c, &
          guess3_c, alpb_c, xfac_c, v_par_c, dxyz_c, ms_c, d_pairs_c) &
          bind(C, name='mopac_cuda_mozyme_pair_gradient') result(rc)
        use iso_c_binding, only : c_int, c_double
        integer(c_int), value :: numat_c, mpack_c, npairs_c, distance_gate_c, force_c
        integer(c_int) :: pair_i_c(*), pair_j_c(*), pair_off_c(*), row_start_c(*), diag_off_c(*)
        integer(c_int) :: iorbs_c(*), nat_c(*), natorb_c(*), npq_c(*), iod_c(*), method_flags_c(*)
        real(c_double) :: coord_c(*), p_c(*), zs_c(*), zp_c(*), zd_c(*), betas_c(*), betap_c(*), betad_c(*)
        real(c_double) :: tore_c(*), alp_c(*), am_c(*), ad_c(*), aq_c(*), dd_c(*), qq_c(*), po_c(*), ddp_c(*)
        real(c_double) :: guess1_c(*), guess2_c(*), guess3_c(*), alpb_c(*), xfac_c(*), v_par_c(*), dxyz_c(*)
        real(c_double), value :: cutof1_c, cutof2_c, cutofp_c, cutofs_c, chnge_c, cnst_c, fpc_9_c, ev_c, a0_c
        real(c_double), value :: trunc_1_c, trunc_2_c
        real(c_double) :: ms_c
        integer(c_int) :: d_pairs_c
        integer(c_int) :: rc
      end function mopac_cuda_mozyme_pair_gradient
    end interface
    integer(c_int), allocatable :: pair_i(:), pair_j(:), pair_off(:), row_start(:), diag_off(:)
    integer(c_int), allocatable :: iorbs_c(:), nat_c(:)
    integer(c_int) :: method_flags(7), d_pairs_c
    real(c_double) :: ms_c
    integer :: ii, jj, ix, npairs, distance_gate
    integer(c_int) :: rc

    code = 1
    ms = 0.d0
    npairs_out = 0
    d_pairs_out = 0
    d_pairs_c = 0_c_int
    if (id /= 0 .or. uhf .or. mode /= 0 .or. numat_in /= numat .or. mpack <= 0) return
    if (.not. allocated(p) .or. .not. allocated(iorbs)) return
    if (lijbo) then
      if (.not. allocated(nijbo)) return
    else
      if (.not. allocated(iij) .or. .not. allocated(numij) .or. .not. allocated(ijall) &
          .or. .not. allocated(iijj)) return
    end if
    !
    !  Pair list: for every atom ii, its partners jj < ii with a stored density
    !  block, sorted by jj (the CSR row order is what the point-charge kernel
    !  binary-searches).  Offsets are 0-based.
    !
    npairs = 0
    if (lijbo) then
      do ii = 2, numat
        do jj = 1, ii - 1
          if (nijbo(ii, jj) >= 0) npairs = npairs + 1
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
          if (nijbo(ii, jj) >= 0) then
            npairs = npairs + 1
            pair_i(npairs) = int(ii, kind=c_int)
            pair_j(npairs) = int(jj, kind=c_int)
            pair_off(npairs) = int(nijbo(ii, jj), kind=c_int)
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
      if (diag_off(ii) < 0) then
        deallocate(pair_i, pair_j, pair_off, row_start, diag_off, iorbs_c, nat_c)
        return
      end if
      row_start(ii + 1) = int(npairs, kind=c_int)
      iorbs_c(ii) = int(iorbs(ii), kind=c_int)
      nat_c(ii) = int(nat(ii), kind=c_int)
    end do
    ! The compact-index route applies the cutof1/cutof2 distance tests inside ijbo().
    distance_gate = 1
    if (lijbo) distance_gate = 0
    method_flags = 0_c_int
    if (method_PM7) method_flags(1) = 1_c_int
    if (method_PM6) method_flags(2) = 1_c_int
    if (method_PM8) method_flags(3) = 1_c_int
    if (method_pm6_org) method_flags(4) = 1_c_int
    if (method_AM1) method_flags(5) = 1_c_int
    if (method_MNDOD) method_flags(6) = 1_c_int
    if (l_feather) method_flags(7) = 1_c_int
    rc = mopac_cuda_mozyme_pair_gradient(int(numat, kind=c_int), int(mpack, kind=c_int), &
        int(npairs, kind=c_int), pair_i, pair_j, pair_off, row_start, diag_off, iorbs_c, nat_c, &
        coord, p, int(distance_gate, kind=c_int), real(cutof1, kind=c_double), &
        real(cutof2, kind=c_double), real(cutofp, kind=c_double), real(cutofs, kind=c_double), &
        real(chnge, kind=c_double), real(const, kind=c_double), real(fpc_9, kind=c_double), &
        real(ev, kind=c_double), real(a0, kind=c_double), real(trunc_1, kind=c_double), &
        real(trunc_2, kind=c_double), merge(1_c_int, 0_c_int, force), method_flags, &
        natorb, npq, zs, zp, zd, betas, betap, betad, &
        iod, tore, alp, am, ad, aq, dd, qq, po, ddp, guess1, guess2, guess3, alpb, xfac, v_par, &
        dxyz, ms_c, d_pairs_c)
    code = int(rc)
    ms = ms_c
    npairs_out = npairs
    d_pairs_out = int(d_pairs_c)
    deallocate(pair_i, pair_j, pair_off, row_start, diag_off, iorbs_c, nat_c)
  end subroutine mozyme_gpu_gradient_run_gpu
#endif

end module mozyme_gpu_gradient
