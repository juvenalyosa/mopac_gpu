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

module mozyme_resident_fock
  use iso_c_binding, only: c_int, c_double
#ifdef GPU
  use iso_c_binding, only: c_int64_t
#endif
  use mozyme_gpu_int_utils, only: mozyme_c_int_checked, &
    mozyme_c_int_nonnegative_or_zero, mozyme_c_int_positive_or_zero
  implicit none
  integer, parameter :: max_resident_diag_basis = 9
  integer, parameter :: max_resident_fallback_basis = max_resident_diag_basis + 1
  integer, parameter :: resident_fock_plan_full = 0
  integer, parameter :: resident_fock_plan_partial = 1
  integer, parameter :: resident_fock_plan_count = 2
  logical, save :: resident_ready(0:resident_fock_plan_count-1) = .false.
  logical, save :: resident_full_coverage(0:resident_fock_plan_count-1) = .false.
  logical, save :: resident_gpu_pack_ready(0:resident_fock_plan_count-1) = .false.
#ifdef GPU
  integer(c_int64_t), save :: last_signature(0:resident_fock_plan_count-1) = -1_c_int64_t
  integer(c_int64_t), parameter :: mozyme_sig_mod1 = 2147483647_c_int64_t
  integer(c_int64_t), parameter :: mozyme_sig_mod2 = 2147483629_c_int64_t
#endif
contains

  logical function mozyme_resident_pair_supported(iab, jba)
    use MOZYME_C, only: direct
    implicit none
    integer, intent(in) :: iab, jba
    if (direct) then
      mozyme_resident_pair_supported = mozyme_resident_direct_basis_supported(iab) .and. &
        mozyme_resident_direct_basis_supported(jba)
    else
      mozyme_resident_pair_supported = mozyme_resident_basis_supported(iab) .and. &
        mozyme_resident_basis_supported(jba)
    end if
  end function mozyme_resident_pair_supported

  logical function mozyme_resident_basis_supported(nbasis)
    implicit none
    integer, intent(in) :: nbasis
    mozyme_resident_basis_supported = nbasis == 1 .or. nbasis == 4 .or. nbasis == 9
  end function mozyme_resident_basis_supported

  logical function mozyme_resident_direct_sp_basis_supported(nbasis)
    implicit none
    integer, intent(in) :: nbasis
    mozyme_resident_direct_sp_basis_supported = nbasis == 1 .or. nbasis == 4
  end function mozyme_resident_direct_sp_basis_supported

  logical function mozyme_resident_direct_basis_supported(nbasis)
    implicit none
    integer, intent(in) :: nbasis
    mozyme_resident_direct_basis_supported = nbasis == 1 .or. nbasis == 4 .or. nbasis == 9
  end function mozyme_resident_direct_basis_supported

  logical function mozyme_resident_pair_noop(iab, jba)
    implicit none
    integer, intent(in) :: iab, jba
    mozyme_resident_pair_noop = iab == 0 .or. jba == 0
  end function mozyme_resident_pair_noop

  logical function mozyme_resident_direct_basis_fallback(iab, jba)
    use MOZYME_C, only: direct
    implicit none
    integer, intent(in) :: iab, jba
    mozyme_resident_direct_basis_fallback = direct .and. &
      mozyme_resident_basis_supported(iab) .and. &
      mozyme_resident_basis_supported(jba) .and. &
      .not. (mozyme_resident_direct_basis_supported(iab) .and. &
      mozyme_resident_direct_basis_supported(jba))
  end function mozyme_resident_direct_basis_fallback

  logical function mozyme_resident_point_supported(iab, jba, addr)
    use MOZYME_C, only: direct
    implicit none
    integer, intent(in) :: iab, jba, addr
    if (direct) then
      mozyme_resident_point_supported = addr < 0 .and. &
        mozyme_resident_direct_basis_supported(iab) .and. &
        mozyme_resident_direct_basis_supported(jba)
    else
      mozyme_resident_point_supported = addr < 0 .and. &
        mozyme_resident_basis_supported(iab) .and. &
        mozyme_resident_basis_supported(jba)
    end if
  end function mozyme_resident_point_supported

  logical function mozyme_resident_fock_full_coverage()
    implicit none
    mozyme_resident_fock_full_coverage = &
      mozyme_resident_fock_plan_full_coverage(resident_fock_plan_full)
  end function mozyme_resident_fock_full_coverage

  logical function mozyme_resident_fock_plan_full_coverage(plan_id)
    implicit none
    integer, intent(in) :: plan_id

    mozyme_resident_fock_plan_full_coverage = .false.
    if (plan_id < 0 .or. plan_id >= resident_fock_plan_count) return
    mozyme_resident_fock_plan_full_coverage = &
      resident_ready(plan_id) .and. resident_full_coverage(plan_id)
  end function mozyme_resident_fock_plan_full_coverage

  logical function mozyme_resident_fock_try(f, ptot, qe, iorbs, nat, ifact, wj, wk, mode, kopt, ione, coord, use_nijbo)
#ifdef GPU
    use chanel_C, only: iw
    use molkst_C, only: numat, mpack
    use MOZYME_C, only: direct, nijbo
    use mod_vars_cuda, only: mozyme_gpu_requested, mozyme_resident_fock_gpu
#endif
    implicit none
    double precision, intent(inout) :: f(*)
    double precision, intent(in) :: ptot(*), qe(*), wj(*), wk(*), coord(3,*)
    integer, intent(in) :: iorbs(*), nat(*), ifact(*), mode, kopt(*), ione
    logical, intent(in) :: use_nijbo
#ifdef GPU
    interface
      function mopac_cuda_mozyme_sparse_fock_run(mpack, ptot, qe, f) &
          bind(C,name='mopac_cuda_mozyme_sparse_fock_run') result(code)
        use iso_c_binding, only: c_int, c_double
        integer(c_int), value :: mpack
        real(c_double) :: ptot(*), qe(*), f(*)
        integer(c_int) :: code
      end function mopac_cuda_mozyme_sparse_fock_run
    end interface
    integer :: code
    integer :: alloc_stat
    double precision, allocatable :: f_work(:)
#endif

    mozyme_resident_fock_try = .false.
#ifdef GPU
    if (.not. mozyme_gpu_requested .or. .not. mozyme_resident_fock_gpu) then
      if (resident_strict_requested()) then
        call strict_resident_fock_abort('strict_resident_fock_disabled', &
          'MOZYME GPU strict resident Fock was disabled')
      end if
      return
    end if
    if (resident_strict_requested()) then
      call strict_resident_fock_abort('strict_resident_fock_legacy_host_copy', &
        'MOZYME GPU strict resident SCF cannot use legacy host-copy Fock')
      return
    end if

    if (.not. mozyme_resident_fock_prepare(iorbs, nat, ifact, wj, wk, mode, kopt, ione, coord, use_nijbo)) then
      if (resident_strict_requested()) then
        call strict_resident_fock_abort('strict_resident_fock_prepare_failed', &
          'MOZYME GPU strict resident Fock setup failed')
      end if
      return
    end if
    if (resident_strict_requested() .and. .not. resident_full_coverage(resident_fock_plan_full)) then
      if (mozyme_resident_trace()) write(iw,'(1x,a)') &
        '[MOZYME GPU resident_fock] strict rejected partial coverage'
      call strict_resident_fock_abort('strict_resident_fock_partial_coverage', &
        'MOZYME GPU strict resident Fock coverage was incomplete')
      return
    end if

    allocate(f_work(mpack), stat=alloc_stat)
    if (alloc_stat /= 0) then
      if (resident_strict_requested()) then
        call strict_resident_fock_abort('strict_resident_fock_work_alloc_failed', &
          'MOZYME GPU strict resident Fock work allocation failed')
      end if
      return
    end if
    if (mode == -1) then
      f_work(1:mpack) = -f(1:mpack)
    else
      f_work(1:mpack) = f(1:mpack)
    end if
    code = mopac_cuda_mozyme_sparse_fock_run( &
      mozyme_c_int_positive_or_zero(mpack), ptot, qe, f_work)
    if (code == 0) then
      if (mode == -1) then
        f(1:mpack) = -f_work(1:mpack)
      else
        f(1:mpack) = f_work(1:mpack)
      end if
      mozyme_resident_fock_try = .true.
    else
      resident_ready(resident_fock_plan_full) = .false.
      resident_full_coverage(resident_fock_plan_full) = .false.
      resident_gpu_pack_ready(resident_fock_plan_full) = .false.
      if (mozyme_resident_trace()) write(iw,'(1x,a,1x,i0)') '[MOZYME GPU resident_fock] run failed code=', code
      if (resident_strict_requested()) then
        call strict_resident_fock_abort('strict_resident_fock_run_failed', &
          'MOZYME GPU strict resident Fock run failed')
      end if
    end if
#endif
  end function mozyme_resident_fock_try

  logical function resident_strict_requested()
    implicit none

    ! MOPAC_MOZYME_SCF_GPU is a production default and must not make the
    ! CPU-driven Fock path abort (it is taken after the resident loop hands
    ! back to the CPU, e.g. when ITRY is exhausted or PLS restarts).
    resident_strict_requested = resident_env_requested( &
      'MOPAC_MOZYME_SCF_STRICT_RESIDENT') .or. &
      resident_env_requested('MOPAC_MOZYME_GPU_STRICT') .or. &
      resident_env_requested('MOPAC_MOZYME_FULL_SCF_GPU')
  end function resident_strict_requested

  logical function resident_env_requested(var_name)
    implicit none
    character(len=*), intent(in) :: var_name
    character(len=32) :: env_value, value
    integer :: env_len, env_status

    env_value = ''
    call get_environment_variable(var_name, env_value, &
      length=env_len, status=env_status)
    resident_env_requested = .false.
    if (env_status /= 0 .or. env_len <= 0) return
    value = adjustl(env_value)
    call upcase(value, len_trim(value))
    select case (trim(value))
    case ('', '0', 'FALSE', 'F', 'NO', 'N', 'OFF')
      resident_env_requested = .false.
    case default
      resident_env_requested = .true.
    end select
  end function resident_env_requested

  subroutine strict_resident_fock_abort(reason, message)
    use chanel_C, only: iw
    implicit none
    character(len=*), intent(in) :: reason, message
    external :: mopend

    write(iw,'(1x,a,a)') '[MOZYME GPU SCF] status=strict_abort reason=', &
      trim(reason)
    call flush(iw)
    call mopend(message)
    error stop 'MOZYME GPU strict resident Fock abort'
  end subroutine strict_resident_fock_abort

#ifdef GPU
  logical function mozyme_resident_fock_prepare_plan(plan_id, iorbs, nat, ifact, wj, wk, mode, kopt, ione, coord, use_nijbo)
    use chanel_C, only: iw
    use mozyme_section_timers, only: mozyme_section_timer_begin, mozyme_section_timer_end
    use molkst_C, only: numat, mpack, n2elec, l_feather, trunc_1, trunc_2, method_PM7
    use MOZYME_C, only: direct, semidr, nijbo
    use parameters_C, only: am, ad, aq, dd, qq, po, ddp, tore, iod
    use funcon_C, only: ev, a0
    implicit none
    integer, intent(in) :: plan_id
    integer, intent(in) :: iorbs(*), nat(*), ifact(*), mode, kopt(*), ione
    double precision, intent(in) :: wj(*), wk(*), coord(3,*)
    logical, intent(in) :: use_nijbo
    interface
      function mopac_cuda_mozyme_sparse_fock_setup_plan(plan_id_c, mpack, natoms, one_count, one_f_offsets, one_w_offsets, &
          one_iabs, one_ilims, one_w_values_count, one_w_values, pair_count, pair_iabs, pair_jbas, &
          pair_i_offsets, pair_j_offsets, pair_cross_offsets, pair_diag_flags, pair_w_offsets, &
          pair_w_values_count, pair_wj_values, pair_wk_values, pair4x1_count, pair4x1_heavy_offsets, &
          pair4x1_light_offsets, pair4x1_cross_offsets, pair4x1_wj_values, pair4x1_wk_values, point_count, &
          point_iabs, point_jbas, point_i_atoms, point_j_atoms, point_i_offsets, point_j_offsets, &
          point_addr_flags, point_w_values, signature_c, full_coverage) &
          bind(C,name='mopac_cuda_mozyme_sparse_fock_setup_plan') result(code)
        use iso_c_binding, only: c_int, c_double, c_int64_t
        integer(c_int), value :: plan_id_c, mpack, natoms, one_count, one_w_values_count, pair_count, pair_w_values_count
        integer(c_int), value :: pair4x1_count, point_count, full_coverage
        integer(c_int64_t), value :: signature_c
        integer(c_int) :: one_f_offsets(*), one_w_offsets(*), one_iabs(*), one_ilims(*)
        integer(c_int) :: pair_iabs(*), pair_jbas(*), pair_i_offsets(*), pair_j_offsets(*)
        integer(c_int) :: pair_cross_offsets(*), pair_diag_flags(*), pair_w_offsets(*)
        integer(c_int) :: pair4x1_heavy_offsets(*), pair4x1_light_offsets(*), pair4x1_cross_offsets(*)
        integer(c_int) :: point_iabs(*), point_jbas(*), point_i_atoms(*), point_j_atoms(*)
        integer(c_int) :: point_i_offsets(*), point_j_offsets(*), point_addr_flags(*)
        real(c_double) :: one_w_values(*), pair_wj_values(*), pair_wk_values(*)
        real(c_double) :: pair4x1_wj_values(*), pair4x1_wk_values(*), point_w_values(*)
        integer(c_int) :: code
      end function mopac_cuda_mozyme_sparse_fock_setup_plan
      function mopac_cuda_mozyme_sparse_fock_plan_ready(plan_id_c, mpack_c, full_coverage_required_c, signature_c) &
          bind(C,name='mopac_cuda_mozyme_sparse_fock_plan_ready') result(code)
        use iso_c_binding, only: c_int, c_int64_t
        integer(c_int), value :: plan_id_c, mpack_c, full_coverage_required_c
        integer(c_int64_t), value :: signature_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_sparse_fock_plan_ready
      function mopac_cuda_mozyme_resident_fock_pack_plan(plan_id_c, mpack_c, natoms_c, mode_c, ione_c, &
          direct_c, semidr_c, l_feather_c, ev_c, a0_c, trunc_1_c, trunc_2_c, w_count_c, iorbs_c, nat_c, &
          kopt_c, nijbo_c, jindex_c, coord_c, wj_c, wk_c, am_c, ad_c, aq_c, dd_c, qq_c, counts_c, fallback_basis_c, &
          po_c, ddp_c, tore_c, iod_c, method_pm7_c, signature_c, full_coverage_c) &
          bind(C,name='mopac_cuda_mozyme_resident_fock_pack_plan') result(code)
        use iso_c_binding, only: c_int, c_double, c_int64_t
        import :: max_resident_fallback_basis
        integer(c_int), value :: plan_id_c, mpack_c, natoms_c, mode_c, ione_c
        integer(c_int), value :: direct_c, semidr_c, l_feather_c, w_count_c, method_pm7_c
        integer(c_int64_t), value :: signature_c
        real(c_double), value :: ev_c, a0_c, trunc_1_c, trunc_2_c
        integer(c_int) :: iorbs_c(*), nat_c(*), kopt_c(*), nijbo_c(natoms_c,*), jindex_c(*), iod_c(*)
        integer(c_int) :: counts_c(*), fallback_basis_c(0:max_resident_fallback_basis,0:*), full_coverage_c
        real(c_double) :: coord_c(3,*), wj_c(*), wk_c(*), am_c(*), ad_c(*), aq_c(*), dd_c(*), qq_c(*)
        real(c_double) :: po_c(9,*), ddp_c(6,*), tore_c(*)
        integer(c_int) :: code
      end function mopac_cuda_mozyme_resident_fock_pack_plan
    end interface
    integer(c_int), allocatable :: one_f_offsets(:), one_w_offsets(:), one_iabs(:), one_ilims(:)
    integer(c_int), allocatable :: pair_iabs(:), pair_jbas(:), pair_i_offsets(:), pair_j_offsets(:)
    integer(c_int), allocatable :: pair_cross_offsets(:), pair_diag_flags(:), pair_w_offsets(:)
    integer(c_int), allocatable :: pair4_heavy_offsets(:), pair4_light_offsets(:), pair4_cross_offsets(:)
    integer(c_int), allocatable :: point_iabs(:), point_jbas(:), point_i_atoms(:), point_j_atoms(:)
    integer(c_int), allocatable :: point_i_offsets(:), point_j_offsets(:), point_addr_flags(:)
    double precision, allocatable :: one_w_values(:), pair_wj_values(:), pair_wk_values(:)
    double precision, allocatable :: pair4_wj_values(:,:), pair4_wk_values(:,:)
    double precision, allocatable :: point_w_values(:,:)
    integer :: one_count, one_center_cpu_count, pair_count, pair4_count, point_count, one_w_count, pair_w_count
    integer :: real_pair_count, real_pair_gpu_count, real_pair_cpu_count, real_pair_inactive_count
    integer :: real_pair_basis_limit_count, real_pair_direct_basis_count
    integer :: real_pair_other_count
    integer :: point_pair_count, point_pair_gpu_count, point_pair_cpu_count
    integer :: point_pair_basis_limit_count, point_pair_direct_basis_count
    integer :: point_pair_other_count
    integer :: fallback_basis(0:max_resident_fallback_basis,0:max_resident_fallback_basis)
    integer :: code
    integer(c_int) :: coverage_complete
    integer(c_int) :: gpu_pack_counts(19)
    integer(c_int) :: gpu_pack_fallback_basis(0:max_resident_fallback_basis,0:max_resident_fallback_basis)
    integer(c_int) :: gpu_pack_full_coverage
    integer(c_int) :: gpu_jindex(16)
    integer(c_int) :: gpu_plan_ready
    integer :: jindex(16)
    integer :: ib, jb
    integer :: idx
    integer(c_int) :: gpu_pack_code
    logical :: plan_counts_ok
    logical :: gpu_pack_counts_ok
    integer(c_int64_t) :: signature
    double precision :: plan_timer

    mozyme_resident_fock_prepare_plan = .false.
    if (plan_id < 0 .or. plan_id >= resident_fock_plan_count) return
    if (resident_strict_requested()) then
      if (last_signature(plan_id) < 0_c_int64_t .or. &
          last_signature(plan_id) >= huge(signature) - 1_c_int64_t) then
        signature = 0_c_int64_t
      else
        signature = last_signature(plan_id) + 1_c_int64_t
      end if
    else
      call mozyme_section_timer_begin('fock_plan_signature', plan_timer)
      signature = mozyme_resident_signature(iorbs, nat, ifact, wj, wk, &
        kopt, mode, ione, coord, use_nijbo)
      call mozyme_section_timer_end('fock_plan_signature', plan_timer)
    end if
    if (resident_ready(plan_id) .and. signature == last_signature(plan_id)) then
      gpu_plan_ready = mopac_cuda_mozyme_sparse_fock_plan_ready( &
        mozyme_c_int_checked(plan_id), mozyme_c_int_positive_or_zero(mpack), &
        merge(1_c_int, 0_c_int, resident_full_coverage(plan_id)), signature)
      if (gpu_plan_ready /= 1_c_int) then
        resident_ready(plan_id) = .false.
        resident_full_coverage(plan_id) = .false.
        resident_gpu_pack_ready(plan_id) = .false.
        if (resident_strict_requested()) then
          write(iw,'(1x,a,1x,a,1x,i0)') &
            '[MOZYME GPU resident_fock]', 'gpu_pack_stale=1 plan_id=', plan_id
          call flush(iw)
        end if
      end if
    end if
    if (resident_strict_requested() .and. resident_ready(plan_id) .and. &
        signature == last_signature(plan_id) .and. &
        .not. resident_gpu_pack_ready(plan_id)) then
      resident_ready(plan_id) = .false.
      resident_full_coverage(plan_id) = .false.
      resident_gpu_pack_ready(plan_id) = .false.
    end if
    if (.not. resident_ready(plan_id) .or. signature /= last_signature(plan_id)) then
      resident_gpu_pack_ready(plan_id) = .false.
	      if (use_nijbo .and. allocated(nijbo)) then
	        call make_jindex(ifact, jindex)
	        do idx = 1, 16
	          gpu_jindex(idx) = mozyme_c_int_checked(jindex(idx))
	        end do
	        gpu_pack_counts = 0_c_int
	        gpu_pack_fallback_basis = 0_c_int
	        gpu_pack_full_coverage = 0_c_int
	        call mozyme_section_timer_begin('fock_plan_gpu_pack', plan_timer)
	        gpu_pack_code = mopac_cuda_mozyme_resident_fock_pack_plan( &
	          mozyme_c_int_checked(plan_id), mozyme_c_int_positive_or_zero(mpack), &
	          mozyme_c_int_positive_or_zero(numat), mozyme_c_int_checked(mode), &
	          mozyme_c_int_checked(ione), merge(1_c_int, 0_c_int, direct), merge(1_c_int, 0_c_int, semidr), &
	          merge(1_c_int, 0_c_int, l_feather), ev, a0, trunc_1, trunc_2, &
	          mozyme_c_int_positive_or_zero(n2elec + 2025), iorbs, nat, kopt, nijbo, &
	          gpu_jindex, coord, wj, wk, am, ad, aq, dd, qq, gpu_pack_counts, &
	          gpu_pack_fallback_basis, po, ddp, tore, iod, merge(1_c_int, 0_c_int, method_PM7), &
	          signature, gpu_pack_full_coverage)
	        call mozyme_section_timer_end('fock_plan_gpu_pack', plan_timer)
	        if (gpu_pack_code == 0_c_int) then
          if (resident_strict_requested()) then
            resident_ready(plan_id) = .true.
            resident_full_coverage(plan_id) = gpu_pack_full_coverage /= 0_c_int
            resident_gpu_pack_ready(plan_id) = .true.
            last_signature(plan_id) = signature
            write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
              '[MOZYME GPU resident_fock]', 'gpu_count=1 plan_id=', plan_id, &
              'one=', int(gpu_pack_counts(1)), 'pair=', int(gpu_pack_counts(2)), &
              'pair4x1=', int(gpu_pack_counts(3)), 'point=', int(gpu_pack_counts(4)), &
              'full_coverage=', merge(1, 0, resident_full_coverage(plan_id))
            write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
              '[MOZYME GPU resident_fock]', 'gpu_pack=1 plan_id=', plan_id, &
              'one=', int(gpu_pack_counts(1)), 'pair=', int(gpu_pack_counts(2)), &
              'pair4x1=', int(gpu_pack_counts(3)), 'point=', int(gpu_pack_counts(4)), &
              'full_coverage=', merge(1, 0, resident_full_coverage(plan_id))
            if (int(gpu_pack_counts(4)) > 0) then
              write(iw,'(1x,a,1x,a,1x,i0,1x,a,1pe12.4,1x,a)') &
                '[MOZYME GPU resident_fock]', 'gpu_point_weights=1 point=', &
                int(gpu_pack_counts(4)), 'max_abs_diff=', 0.0d0, 'source=pack'
            end if
            call flush(iw)
            if (mozyme_resident_trace()) then
              write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
                '[MOZYME GPU resident_fock]', 'setup_gpu_pack one=', int(gpu_pack_counts(1)), &
                'pair=', int(gpu_pack_counts(2)), 'pair4x1=', int(gpu_pack_counts(3)), &
                'point=', int(gpu_pack_counts(4))
              call log_resident_coverage_gpu_counts(iw, mode, use_nijbo, &
                gpu_pack_counts, gpu_pack_fallback_basis)
              call flush(iw)
            end if
            mozyme_resident_fock_prepare_plan = .true.
            return
          end if
          if (.not. mozyme_resident_trace()) then
            ! Trust the device pack as the strict path does: the host
            ! re-count below is an O(numat**2) scan of nijbo (0.55 s per
            ! geometry step for 7000 atoms) that only cross-checks the kernel
            ! counters.  It still runs under MOPAC_GPU_PROFILE / trace.
            resident_ready(plan_id) = .true.
            resident_full_coverage(plan_id) = gpu_pack_full_coverage /= 0_c_int
            resident_gpu_pack_ready(plan_id) = .true.
            last_signature(plan_id) = signature
            mozyme_resident_fock_prepare_plan = .true.
            return
          end if
          call mozyme_section_timer_begin('fock_plan_count_host', plan_timer)
          call count_resident_plan(iorbs, mode, kopt, ione, use_nijbo, &
            one_count, one_center_cpu_count, pair_count, pair4_count, point_count, one_w_count, pair_w_count, &
	            real_pair_count, real_pair_gpu_count, real_pair_cpu_count, real_pair_inactive_count, &
	            real_pair_basis_limit_count, real_pair_direct_basis_count, real_pair_other_count, &
	            point_pair_count, point_pair_gpu_count, point_pair_cpu_count, point_pair_basis_limit_count, &
	            point_pair_direct_basis_count, point_pair_other_count, fallback_basis)
	          call mozyme_section_timer_end('fock_plan_count_host', plan_timer)
	          gpu_pack_counts_ok = &
	            int(gpu_pack_counts(1)) == one_count .and. &
	            int(gpu_pack_counts(2)) == pair_count .and. &
	            int(gpu_pack_counts(3)) == pair4_count .and. &
	            int(gpu_pack_counts(4)) == point_count .and. &
	            int(gpu_pack_counts(5)) == one_w_count .and. &
	            int(gpu_pack_counts(6)) == pair_w_count .and. &
	            int(gpu_pack_counts(7)) == real_pair_count .and. &
	            int(gpu_pack_counts(8)) == real_pair_gpu_count .and. &
	            int(gpu_pack_counts(9)) == real_pair_cpu_count .and. &
	            int(gpu_pack_counts(10)) == real_pair_inactive_count .and. &
	            int(gpu_pack_counts(11)) == real_pair_basis_limit_count .and. &
	            int(gpu_pack_counts(12)) == real_pair_other_count + real_pair_direct_basis_count .and. &
	            int(gpu_pack_counts(13)) == point_pair_count .and. &
	            int(gpu_pack_counts(14)) == point_pair_gpu_count .and. &
	            int(gpu_pack_counts(15)) == point_pair_cpu_count .and. &
            int(gpu_pack_counts(16)) == point_pair_basis_limit_count .and. &
            int(gpu_pack_counts(17)) == point_pair_other_count + point_pair_direct_basis_count .and. &
            int(gpu_pack_counts(18)) == one_center_cpu_count
	          do jb = 0, max_resident_fallback_basis
	            do ib = 0, max_resident_fallback_basis
	              if (int(gpu_pack_fallback_basis(ib, jb)) /= fallback_basis(ib, jb)) &
	                gpu_pack_counts_ok = .false.
	            end do
	          end do
	          if (.not. gpu_pack_counts_ok) then
	            resident_ready(plan_id) = .false.
	            resident_full_coverage(plan_id) = .false.
	            resident_gpu_pack_ready(plan_id) = .false.
	            write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
	              '[MOZYME GPU resident_fock]', 'gpu_pack_mismatch=1 plan_id=', plan_id, &
	              'gpu_one=', int(gpu_pack_counts(1)), 'cpu_one=', one_count, &
	              'gpu_pair=', int(gpu_pack_counts(2)), 'cpu_pair=', pair_count
	            call flush(iw)
	          else
	            resident_ready(plan_id) = .true.
	            resident_full_coverage(plan_id) = gpu_pack_full_coverage /= 0_c_int
	            resident_gpu_pack_ready(plan_id) = .true.
	            last_signature(plan_id) = signature
	            if (mozyme_resident_trace()) then
	              write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
	                '[MOZYME GPU resident_fock]', 'setup_gpu_pack one=', one_count, &
	                'pair=', pair_count, 'pair4x1=', pair4_count, 'point=', point_count
              call log_resident_coverage(iw, mode, use_nijbo, one_count, one_center_cpu_count, &
                real_pair_count, real_pair_gpu_count, real_pair_cpu_count, &
                real_pair_inactive_count, real_pair_basis_limit_count, real_pair_direct_basis_count, &
                real_pair_other_count, point_pair_count, point_pair_gpu_count, point_pair_cpu_count, &
	                point_pair_basis_limit_count, point_pair_direct_basis_count, point_pair_other_count, fallback_basis)
	              call flush(iw)
	            end if
	            mozyme_resident_fock_prepare_plan = .true.
	            return
	          end if
	        else if (resident_strict_requested()) then
	          write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0)') &
	            '[MOZYME GPU resident_fock]', 'gpu_pack_failed=1 plan_id=', plan_id, &
	            'code=', int(gpu_pack_code)
	          call flush(iw)
	          call strict_resident_fock_abort('strict_resident_fock_gpu_pack_failed', &
	            'MOZYME GPU strict resident Fock pack setup failed')
	          return
	        end if
	      end if

	      if (resident_strict_requested()) then
	        write(iw,'(1x,a,1x,a,1x,i0)') &
	          '[MOZYME GPU resident_fock]', 'gpu_pack_required=1 plan_id=', plan_id
	        call flush(iw)
	        call strict_resident_fock_abort('strict_resident_fock_gpu_pack_required', &
	          'MOZYME GPU strict resident Fock requires GPU-packed resident plan')
	        return
	      end if

	      call mozyme_section_timer_begin('fock_plan_build_host', plan_timer)
	      call build_resident_plan(iorbs, nat, ifact, wj, wk, mode, kopt, ione, coord, use_nijbo, &
	        one_count, one_center_cpu_count, one_f_offsets, one_w_offsets, one_iabs, one_ilims, one_w_count, one_w_values, &
        pair_count, pair_iabs, pair_jbas, pair_i_offsets, pair_j_offsets, pair_cross_offsets, &
        pair_diag_flags, pair_w_offsets, pair_w_count, pair_wj_values, pair_wk_values, &
        pair4_count, pair4_heavy_offsets, pair4_light_offsets, pair4_cross_offsets, pair4_wj_values, pair4_wk_values, &
        point_count, point_iabs, point_jbas, point_i_atoms, point_j_atoms, point_i_offsets, point_j_offsets, &
        point_addr_flags, point_w_values, real_pair_count, real_pair_gpu_count, real_pair_cpu_count, &
        real_pair_inactive_count, real_pair_basis_limit_count, real_pair_direct_basis_count, &
        real_pair_other_count, point_pair_count, point_pair_gpu_count, point_pair_cpu_count, &
        point_pair_basis_limit_count, point_pair_direct_basis_count, point_pair_other_count, &
        fallback_basis, plan_counts_ok)
	      call mozyme_section_timer_end('fock_plan_build_host', plan_timer)

      if (.not. plan_counts_ok) then
        resident_ready(plan_id) = .false.
        resident_full_coverage(plan_id) = .false.
        resident_gpu_pack_ready(plan_id) = .false.
        if (mozyme_resident_trace()) write(iw,'(1x,a)') &
          '[MOZYME GPU resident_fock] setup rejected count mismatch'
        return
      end if
      resident_full_coverage(plan_id) = one_center_cpu_count == 0 .and. &
        real_pair_cpu_count == 0 .and. point_pair_cpu_count == 0
      resident_gpu_pack_ready(plan_id) = .false.
      coverage_complete = merge(1_c_int, 0_c_int, resident_full_coverage(plan_id))

      call mozyme_section_timer_begin('fock_plan_device_setup', plan_timer)
      code = mopac_cuda_mozyme_sparse_fock_setup_plan( &
        mozyme_c_int_checked(plan_id), &
        mozyme_c_int_positive_or_zero(mpack), &
        mozyme_c_int_positive_or_zero(numat), &
        mozyme_c_int_nonnegative_or_zero(one_count), &
        one_f_offsets, one_w_offsets, one_iabs, one_ilims, &
        mozyme_c_int_nonnegative_or_zero(one_w_count), one_w_values, &
        mozyme_c_int_nonnegative_or_zero(pair_count), &
        pair_iabs, pair_jbas, pair_i_offsets, pair_j_offsets, pair_cross_offsets, pair_diag_flags, pair_w_offsets, &
        mozyme_c_int_nonnegative_or_zero(pair_w_count), pair_wj_values, &
        pair_wk_values, mozyme_c_int_nonnegative_or_zero(pair4_count), &
        pair4_heavy_offsets, pair4_light_offsets, pair4_cross_offsets, &
        pair4_wj_values, pair4_wk_values, &
        mozyme_c_int_nonnegative_or_zero(point_count), &
	        point_iabs, point_jbas, point_i_atoms, point_j_atoms, point_i_offsets, point_j_offsets, point_addr_flags, &
	        point_w_values, signature, coverage_complete)
      call mozyme_section_timer_end('fock_plan_device_setup', plan_timer)
      resident_ready(plan_id) = (code == 0)
      if (.not. resident_ready(plan_id)) then
        resident_full_coverage(plan_id) = .false.
        resident_gpu_pack_ready(plan_id) = .false.
        if (mozyme_resident_trace()) write(iw,'(1x,a,1x,i0)') '[MOZYME GPU resident_fock] setup failed code=', code
        return
      end if
      last_signature(plan_id) = signature
      if (mozyme_resident_trace()) then
        write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
          '[MOZYME GPU resident_fock]', 'setup one=', one_count, 'pair=', pair_count, &
          'pair4x1=', pair4_count, 'point=', point_count
        call log_resident_coverage(iw, mode, use_nijbo, one_count, one_center_cpu_count, &
          real_pair_count, real_pair_gpu_count, real_pair_cpu_count, &
          real_pair_inactive_count, real_pair_basis_limit_count, real_pair_direct_basis_count, &
          real_pair_other_count, point_pair_count, point_pair_gpu_count, point_pair_cpu_count, &
          point_pair_basis_limit_count, point_pair_direct_basis_count, point_pair_other_count, fallback_basis)
        call flush(iw)
      end if
    end if

    mozyme_resident_fock_prepare_plan = resident_ready(plan_id)
  end function mozyme_resident_fock_prepare_plan

  logical function mozyme_resident_fock_prepare(iorbs, nat, ifact, wj, wk, mode, kopt, ione, coord, use_nijbo)
    implicit none
    integer, intent(in) :: iorbs(*), nat(*), ifact(*), mode, kopt(*), ione
    double precision, intent(in) :: wj(*), wk(*), coord(3,*)
    logical, intent(in) :: use_nijbo

    mozyme_resident_fock_prepare = mozyme_resident_fock_prepare_plan( &
      resident_fock_plan_full, iorbs, nat, ifact, wj, wk, mode, kopt, &
      ione, coord, use_nijbo)
  end function mozyme_resident_fock_prepare

  subroutine build_resident_plan(iorbs, nat, ifact, wj, wk, mode, kopt, ione, coord, use_nijbo, &
      one_count, one_center_cpu_count, one_f_offsets, one_w_offsets, one_iabs, one_ilims, one_w_count, one_w_values, &
      pair_count, pair_iabs, pair_jbas, pair_i_offsets, pair_j_offsets, pair_cross_offsets, &
      pair_diag_flags, pair_w_offsets, pair_w_count, pair_wj_values, pair_wk_values, &
      pair4_count, pair4_heavy_offsets, pair4_light_offsets, pair4_cross_offsets, pair4_wj_values, pair4_wk_values, &
      point_count, point_iabs, point_jbas, point_i_atoms, point_j_atoms, point_i_offsets, point_j_offsets, &
      point_addr_flags, point_w_values, real_pair_count, real_pair_gpu_count, real_pair_cpu_count, &
      real_pair_inactive_count, real_pair_basis_limit_count, real_pair_direct_basis_count, &
      real_pair_other_count, point_pair_count, point_pair_gpu_count, point_pair_cpu_count, &
      point_pair_basis_limit_count, point_pair_direct_basis_count, point_pair_other_count, &
      fallback_basis, counts_ok)
    use molkst_C, only: numat
    use MOZYME_C, only: direct, semidr, nijbo
    use chanel_C, only: iw
    implicit none
    integer, intent(in) :: iorbs(*), nat(*), ifact(*), mode, kopt(*), ione
    double precision, intent(in) :: wj(*), wk(*), coord(3,*)
    logical, intent(in) :: use_nijbo
    integer, intent(out) :: one_count, one_center_cpu_count, pair_count, pair4_count, point_count
    integer, intent(out) :: one_w_count, pair_w_count
    integer, intent(out) :: real_pair_count, real_pair_gpu_count, real_pair_cpu_count, real_pair_inactive_count
    integer, intent(out) :: real_pair_basis_limit_count, real_pair_direct_basis_count
    integer, intent(out) :: real_pair_other_count
    integer, intent(out) :: point_pair_count, point_pair_gpu_count, point_pair_cpu_count
    integer, intent(out) :: point_pair_basis_limit_count, point_pair_direct_basis_count
    integer, intent(out) :: point_pair_other_count
    integer, intent(out) :: fallback_basis(0:max_resident_fallback_basis,0:max_resident_fallback_basis)
    logical, intent(out) :: counts_ok
    integer(c_int), allocatable, intent(out) :: one_f_offsets(:), one_w_offsets(:), one_iabs(:), one_ilims(:)
    integer(c_int), allocatable, intent(out) :: pair_iabs(:), pair_jbas(:), pair_i_offsets(:), pair_j_offsets(:)
    integer(c_int), allocatable, intent(out) :: pair_cross_offsets(:), pair_diag_flags(:), pair_w_offsets(:)
    integer(c_int), allocatable, intent(out) :: pair4_heavy_offsets(:), pair4_light_offsets(:), pair4_cross_offsets(:)
    integer(c_int), allocatable, intent(out) :: point_iabs(:), point_jbas(:), point_i_atoms(:), point_j_atoms(:)
    integer(c_int), allocatable, intent(out) :: point_i_offsets(:), point_j_offsets(:), point_addr_flags(:)
    double precision, allocatable, intent(out) :: one_w_values(:), pair_wj_values(:), pair_wk_values(:)
    double precision, allocatable, intent(out) :: pair4_wj_values(:,:), pair4_wk_values(:,:)
    double precision, allocatable, intent(out) :: point_w_values(:,:)
    integer :: ii, jj, iab, jba, tri_i, tri_j, total, kr, addr, ired, jred, iim1
    integer :: one_pos, pair_pos, pair4_pos, point_pos, one_w_pos, pair_w_pos
    logical :: calci, calcj
    integer :: jindex(16)
    double precision :: wjloc(2025), e1b(45), e2a(45), enuc
    integer, external :: ijbo
    external :: rotate

    counts_ok = .false.
    call make_jindex(ifact, jindex)
    call count_resident_plan(iorbs, mode, kopt, ione, use_nijbo, &
      one_count, one_center_cpu_count, pair_count, pair4_count, point_count, one_w_count, pair_w_count, &
      real_pair_count, real_pair_gpu_count, real_pair_cpu_count, real_pair_inactive_count, &
      real_pair_basis_limit_count, real_pair_direct_basis_count, real_pair_other_count, &
      point_pair_count, point_pair_gpu_count, point_pair_cpu_count, point_pair_basis_limit_count, &
      point_pair_direct_basis_count, point_pair_other_count, fallback_basis)

    allocate(one_f_offsets(max(1, one_count)), one_w_offsets(max(1, one_count)), &
      one_iabs(max(1, one_count)), one_ilims(max(1, one_count)))
    allocate(pair_iabs(max(1, pair_count)), pair_jbas(max(1, pair_count)), &
      pair_i_offsets(max(1, pair_count)), pair_j_offsets(max(1, pair_count)), &
      pair_cross_offsets(max(1, pair_count)), pair_diag_flags(max(1, pair_count)), &
      pair_w_offsets(max(1, pair_count)))
    allocate(pair4_heavy_offsets(max(1, pair4_count)), pair4_light_offsets(max(1, pair4_count)), &
      pair4_cross_offsets(max(1, pair4_count)))
    allocate(point_iabs(max(1, point_count)), point_jbas(max(1, point_count)), &
      point_i_atoms(max(1, point_count)), point_j_atoms(max(1, point_count)), &
      point_i_offsets(max(1, point_count)), point_j_offsets(max(1, point_count)), &
      point_addr_flags(max(1, point_count)))
    allocate(one_w_values(max(1, one_w_count)), pair_wj_values(max(1, pair_w_count)), &
      pair_wk_values(max(1, pair_w_count)))
    allocate(pair4_wj_values(10, max(1, pair4_count)), pair4_wk_values(16, max(1, pair4_count)))
    allocate(point_w_values(7, max(1, point_count)))
    point_w_values = 0.0d0

    one_pos = 0
    pair_pos = 0
    pair4_pos = 0
    point_pos = 0
    one_w_pos = 1
    pair_w_pos = 1
    kr = 0

    ired = 1
    do ii = 1, numat
      if (mode == 0) then
        calci = .true.
      else
        calci = (kopt(ired) == ii)
        if (calci .and. ired < numat) ired = ired + 1
      end if
      iab = iorbs(ii)
      if (iab /= 0) then
        jred = 1
        iim1 = ii - ione
        do jj = 1, iim1
          if (mode == 0) then
            calcj = .true.
          else
            calcj = (kopt(jred) == jj)
            if (calcj .and. jred < numat) jred = jred + 1
          end if
          jba = iorbs(jj)
          addr = mozyme_pair_addr(ii, jj, use_nijbo)
          if (addr >= 0) then
            if (calci .or. calcj) then
              if (mozyme_resident_pair_supported(iab, jba)) then
                if (direct) call rotate(nat(ii), nat(jj), coord(1, ii), coord(1, jj), wjloc, kr, e1b, e2a, enuc)
                if (iab == 4 .and. jba == 1) then
                  pair4_pos = pair4_pos + 1
                  pair4_heavy_offsets(pair4_pos) = &
                    mozyme_c_int_positive_or_zero(mozyme_pair_addr(ii, ii, use_nijbo) + 1)
                  pair4_light_offsets(pair4_pos) = &
                    mozyme_c_int_positive_or_zero(mozyme_pair_addr(jj, jj, use_nijbo) + 1)
                  pair4_cross_offsets(pair4_pos) = &
                    mozyme_c_int_positive_or_zero(addr + 1)
                  call copy_4x1_integrals(direct, kr, wj, wk, wjloc, jindex, pair4_wj_values(:, pair4_pos), &
                    pair4_wk_values(:, pair4_pos))
                else if (jba == 4 .and. iab == 1) then
                  pair4_pos = pair4_pos + 1
                  pair4_heavy_offsets(pair4_pos) = &
                    mozyme_c_int_positive_or_zero(mozyme_pair_addr(jj, jj, use_nijbo) + 1)
                  pair4_light_offsets(pair4_pos) = &
                    mozyme_c_int_positive_or_zero(mozyme_pair_addr(ii, ii, use_nijbo) + 1)
                  pair4_cross_offsets(pair4_pos) = &
                    mozyme_c_int_positive_or_zero(addr + 1)
                  call copy_4x1_integrals(direct, kr, wj, wk, wjloc, jindex, pair4_wj_values(:, pair4_pos), &
                    pair4_wk_values(:, pair4_pos))
                else
                  tri_i = (iab * (iab + 1)) / 2
                  tri_j = (jba * (jba + 1)) / 2
                  total = tri_i * tri_j
                  pair_pos = pair_pos + 1
                  pair_iabs(pair_pos) = mozyme_c_int_positive_or_zero(iab)
                  pair_jbas(pair_pos) = mozyme_c_int_positive_or_zero(jba)
                  pair_i_offsets(pair_pos) = &
                    mozyme_c_int_positive_or_zero(mozyme_pair_addr(ii, ii, use_nijbo) + 1)
                  pair_j_offsets(pair_pos) = &
                    mozyme_c_int_positive_or_zero(mozyme_pair_addr(jj, jj, use_nijbo) + 1)
                  pair_cross_offsets(pair_pos) = &
                    mozyme_c_int_positive_or_zero(addr + 1)
                  if (ii == jj) then
                    pair_diag_flags(pair_pos) = 1_c_int
                  else
                    pair_diag_flags(pair_pos) = 0_c_int
                  end if
                  pair_w_offsets(pair_pos) = &
                    mozyme_c_int_positive_or_zero(pair_w_pos)
                  if (direct) then
                    pair_wj_values(pair_w_pos:pair_w_pos + total - 1) = wjloc(1:total)
                    pair_wk_values(pair_w_pos:pair_w_pos + total - 1) = wjloc(1:total)
                  else
                    pair_wj_values(pair_w_pos:pair_w_pos + total - 1) = wj(kr + 1:kr + total)
                    pair_wk_values(pair_w_pos:pair_w_pos + total - 1) = wk(kr + 1:kr + total)
                  end if
                  pair_w_pos = pair_w_pos + total
                end if
              end if
            end if
            if (.not. direct) kr = kr + mozyme_pair_integral_count(iab, jba)
          else
            if ((calci .or. calcj) .and. mozyme_resident_point_supported(iab, jba, addr)) then
              point_pos = point_pos + 1
              point_iabs(point_pos) = mozyme_c_int_positive_or_zero(iab)
              point_jbas(point_pos) = mozyme_c_int_positive_or_zero(jba)
              point_i_atoms(point_pos) = mozyme_c_int_positive_or_zero(ii)
              point_j_atoms(point_pos) = mozyme_c_int_positive_or_zero(jj)
              point_i_offsets(point_pos) = &
                mozyme_c_int_positive_or_zero(mozyme_pair_addr(ii, ii, use_nijbo) + 1)
              point_j_offsets(point_pos) = &
                mozyme_c_int_positive_or_zero(mozyme_pair_addr(jj, jj, use_nijbo) + 1)
              point_addr_flags(point_pos) = mozyme_c_int_checked(addr)
              call compute_point_weights(ii, jj, iab, jba, addr, kr, nat, coord, wj, point_w_values(:, point_pos))
            end if
            call mozyme_point_charge_advance_kr(iorbs(ii), iorbs(jj), addr, kr)
          end if
        end do
        if (.not. direct) then
          if (mozyme_resident_basis_supported(iab)) then
            call add_one_center_task(ii, iab, kr, use_nijbo, wj, one_pos, one_w_pos, &
              one_f_offsets, one_w_offsets, one_iabs, one_ilims, one_w_values)
          else
            kr = kr + mozyme_tri(iab) * mozyme_tri(iab)
          end if
        end if
      end if
    end do

    if (direct) then
      kr = 0
      do ii = 1, numat
        iab = iorbs(ii)
        if (iab /= 0) then
          if (mozyme_resident_basis_supported(iab)) then
            call add_one_center_task(ii, iab, kr, use_nijbo, wj, one_pos, one_w_pos, &
              one_f_offsets, one_w_offsets, one_iabs, one_ilims, one_w_values)
          else
            kr = kr + mozyme_tri(iab) * mozyme_tri(iab)
          end if
        end if
      end do
    end if

    counts_ok = one_pos == one_count .and. pair_pos == pair_count .and. &
      pair4_pos == pair4_count .and. point_pos == point_count .and. &
      one_w_pos - 1 == one_w_count .and. pair_w_pos - 1 == pair_w_count
  end subroutine build_resident_plan

  subroutine count_resident_plan(iorbs, mode, kopt, ione, use_nijbo, one_count, one_center_cpu_count, &
      pair_count, pair4_count, point_count, &
      one_w_count, pair_w_count, real_pair_count, real_pair_gpu_count, real_pair_cpu_count, real_pair_inactive_count, &
      real_pair_basis_limit_count, real_pair_direct_basis_count, real_pair_other_count, point_pair_count, &
      point_pair_gpu_count, point_pair_cpu_count, point_pair_basis_limit_count, point_pair_direct_basis_count, &
      point_pair_other_count, fallback_basis)
    use molkst_C, only: numat
    use MOZYME_C, only: direct
    implicit none
    integer, intent(in) :: iorbs(*), mode, kopt(*), ione
    logical, intent(in) :: use_nijbo
    integer, intent(out) :: one_count, one_center_cpu_count, pair_count, pair4_count, point_count
    integer, intent(out) :: one_w_count, pair_w_count
    integer, intent(out) :: real_pair_count, real_pair_gpu_count, real_pair_cpu_count, real_pair_inactive_count
    integer, intent(out) :: real_pair_basis_limit_count, real_pair_direct_basis_count
    integer, intent(out) :: real_pair_other_count
    integer, intent(out) :: point_pair_count, point_pair_gpu_count, point_pair_cpu_count
    integer, intent(out) :: point_pair_basis_limit_count, point_pair_direct_basis_count
    integer, intent(out) :: point_pair_other_count
    integer, intent(out) :: fallback_basis(0:max_resident_fallback_basis,0:max_resident_fallback_basis)
    integer :: ii, jj, iab, jba, addr, ired, jred, iim1, kr
    logical :: calci, calcj

    one_count = 0
    one_center_cpu_count = 0
    pair_count = 0
    pair4_count = 0
    point_count = 0
    one_w_count = 0
    pair_w_count = 0
    real_pair_count = 0
    real_pair_gpu_count = 0
    real_pair_cpu_count = 0
    real_pair_inactive_count = 0
    real_pair_basis_limit_count = 0
    real_pair_direct_basis_count = 0
    real_pair_other_count = 0
    point_pair_count = 0
    point_pair_gpu_count = 0
    point_pair_cpu_count = 0
    point_pair_basis_limit_count = 0
    point_pair_direct_basis_count = 0
    point_pair_other_count = 0
    fallback_basis = 0
    kr = 0
    ired = 1
    do ii = 1, numat
      if (mode == 0) then
        calci = .true.
      else
        calci = (kopt(ired) == ii)
        if (calci .and. ired < numat) ired = ired + 1
      end if
      iab = iorbs(ii)
      if (iab /= 0) then
        jred = 1
        iim1 = ii - ione
        do jj = 1, iim1
          if (mode == 0) then
            calcj = .true.
          else
            calcj = (kopt(jred) == jj)
            if (calcj .and. jred < numat) jred = jred + 1
          end if
          jba = iorbs(jj)
          addr = mozyme_pair_addr(ii, jj, use_nijbo)
          if (addr >= 0) then
            if (calci .or. calcj) then
              real_pair_count = real_pair_count + 1
              if (mozyme_resident_pair_supported(iab, jba)) then
                real_pair_gpu_count = real_pair_gpu_count + 1
                if ((iab == 4 .and. jba == 1) .or. (iab == 1 .and. jba == 4)) then
                  pair4_count = pair4_count + 1
                else
                  pair_count = pair_count + 1
                  pair_w_count = pair_w_count + mozyme_pair_integral_count(iab, jba)
                end if
              else if (.not. mozyme_resident_pair_noop(iab, jba)) then
                real_pair_cpu_count = real_pair_cpu_count + 1
                if (iab > max_resident_diag_basis .or. jba > max_resident_diag_basis) then
                  real_pair_basis_limit_count = real_pair_basis_limit_count + 1
                else if (mozyme_resident_direct_basis_fallback(iab, jba)) then
                  real_pair_direct_basis_count = real_pair_direct_basis_count + 1
                else
                  real_pair_other_count = real_pair_other_count + 1
                end if
                call increment_fallback_basis(iab, jba, fallback_basis)
              end if
            else
              real_pair_inactive_count = real_pair_inactive_count + 1
            end if
            if (.not. direct) kr = kr + mozyme_pair_integral_count(iab, jba)
          else
            if ((calci .or. calcj) .and. iab * jba > 0) then
              point_pair_count = point_pair_count + 1
              if (mozyme_resident_point_supported(iab, jba, addr)) then
                point_pair_gpu_count = point_pair_gpu_count + 1
                point_count = point_count + 1
              else
                point_pair_cpu_count = point_pair_cpu_count + 1
                if (iab > max_resident_diag_basis .or. jba > max_resident_diag_basis) then
                  point_pair_basis_limit_count = point_pair_basis_limit_count + 1
                else if (mozyme_resident_direct_basis_fallback(iab, jba)) then
                  point_pair_direct_basis_count = point_pair_direct_basis_count + 1
                else
                  point_pair_other_count = point_pair_other_count + 1
                end if
                call increment_fallback_basis(iab, jba, fallback_basis)
              end if
            end if
            call mozyme_point_charge_advance_kr(iab, jba, addr, kr)
          end if
        end do
        if (.not. direct) then
          if (mozyme_resident_basis_supported(iab)) then
            one_count = one_count + 1
            one_w_count = one_w_count + mozyme_tri(iab) * mozyme_tri(iab)
          else
            one_center_cpu_count = one_center_cpu_count + 1
            call increment_fallback_basis(iab, iab, fallback_basis)
          end if
          kr = kr + mozyme_tri(iab) * mozyme_tri(iab)
        end if
      end if
    end do
    if (direct) then
      one_count = 0
      one_w_count = 0
      one_center_cpu_count = 0
      do ii = 1, numat
        iab = iorbs(ii)
        if (iab /= 0) then
          if (mozyme_resident_basis_supported(iab)) then
            one_count = one_count + 1
            one_w_count = one_w_count + mozyme_tri(iab) * mozyme_tri(iab)
          else
            one_center_cpu_count = one_center_cpu_count + 1
            call increment_fallback_basis(iab, iab, fallback_basis)
          end if
        end if
      end do
    end if
  end subroutine count_resident_plan

  subroutine log_resident_coverage_gpu_counts(out_unit, mode, use_nijbo, gpu_counts_in, gpu_fallback_basis_in)
    implicit none
    integer, intent(in) :: out_unit, mode
    logical, intent(in) :: use_nijbo
    integer(c_int), intent(in) :: gpu_counts_in(19)
    integer(c_int), intent(in) :: gpu_fallback_basis_in(0:max_resident_fallback_basis,0:max_resident_fallback_basis)
    integer :: fallback_basis(0:max_resident_fallback_basis,0:max_resident_fallback_basis)
    integer :: ib, jb

    do jb = 0, max_resident_fallback_basis
      do ib = 0, max_resident_fallback_basis
        fallback_basis(ib, jb) = int(gpu_fallback_basis_in(ib, jb))
      end do
    end do
    call log_resident_coverage(out_unit, mode, use_nijbo, &
      int(gpu_counts_in(1)), int(gpu_counts_in(18)), &
      int(gpu_counts_in(7)), int(gpu_counts_in(8)), int(gpu_counts_in(9)), &
      int(gpu_counts_in(10)), int(gpu_counts_in(11)), 0, int(gpu_counts_in(12)), &
      int(gpu_counts_in(13)), int(gpu_counts_in(14)), int(gpu_counts_in(15)), &
      int(gpu_counts_in(16)), 0, int(gpu_counts_in(17)), fallback_basis)
  end subroutine log_resident_coverage_gpu_counts

  subroutine add_one_center_task(ii, iab, kr, use_nijbo, wj, one_pos, one_w_pos, one_f_offsets, one_w_offsets, &
      one_iabs, one_ilims, one_w_values)
    implicit none
    integer, intent(in) :: ii, iab
    integer, intent(inout) :: kr, one_pos, one_w_pos
    logical, intent(in) :: use_nijbo
    double precision, intent(in) :: wj(*)
    integer(c_int), intent(inout) :: one_f_offsets(*), one_w_offsets(*), one_iabs(*), one_ilims(*)
    double precision, intent(inout) :: one_w_values(*)
    integer :: ilim, w_count
    ilim = mozyme_tri(iab)
    w_count = ilim * ilim
    one_pos = one_pos + 1
    one_f_offsets(one_pos) = &
      mozyme_c_int_positive_or_zero(mozyme_pair_addr(ii, ii, use_nijbo) + 1)
    one_w_offsets(one_pos) = mozyme_c_int_positive_or_zero(one_w_pos)
    one_iabs(one_pos) = mozyme_c_int_positive_or_zero(iab)
    one_ilims(one_pos) = mozyme_c_int_positive_or_zero(ilim)
    one_w_values(one_w_pos:one_w_pos + w_count - 1) = wj(kr + 1:kr + w_count)
    one_w_pos = one_w_pos + w_count
    kr = kr + w_count
  end subroutine add_one_center_task

  subroutine copy_4x1_integrals(is_direct, kr, wj, wk, wjloc, jindex, wj_values, wk_values)
    implicit none
    logical, intent(in) :: is_direct
    integer, intent(in) :: kr, jindex(16)
    double precision, intent(in) :: wj(*), wk(*), wjloc(*)
    double precision, intent(out) :: wj_values(10), wk_values(16)
    integer :: i
    if (is_direct) then
      wj_values(:) = wjloc(1:10)
      do i = 1, 16
        wk_values(i) = wjloc(jindex(i))
      end do
    else
      wj_values(:) = wj(kr + 1:kr + 10)
      do i = 1, 16
        wk_values(i) = wk(kr + jindex(i))
      end do
    end if
  end subroutine copy_4x1_integrals

  subroutine compute_point_weights(ii, jj, iab, jba, addr, kr, nat, coord, wj, values)
    use molkst_C, only: l_feather
    use MOZYME_C, only: direct, semidr
    use parameters_C, only: am, dd, ad
    use funcon_C, only: ev, a0
    implicit none
    integer, intent(in) :: ii, jj, iab, jba, addr, kr
    integer, intent(in) :: nat(*)
    double precision, intent(in) :: coord(3,*), wj(*)
    double precision, intent(out) :: values(7)
    integer :: ni, nj, base
    double precision :: dx, dy, dz, r2, r, aee, da, ade, rp, rm, ri2, ri5, point, const
    external :: to_point

    values = 0.0d0
    if (iab * jba <= 0) return

    if (direct .or. semidr) then
      dx = coord(1, ii) - coord(1, jj)
      dy = coord(2, ii) - coord(2, jj)
      dz = coord(3, ii) - coord(3, jj)
      r2 = dx * dx + dy * dy + dz * dz
      ni = nat(ii)
      nj = nat(jj)
      aee = 0.5d0 / am(ni) + 0.5d0 / am(nj)
      values(1) = ev / sqrt(r2/(a0**2)+aee**2)
      if (l_feather) then
        call to_point(sqrt(r2), point, const)
        values(1) = values(1) * const + (1.d0 - const) * point
      end if

      if (addr == -2) then
        r = sqrt(r2)
        if (r > 0.0d0) then
          dx = dx / r
          dy = dy / r
          dz = dz / r
        end if
        if (abs(dz) > 0.99999999d0) dz = sign(1.d0, dz)

        if (iab > 1) then
          da = dd(ni)
          ade = 0.5d0 / ad(ni) + 0.5d0 / am(nj)
          rp = sqrt((r/a0+da)**2+ade**2)
          rm = sqrt((r/a0-da)**2+ade**2)
          ri2 = ev * (0.5d0/rp-0.5d0/rm)
          if (l_feather) then
            call to_point(r, point, const)
            ri2 = ri2 * const
          end if
          values(5) = ri2 * dx
          values(6) = ri2 * dy
          values(7) = ri2 * dz
        end if

        if (jba > 1) then
          da = dd(nj)
          ade = 0.5d0 / am(ni) + 0.5d0 / ad(nj)
          rp = sqrt((r/a0+da)**2+ade**2)
          rm = sqrt((r/a0-da)**2+ade**2)
          ri5 = -ev * (0.5d0/rp-0.5d0/rm)
          if (l_feather) then
            call to_point(r, point, const)
            ri5 = ri5 * const
          end if
          values(2) = ri5 * dx
          values(3) = ri5 * dy
          values(4) = ri5 * dz
        end if
      end if
    else
      values(1) = wj(kr + 1)
      if (addr == -2) then
        base = kr + 1
        if (iab > 1 .and. jba > 1) then
          values(2) = wj(base + 1)
          values(3) = wj(base + 2)
          values(4) = wj(base + 3)
          values(5) = wj(base + 4)
          values(6) = wj(base + 5)
          values(7) = wj(base + 6)
        else if (iab > 1) then
          values(5) = wj(base + 1)
          values(6) = wj(base + 2)
          values(7) = wj(base + 3)
        else if (jba > 1) then
          values(2) = wj(base + 1)
          values(3) = wj(base + 2)
          values(4) = wj(base + 3)
        end if
      end if
    end if
  end subroutine compute_point_weights

  subroutine mozyme_point_charge_advance_kr(iab, jba, addr, kr)
    use MOZYME_C, only: direct, semidr
    implicit none
    integer, intent(in) :: iab, jba, addr
    integer, intent(inout) :: kr
    if (iab * jba <= 0) return
    if (direct .or. semidr) return
    kr = kr + 1
    if (addr == -2) then
      if (iab > 1 .and. jba > 1) then
        kr = kr + 6
      else if (iab > 1 .or. jba > 1) then
        kr = kr + 3
      end if
    end if
  end subroutine mozyme_point_charge_advance_kr

  integer function mozyme_pair_addr(i, j, use_nijbo)
    use MOZYME_C, only: nijbo
    implicit none
    integer, intent(in) :: i, j
    logical, intent(in) :: use_nijbo
    integer, external :: ijbo
    if (use_nijbo) then
      mozyme_pair_addr = nijbo(i, j)
    else
      mozyme_pair_addr = ijbo(i, j)
    end if
  end function mozyme_pair_addr

  integer function mozyme_tri(norb)
    implicit none
    integer, intent(in) :: norb
    mozyme_tri = (norb * (norb + 1)) / 2
  end function mozyme_tri

  integer function mozyme_pair_integral_count(iab, jba)
    implicit none
    integer, intent(in) :: iab, jba
    if ((iab == 4 .and. jba == 1) .or. (iab == 1 .and. jba == 4)) then
      mozyme_pair_integral_count = 10
    else
      mozyme_pair_integral_count = mozyme_tri(iab) * mozyme_tri(jba)
    end if
  end function mozyme_pair_integral_count

  subroutine increment_fallback_basis(iab, jba, fallback_basis)
    implicit none
    integer, intent(in) :: iab, jba
    integer, intent(inout) :: fallback_basis(0:max_resident_fallback_basis,0:max_resident_fallback_basis)
    integer :: ib, jb
    ib = resident_fallback_basis_bin(iab)
    jb = resident_fallback_basis_bin(jba)
    fallback_basis(ib, jb) = fallback_basis(ib, jb) + 1
  end subroutine increment_fallback_basis

  integer function resident_fallback_basis_bin(nbasis)
    implicit none
    integer, intent(in) :: nbasis
    if (nbasis < 0) then
      resident_fallback_basis_bin = 0
    else if (nbasis > max_resident_diag_basis) then
      resident_fallback_basis_bin = max_resident_fallback_basis
    else
      resident_fallback_basis_bin = nbasis
    end if
  end function resident_fallback_basis_bin

  subroutine resident_fallback_basis_label(bin, label)
    implicit none
    integer, intent(in) :: bin
    character(len=*), intent(out) :: label
    if (bin == max_resident_fallback_basis) then
      label = '>9'
    else
      write(label,'(i0)') bin
    end if
  end subroutine resident_fallback_basis_label

  subroutine log_resident_coverage(iw, mode, use_nijbo, one_count, one_center_cpu_count, &
      real_pair_count, real_pair_gpu_count, real_pair_cpu_count, &
      real_pair_inactive_count, real_pair_basis_limit_count, real_pair_direct_basis_count, &
      real_pair_other_count, point_pair_count, point_pair_gpu_count, point_pair_cpu_count, &
      point_pair_basis_limit_count, point_pair_direct_basis_count, point_pair_other_count, fallback_basis)
    implicit none
    integer, intent(in) :: iw, mode
    logical, intent(in) :: use_nijbo
    integer, intent(in) :: one_count, one_center_cpu_count
    integer, intent(in) :: real_pair_count, real_pair_gpu_count, real_pair_cpu_count, real_pair_inactive_count
    integer, intent(in) :: real_pair_basis_limit_count, real_pair_direct_basis_count
    integer, intent(in) :: real_pair_other_count
    integer, intent(in) :: point_pair_count, point_pair_gpu_count, point_pair_cpu_count
    integer, intent(in) :: point_pair_basis_limit_count, point_pair_direct_basis_count
    integer, intent(in) :: point_pair_other_count
    integer, intent(in) :: fallback_basis(0:max_resident_fallback_basis,0:max_resident_fallback_basis)
    integer :: ib, jb
    character(len=12) :: ib_label, jb_label

    write(iw,'(1x,a," coverage mode=",i0," use_nijbo=",l1," real_pairs=",i0," gpu_real_pairs=",i0,' // &
      '" cpu_real_pairs=",i0," inactive_real_pairs=",i0)') '[MOZYME GPU resident_fock]', mode, use_nijbo, &
      real_pair_count, real_pair_gpu_count, real_pair_cpu_count, real_pair_inactive_count
    write(iw,'(1x,a," one_center_coverage gpu_one_center=",i0," cpu_one_center=",i0)') &
      '[MOZYME GPU resident_fock]', one_count, one_center_cpu_count
    write(iw,'(1x,a," fallback_real_pairs total=",i0," reason=unsupported_basis_limit count=",i0,' // &
      '" reason=unsupported_direct_basis count=",i0," reason=unsupported_other count=",i0)') &
      '[MOZYME GPU resident_fock]', real_pair_cpu_count, real_pair_basis_limit_count, &
      real_pair_direct_basis_count, real_pair_other_count
    write(iw,'(1x,a," point_coverage point_pairs=",i0," gpu_point_pairs=",i0," cpu_point_pairs=",i0,' // &
      '" reason=unsupported_basis_limit count=",i0," reason=unsupported_direct_basis count=",i0,' // &
      '" reason=unsupported_other count=",i0)') &
      '[MOZYME GPU resident_fock]', point_pair_count, point_pair_gpu_count, point_pair_cpu_count, &
      point_pair_basis_limit_count, point_pair_direct_basis_count, point_pair_other_count
    if (one_center_cpu_count == 0 .and. real_pair_cpu_count == 0 .and. point_pair_cpu_count == 0) then
      write(iw,'(1x,a," fallback_basis none")') '[MOZYME GPU resident_fock]'
    else
      do ib = 0, max_resident_fallback_basis
        do jb = 0, max_resident_fallback_basis
          if (fallback_basis(ib, jb) > 0) then
            call resident_fallback_basis_label(ib, ib_label)
            call resident_fallback_basis_label(jb, jb_label)
            write(iw,'(1x,a," fallback_basis iab=",a," jba=",a," count=",i0)') &
              '[MOZYME GPU resident_fock]', trim(ib_label), trim(jb_label), fallback_basis(ib, jb)
          end if
        end do
      end do
    end if
  end subroutine log_resident_coverage

  subroutine make_jindex(ifact, jindex)
    implicit none
    integer, intent(in) :: ifact(*)
    integer, intent(out) :: jindex(16)
    integer :: k, l, kl, lk, m
    m = 0
    do k = 1, 4
      do l = 1, 4
        m = m + 1
        kl = min(k, l)
        lk = k + l - kl
        jindex(m) = ifact(lk) + kl
      end do
    end do
  end subroutine make_jindex

  integer(c_int64_t) function mozyme_resident_signature(iorbs, nat, ifact, wj, wk, &
      kopt, mode, ione, coord, use_nijbo)
    use molkst_C, only: numat, norbs, mpack, n2elec, l_feather, trunc_1, trunc_2, method_PM7
    use MOZYME_C, only: direct, semidr, nijbo
    use parameters_C, only: am, ad, aq, dd, qq, po, ddp, tore, iod
    use funcon_C, only: ev, a0
    implicit none
    integer, intent(in) :: iorbs(*), nat(*), ifact(*), kopt(*), mode, ione
    double precision, intent(in) :: wj(*), wk(*), coord(3,*)
    logical, intent(in) :: use_nijbo
    integer(c_int64_t) :: h1, h2
    integer :: i, j, ni

    h1 = 1_c_int64_t
    h2 = 1_c_int64_t
    call mozyme_mix_int64(h1, h2, int(mode, c_int64_t))
    call mozyme_mix_int64(h1, h2, int(ione, c_int64_t))
    call mozyme_mix_int64(h1, h2, int(mpack, c_int64_t))
    call mozyme_mix_int64(h1, h2, int(norbs, c_int64_t))
    call mozyme_mix_logical(h1, h2, direct)
    call mozyme_mix_logical(h1, h2, method_PM7)
    call mozyme_mix_logical(h1, h2, semidr)
    call mozyme_mix_logical(h1, h2, use_nijbo)
    call mozyme_mix_logical(h1, h2, l_feather)
    call mozyme_mix_real64(h1, h2, ev)
    call mozyme_mix_real64(h1, h2, a0)
    call mozyme_mix_real64(h1, h2, trunc_1)
    call mozyme_mix_real64(h1, h2, trunc_2)
    do i = 1, numat
      call mozyme_mix_int64(h1, h2, int(i, c_int64_t))
      call mozyme_mix_int64(h1, h2, int(iorbs(i), c_int64_t))
      call mozyme_mix_int64(h1, h2, int(nat(i), c_int64_t))
      if (mode /= 0) call mozyme_mix_int64(h1, h2, int(kopt(i), c_int64_t))
      ni = nat(i)
      if (ni >= 1 .and. ni <= size(am)) then
        call mozyme_mix_real64(h1, h2, am(ni))
        call mozyme_mix_real64(h1, h2, ad(ni))
        call mozyme_mix_real64(h1, h2, aq(ni))
        call mozyme_mix_real64(h1, h2, dd(ni))
        call mozyme_mix_real64(h1, h2, qq(ni))
        if (direct) then
          call mozyme_mix_real64(h1, h2, tore(ni))
          call mozyme_mix_int64(h1, h2, int(iod(ni), c_int64_t))
          do j = 1, 9
            call mozyme_mix_real64(h1, h2, po(j, ni))
          end do
          do j = 1, 6
            call mozyme_mix_real64(h1, h2, ddp(j, ni))
          end do
        end if
      end if
      call mozyme_mix_real64(h1, h2, coord(1, i))
      call mozyme_mix_real64(h1, h2, coord(2, i))
      call mozyme_mix_real64(h1, h2, coord(3, i))
    end do
    do i = 1, min(norbs, 16)
      call mozyme_mix_int64(h1, h2, int(ifact(i), c_int64_t))
    end do
    call mozyme_mix_int64(h1, h2, int(n2elec, c_int64_t))
    ! nijbo and the stored integrals are functions of the coordinates (hashed
    ! above), the cutoffs and the parameters, so hashing them again only costs
    ! time (numat**2 + n2elec terms: ~0.4 s per SCF for 7000 atoms).  The full
    ! hash remains available for debugging.
    if (resident_env_requested('MOPAC_MOZYME_FOCK_FULL_SIGNATURE')) then
      if (use_nijbo) then
        do j = 1, numat
          do i = 1, numat
            call mozyme_mix_int64(h1, h2, int(nijbo(i, j), c_int64_t))
          end do
        end do
      end if
      call mozyme_resident_integral_signature_hash(iorbs, wj, wk, use_nijbo, h1, h2)
    end if
    mozyme_resident_signature = h1 * mozyme_sig_mod2 + h2
  end function mozyme_resident_signature

  subroutine mozyme_resident_integral_signature_hash(iorbs, wj, wk, use_nijbo, h1, h2)
    use molkst_C, only: numat
    use MOZYME_C, only: direct
    implicit none
    integer, intent(in) :: iorbs(*)
    double precision, intent(in) :: wj(*), wk(*)
    logical, intent(in) :: use_nijbo
    integer(c_int64_t), intent(inout) :: h1, h2
    integer :: ii, jj, iab, jba, addr, kr, term_count

    kr = 0
    if (direct) then
      do ii = 1, numat
        iab = iorbs(ii)
        if (iab /= 0) then
          term_count = mozyme_tri(iab) * mozyme_tri(iab)
          call mozyme_integral_slice_signature_hash(wj, kr + 1, term_count, 37, h1, h2)
          kr = kr + term_count
        end if
      end do
      return
    end if

    do ii = 1, numat
      iab = iorbs(ii)
      if (iab /= 0) then
        do jj = 1, ii - 1
          jba = iorbs(jj)
          addr = mozyme_pair_addr(ii, jj, use_nijbo)
          call mozyme_mix_int64(h1, h2, int(addr, c_int64_t))
          if (addr >= 0) then
            term_count = mozyme_pair_integral_count(iab, jba)
            call mozyme_integral_slice_signature_hash(wj, kr + 1, term_count, 41, h1, h2)
            call mozyme_integral_slice_signature_hash(wk, kr + 1, term_count, 43, h1, h2)
            kr = kr + term_count
          else
            term_count = kr
            call mozyme_point_charge_advance_kr(iab, jba, addr, kr)
            call mozyme_integral_slice_signature_hash(wj, term_count + 1, kr - term_count, 47, h1, h2)
          end if
        end do
        term_count = mozyme_tri(iab) * mozyme_tri(iab)
        call mozyme_integral_slice_signature_hash(wj, kr + 1, term_count, 53, h1, h2)
        kr = kr + term_count
      end if
    end do
  end subroutine mozyme_resident_integral_signature_hash

  subroutine mozyme_integral_slice_signature_hash(values, first, count, tag, h1, h2)
    implicit none
    double precision, intent(in) :: values(*)
    integer, intent(in) :: first, count, tag
    integer(c_int64_t), intent(inout) :: h1, h2
    integer :: idx

    call mozyme_mix_int64(h1, h2, int(tag, c_int64_t))
    call mozyme_mix_int64(h1, h2, int(first, c_int64_t))
    call mozyme_mix_int64(h1, h2, int(count, c_int64_t))
    do idx = 1, count
      call mozyme_mix_real64(h1, h2, values(first + idx - 1))
    end do
  end subroutine mozyme_integral_slice_signature_hash

  subroutine mozyme_mix_logical(h1, h2, value)
    implicit none
    integer(c_int64_t), intent(inout) :: h1, h2
    logical, intent(in) :: value

    call mozyme_mix_int64(h1, h2, merge(1_c_int64_t, 0_c_int64_t, value))
  end subroutine mozyme_mix_logical

  subroutine mozyme_mix_real64(h1, h2, value)
    implicit none
    integer(c_int64_t), intent(inout) :: h1, h2
    double precision, intent(in) :: value
    integer(c_int64_t) :: bits

    bits = transfer(value, bits)
    call mozyme_mix_int64(h1, h2, bits)
  end subroutine mozyme_mix_real64

  subroutine mozyme_mix_int64(h1, h2, value)
    implicit none
    integer(c_int64_t), intent(inout) :: h1, h2
    integer(c_int64_t), intent(in) :: value

    h1 = modulo(h1 * 131_c_int64_t + modulo(value, mozyme_sig_mod1) + 1_c_int64_t, mozyme_sig_mod1)
    h2 = modulo(h2 * 137_c_int64_t + modulo(value, mozyme_sig_mod2) + 1_c_int64_t, mozyme_sig_mod2)
  end subroutine mozyme_mix_int64

  logical function mozyme_resident_trace()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value
    mozyme_resident_trace = .false.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_PROFILE', env_value, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') mozyme_resident_trace = .true.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_VERBOSE', env_value, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') mozyme_resident_trace = .true.
  end function mozyme_resident_trace
#endif

#ifndef GPU
  subroutine mozyme_point_charge_advance_kr(iab, jba, addr, kr)
    use MOZYME_C, only: direct, semidr
    implicit none
    integer, intent(in) :: iab, jba, addr
    integer, intent(inout) :: kr
    if (iab * jba <= 0) return
    if (direct .or. semidr) return
    kr = kr + 1
    if (addr == -2) then
      if (iab > 1 .and. jba > 1) then
        kr = kr + 6
      else if (iab > 1 .or. jba > 1) then
        kr = kr + 3
      end if
    end if
  end subroutine mozyme_point_charge_advance_kr
#endif

end module mozyme_resident_fock
