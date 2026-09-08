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

subroutine iter_for_MOZYME (ee)
    use molkst_C, only: norbs, step_num, step_num0, numcal, numcal0, nscf, escf, &
       & numat,  enuclr, atheat, emin, keywrd, moperr, line, use_disk
!
    use chanel_C, only: iw, iend, end_fn
!
    use MOZYME_C, only : icocc, icvir, iorbs, ncocc, &
         ncvir, nncf, nnce, cocc, cvir, nvirtual, noccupied, &
         cocc_dim, icocc_dim, icvir_dim, cvir_dim, nelred, &
         norred, numred, ncf, nce, scfref, thresh, ovmax, tiny, shift, &
         energy_diff, sumt, sumb, ijc, use_three_point_extrap, &
         ws, p1, p2, p3, partf, partp, idiag, mode, &
         gpu_occ_enabled, gpu_virt_enabled, at_res, allres
!
    use funcon_C, only: fpc_9
#ifdef GPU
    use mod_vars_cuda, only: lgpu, mozyme_gpu, mozyme_check_gpu
#endif
    use common_arrays_C, only : f, p
    use iter_C, only : pold
    use cosmo_C, only: useps, lpka, solv_energy
    use linear_cosmo, only : c_proc
    use mozyme_gpu_scf_driver, only : mozyme_gpu_scf_early_probe, &
      mozyme_gpu_scf_force_final_reorth, &
      mozyme_gpu_scf_requested, mozyme_gpu_scf_no_fallback_required, &
      mozyme_gpu_scf_try
    use mozyme_gpu_makvec, only : mozyme_gpu_makvec_try
    use mozyme_gpu_relocalize, only : mozyme_gpu_relocalize_try
    use mozyme_gpu_reorth, only : mozyme_gpu_reorth_try
    use mozyme_gpu_tidy, only : mozyme_gpu_tidy_try
    use mozyme_section_timers, only : mozyme_section_timer_begin, &
      mozyme_section_timer_end, mozyme_section_timer_report_all
    implicit none
!
    double precision, intent (out) :: ee
   !
   !***********************************************************************
   !
   !                   The Occupied Set
   !
   !  COCC:     Starting localized filled M.O.s        Atomic Orbitals
   !  NCOCC:    Starting address of the filled LMOs    Atomic Orbitals
   !  ICOCC:    Atom indices for the filled LMOs       Atoms
   !  NCF:      Number of atoms in each filled LMO     Atoms
   !  NNCF:     Starting address of each M.O. in NCF   Atoms
   !
   !                   The Virtual Set
   !
   !  CVIR:     Starting localized empty M.O.s         Atomic Orbitals
   !  NCVIR:    Starting address of the empty LMOs     Atomic Orbitals
   !  ICVIR:    Atom indices for the empty LMOs        Atoms
   !  NCE:      Number of atoms in each empty LMO      Atoms
   !  NNCE:     Starting address of each M.O. in NCE   Atoms
   !
   !***********************************************************************
    character :: xchar = " "
    logical, save :: prtpls, orthog, times, scf1, bigscf, debug, prtden, &
         & prtfok, okscf, opend, panic = .true., store_useps
    integer, save :: i, idnout, nocc1, nvir1, istabl, idiagg, nhb, itrmax, &
      niter, lno, iemax, iemin, lnv, mn, indi, j, k, l, nij = 0, re_local, &
      icalcn = 0, imol = 0, nmol = 0, lstart = 0, add_niter
    integer :: resident_fock_mode
    integer :: resident_tidy_code, resident_tidy_resize_status
    integer, save :: resident_tidy_imode(2) = 0
    double precision, save :: selcon, eold,  sum
    integer, external :: ijbo
    logical, external :: PLS_faulty
    external :: check_gpu
    external :: mozyme_gpu_strict_abort
    double precision, external :: helecz, reada
    integer, dimension (:), allocatable :: iwork
    double precision, dimension (:), allocatable :: rwork
    integer :: bad_occ, bad_virt, res_idx
    character(len=4) :: res_name
    logical :: gpu_error_occ, gpu_error_virt
    logical :: resident_gpu_handled, resident_scf_complete
    logical :: resident_isitsc_done, resident_isitsc_okscf
    logical :: resident_loop_control_needed
    logical :: resident_early_probe_handled
    logical :: resident_strict_required
    logical :: resident_pls_restart_needed
    logical :: resident_density_current
    logical :: resident_initial_setup_needed
    logical :: resident_pre_tidy_attempt_needed
    logical :: resident_final_reorth_due
    logical :: resident_final_reorth_done
    logical :: resident_tidy_done
    logical :: resident_tidy_select_lmos
    logical :: resident_tidy_mode_due
    logical :: makvec_gpu_done
    double precision :: mozyme_timer
    add_niter = 0
    resident_strict_required = mozyme_gpu_scf_no_fallback_required()
    resident_final_reorth_done = .false.
!
        80  continue
    resident_density_current = .false.
    if (nmol /= numcal) then
      !
      !  INITIALIZE
      !
      !     THRESH          ERROR IN H.O.F., RELATIVE TO THRESH=1.D-13
      !
      !     1.D-8             0.21??     SCF unstable
      !     1.D-9             0.0294     SCF stable
      !     1.D-10            0.0112     SCF stable
      !     1.D-11            0.0040     SCF stable
      !     1.D-12            0.0011     SCF stable
      !
      !   IF THRESH IS CHANGED, THEN ALSO CHANGE DEFAULT IN WRTKEY
      !
      thresh = 1.d-13
      scfref = 0.d0
      i = Index (keywrd, " RELTHR")
      if (i /= 0) then
        thresh = reada (keywrd, i) * thresh
      end if
      i = Index (keywrd, " THRESH")
      if (i /= 0) then
        thresh = reada (keywrd, i)
      end if!

      thresh = Max (thresh, 1.d-25)
      prtpls = (Index (keywrd, " PL ") /= 0)
      orthog = (Index (keywrd, " REORTH") /= 0)
      times = (Index (keywrd, " TIMES") /= 0)
      i = Index (keywrd, " DENOUT=")
      if (i /= 0) then
        idnout = Nint (reada (keywrd, i+8))
      else
        idnout = 10000000
      end if
      itrmax = 2000
      if (Index (keywrd, " ITRY") /= 0) then
        itrmax = Nint (reada (keywrd, Index (keywrd, " ITRY")))
      end if
      re_local = 100000000
      i = index(keywrd," RE-LOC")
      if (i /= 0) then
        j = index(keywrd(i + 7:), " ") + i + 7
        if (index(keywrd(i:j), "=") /= 0) then ! Allow for RE-LOC=, RE-LOCAL=, etc.
          i = index(keywrd(i:j), "=") + i
          re_local = nint(reada(keywrd,i))
        end if
      end if
      nocc1 = 0
      nvir1 = 0
      scf1 = (Index (keywrd, " 1SCF") /= 0)
      bigscf = (Index (keywrd, " BIGSCF") /= 0)
      if (Index (keywrd, " OLDEN") == 0) then
        bigscf = .true.
      end if
      if (resident_strict_required .and. Index (keywrd, " OLDEN") /= 0) then
        bigscf = .true.
      end if
      debug = (Index (keywrd, " DEBUG") /= 0)
      prtden = (Index (keywrd, " DENS") /= 0 .and. debug)
      prtfok = (Index (keywrd, " FOCK") /= 0 .and. debug)
      debug = (Index (keywrd, " ITER") /= 0)
      idiagg = 0
      nocc1 = noccupied
      store_useps = useps
      if (resident_strict_required) then
        if (Index (keywrd, " DENOUT") /= 0) then
          call mozyme_gpu_strict_abort('strict_denout_host_output', &
            'MOZYME GPU strict resident SCF does not support DENOUT host density output')
          return
        end if
        if (Index (keywrd, " OLDEN") /= 0) then
          call mozyme_gpu_strict_abort('strict_olden_host_lmo_restore', &
            'MOZYME GPU strict resident SCF does not support OLDEN host LMO restore')
          return
        end if
        if (lpka) then
          call mozyme_gpu_strict_abort('strict_solvent_fock', &
            'MOZYME GPU strict resident SCF does not support LPKA solvent Fock')
          return
        end if
        if (Index (keywrd, " PKA") /= 0) then
          call mozyme_gpu_strict_abort('strict_pka_host_output', &
            'MOZYME GPU strict resident SCF does not support PKA host pKa output')
          return
        end if
      end if
      if (Index (keywrd, " LEWIS") /= 0) then
        if (resident_strict_required) then
          makvec_gpu_done = mozyme_gpu_makvec_try()
          if (.not. makvec_gpu_done) then
            call mozyme_gpu_strict_abort('strict_lewis_gpu_makvec_failed', &
              'MOZYME GPU strict resident SCF could not complete LEWIS setup on GPU')
          end if
          return
        end if
        call makvec()
        return
      end if
      if (Index (keywrd, " OLDEN") /= 0) then
          call mozyme_section_timer_begin('iter_olden_load', mozyme_timer)
          if (use_disk) call pinout(0, (index(keywrd, "SILENT") == 0))
          call mozyme_section_timer_end('iter_olden_load', mozyme_timer)
          if (add_niter /= 0)  call l_control("OLDEN", len_trim("OLDEN"), -1)
          if (add_niter /= 0)  call l_control("SILENT", len_trim("SILENT"), -1)
          if (moperr) return
          if (resident_strict_required) then
            write(iw,'(1x,a)') &
              '[MOZYME GPU SCF] olden_setup=host_lmo_restore setup_only=1'
            call flush(iw)
          else
            call mozyme_section_timer_begin('iter_density_olden', mozyme_timer)
            call density_for_MOZYME (p, 0, noccupied, partp)
            call mozyme_section_timer_end('iter_density_olden', mozyme_timer)
            partp = p
          end if
      else
        useps = .false.
        if (index(keywrd, "OLD_SCF") /= 0) then
          if (resident_strict_required) then
            call mozyme_gpu_strict_abort('strict_old_scf_existing_lmo', &
              'MOZYME GPU strict resident SCF does not accept OLD_SCF host-existing LMOs as makvec proof')
            return
          end if
        else
          makvec_gpu_done = mozyme_gpu_makvec_try()
          if (.not. makvec_gpu_done) then
            if (resident_strict_required) then
              call mozyme_gpu_strict_abort('strict_cpu_makvec', &
                'MOZYME GPU strict resident SCF does not support CPU makvec initial LMO construction')
              return
            end if
            call mozyme_section_timer_begin('iter_makvec', mozyme_timer)
            call makvec()
            call mozyme_section_timer_end('iter_makvec', mozyme_timer)
          end if
        end if
      end if
      if (moperr) return
      !
      !  A NEW MOLECULE, THEREFORE SEARCH FOR ALL WEAK INTERACTIONS.
      !
      nhb = 0
    else
      !
      !  A MODIFIED GEOMETRY, THEREFORE SEARCH ONLY FOR NEW
      !  OR VERY WEAK INTERACTIONS.
      !
      nhb = 3
    end if
    if (mod(nscf + 1, re_local) == 0) then
      if (resident_strict_required) then
        call mozyme_section_timer_begin('iter_reloc_occ', mozyme_timer)
        if (.not. mozyme_gpu_relocalize_try("OCCUPIED")) then
          call mozyme_section_timer_end('iter_reloc_occ', mozyme_timer)
          call mozyme_gpu_strict_abort('strict_cpu_relocalization', &
            'MOZYME GPU strict resident SCF could not complete occupied re-localization on GPU')
          return
        end if
        call mozyme_section_timer_end('iter_reloc_occ', mozyme_timer)
        call mozyme_section_timer_begin('iter_reloc_virt', mozyme_timer)
        if (.not. mozyme_gpu_relocalize_try("VIRTUAL")) then
          call mozyme_section_timer_end('iter_reloc_virt', mozyme_timer)
          call mozyme_gpu_strict_abort('strict_cpu_relocalization', &
            'MOZYME GPU strict resident SCF could not complete virtual re-localization on GPU')
          return
        end if
        call mozyme_section_timer_end('iter_reloc_virt', mozyme_timer)
      else
        write(iw,"(/10x,a,/)")"  LMOs being Re-Localized"
        call mozyme_section_timer_begin('iter_reloc_occ', mozyme_timer)
        call local_for_MOZYME("OCCUPIED")
        call mozyme_section_timer_end('iter_reloc_occ', mozyme_timer)
        call mozyme_section_timer_begin('iter_reloc_virt', mozyme_timer)
        call local_for_MOZYME("VIRTUAL")
        call mozyme_section_timer_end('iter_reloc_virt', mozyme_timer)
      end if
    end if
    useps = store_useps
    if (lpka) useps = .true.
    if (icalcn /= step_num) then
      istabl = 0
      eold = 0.d0
      ovmax = 0.d0
      call scfcri (selcon)
      idiagg = 0
      tiny = 0.d0
    end if
    if (index(keywrd," tighten")  /= 0) then
      call l_control("tighten", len_trim("tighten"), -1)
      selcon = max( selcon*0.1d0, 1.d-3)
    end if
   !
   ! Fill the array IDIAG pointing to the diagonal elements of P
   !
    l = 0
    do i = 1, numat
      j = ijbo (i, i)
      do k = 1, iorbs(i)
        l = l + 1
        j = j + k
        idiag(l) = j
      end do
    end do
   !
   !  Zero out POLD and P1 so that the new calculation is not affected
   !  by a previous one.
   !
    pold = 0.d0
    p1 = 0.d0
    nscf = nscf + 1
    resident_final_reorth_due = orthog .and. &
      (Mod(nscf + 1, 10) == 1 .or. &
      (resident_strict_required .and. mozyme_gpu_scf_force_final_reorth()))
   !
   !  Force IDIAGG to be even - this is to ensure that DIAGG1
   !  re-builds the interaction list, in case the temporary space
   !  has been used between the calls to ITER.
   !
    if (Mod(idiagg, 2) == 1) idiagg = idiagg + 1
    iemin = 0
    iemax = 0
    niter = 0
    shift = 0.0d0
    use_three_point_extrap = .true.
    if (times) then
      call timer (" At start of ITER")
    end if
!***********************************************************
!
!   Everything is now set up to allow the SCF to be run
!
!***********************************************************
    do  !  Big loop to run the SCF
!----------------
      nocc1 = nelred / 2
      nvir1 = norred - nocc1
      resident_initial_setup_needed = (imol /= numcal .or. &
        (icalcn /= step_num .and. numat > numred+1))
      resident_loop_control_needed = (.not. bigscf .and. numcal == 1+numcal0)
      resident_pre_tidy_attempt_needed = mozyme_gpu_scf_requested() .and. &
        niter == 0 .and. resident_initial_setup_needed .and. &
        nocc1 > 0 .and. nvir1 > 0 .and. &
        itrmax > niter .and. .not. resident_loop_control_needed .and. &
        .not. lpka
      if (resident_pre_tidy_attempt_needed) then
        resident_tidy_done = .false.
        resident_early_probe_handled = mozyme_gpu_scf_early_probe(niter, &
          nocc1, nvir1, itrmax, selcon)
        if (resident_early_probe_handled) then
          if (resident_strict_required) then
            call mozyme_gpu_strict_abort('strict_early_probe_fallback', &
              'MOZYME GPU strict resident SCF does not support early probe fallback')
          end if
          call mozyme_section_timer_report_all()
          return
        end if
        if (resident_strict_required) then
          if (nmol /= numcal) resident_tidy_imode = step_num
          resident_tidy_mode_due = step_num > 1+step_num0 .and. &
            step_num /= resident_tidy_imode(1)
          resident_tidy_select_lmos = resident_tidy_mode_due .and. &
            numat > numred+1
            do
              call mozyme_section_timer_begin('iter_tidy_occ', mozyme_timer)
              if (mozyme_gpu_tidy_try(1, lno, mn, &
                  use_selmos=resident_tidy_select_lmos, &
                  error_code=resident_tidy_code)) then
                if (resident_tidy_mode_due) resident_tidy_imode(1) = step_num
                call mozyme_section_timer_end('iter_tidy_occ', mozyme_timer)
                exit
              end if
              call mozyme_section_timer_end('iter_tidy_occ', mozyme_timer)
              if (resident_tidy_code == -506) then
                call mozyme_gpu_grow_lmo_storage(resident_tidy_resize_status)
                if (resident_tidy_resize_status == 0) cycle
                call mozyme_gpu_strict_abort('strict_resident_tidy_resize_failed', &
                  'MOZYME GPU strict resident SCF could not grow LMO storage for occupied TIDY')
                return
              end if
              call mozyme_gpu_strict_abort('strict_resident_tidy_failed', &
                'MOZYME GPU strict resident SCF could not complete occupied TIDY on GPU')
              return
            end do
            resident_tidy_mode_due = step_num > 1+step_num0 .and. &
              step_num /= resident_tidy_imode(2)
            resident_tidy_select_lmos = resident_tidy_mode_due .and. &
              numat > numred+1
            do
              call mozyme_section_timer_begin('iter_tidy_virt', mozyme_timer)
              if (mozyme_gpu_tidy_try(2, lnv, mn, &
                  use_selmos=resident_tidy_select_lmos, &
                  error_code=resident_tidy_code)) then
                if (resident_tidy_mode_due) resident_tidy_imode(2) = step_num
                call mozyme_section_timer_end('iter_tidy_virt', mozyme_timer)
                exit
              end if
              call mozyme_section_timer_end('iter_tidy_virt', mozyme_timer)
              if (resident_tidy_code == -506) then
                call mozyme_gpu_grow_lmo_storage(resident_tidy_resize_status)
                if (resident_tidy_resize_status == 0) cycle
                call mozyme_gpu_strict_abort('strict_resident_tidy_resize_failed', &
                  'MOZYME GPU strict resident SCF could not grow LMO storage for virtual TIDY')
                return
              end if
              call mozyme_gpu_strict_abort('strict_resident_tidy_failed', &
                'MOZYME GPU strict resident SCF could not complete virtual TIDY on GPU')
              return
            end do
            resident_tidy_done = .true.
            nocc1 = nelred / 2
            nvir1 = norred - nocc1
            if (nocc1 <= 0 .or. nvir1 <= 0) then
              call mozyme_gpu_strict_abort('strict_resident_tidy_empty_space', &
                'MOZYME GPU strict resident SCF produced empty selected TIDY space')
              return
            end if
        end if
        if (nmol == numcal .and. numat > numred+1) then
          resident_fock_mode = 1
        else
          resident_fock_mode = 0
        end if
        ! The resident driver owns initial setup before CPU tidy/check/setup
        ! bookends for full and selected partial-active-space paths.
        ! Unsupported cases fall through to the existing CPU path unless the
        ! strict resident proof contract is active.
        call mozyme_section_timer_begin('iter_resident_scf_boundary', mozyme_timer)
        resident_scf_complete = .false.
        resident_isitsc_done = .false.
        resident_isitsc_okscf = .false.
        resident_gpu_handled = .false.
        if (.not. resident_loop_control_needed) then
          resident_gpu_handled = mozyme_gpu_scf_try(ee, niter, nocc1, nvir1, &
            itrmax, selcon, resident_fock_mode, idiagg, nhb, resident_fock_mode, &
            resident_scf_complete, eold, iemin, iemax, lstart, &
            initial_setup=.true., block_on_failure=resident_strict_required, &
            final_reorth=resident_strict_required .and. &
              resident_final_reorth_due, &
            final_reorth_done=resident_final_reorth_done, &
            initial_tidy_done=resident_tidy_done)
        end if
        if (resident_gpu_handled) then
          resident_isitsc_done = .true.
          resident_isitsc_okscf = resident_scf_complete
          escf = (ee+enuclr) * fpc_9 + atheat
          if (useps) then
                escf = escf + solv_energy * fpc_9
          end if
          if (resident_initial_setup_needed) then
            icalcn = step_num
            imol = numcal
          end if
          call mozyme_section_timer_end('iter_resident_scf_boundary', mozyme_timer)
          if (resident_scf_complete) then
            energy_diff = escf - eold
            eold = escf
            okscf = .true.
            resident_density_current = .true.
            exit
          end if
          goto 700
        end if
        call mozyme_section_timer_end('iter_resident_scf_boundary', mozyme_timer)
        if (resident_strict_required) then
          call mozyme_gpu_strict_abort('strict_pre_tidy_failed', &
            'MOZYME GPU strict resident SCF failed before CPU tidy')
          return
        end if
      else if (resident_strict_required) then
        call mozyme_gpu_strict_abort('strict_pre_tidy_not_started', &
          'MOZYME GPU strict resident SCF could not start before CPU tidy')
        return
      end if
      do
        call mozyme_section_timer_begin('iter_tidy_occ', mozyme_timer)
        call tidy (noccupied, ncf, icocc, icocc_dim, cocc, cocc_dim, nncf, ncocc, lno, mn, 1)
        call mozyme_section_timer_end('iter_tidy_occ', mozyme_timer)
        if (moperr) then
!
!  During a run of "tidy", the amount of expansion space for the LMO's to use had
!  become small.  The LMOs were stored to disc.  The old arrays will now be deleted
!  and re-created 60% larger than before.  Then the LMOs are read off disc
!
          if (use_disk) then
            deallocate (icocc, cocc, icvir, cvir)

!
!  Re-allocate more memory
!
            icocc_dim = Nint(icocc_dim*1.6)
            cocc_dim = Nint(cocc_dim*1.6)
            icvir_dim = Nint(icvir_dim*1.6)
            cvir_dim = Nint(cvir_dim*1.6)
            allocate (icocc(icocc_dim), cocc(cocc_dim), &
                  & icvir(icvir_dim), cvir(cvir_dim), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
!
!  Read in old density
!
            call pinout (0, .false.)
          else
!
!  Re-allocate LMO memory without use of disk
!
            allocate (iwork(icocc_dim), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            iwork = icocc
            deallocate (icocc)
            allocate (icocc(Nint(icocc_dim*1.6)), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            icocc(:icocc_dim) = iwork(:icocc_dim)
            icocc(icocc_dim+1:) = 0
            icocc_dim = Nint(icocc_dim*1.6)
            deallocate (iwork)

            allocate (rwork(cocc_dim), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            rwork = cocc
            deallocate (cocc)
            allocate (cocc(Nint(cocc_dim*1.6)), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            cocc(:cocc_dim) = rwork(:cocc_dim)
            cocc(cocc_dim+1:) = 0.0d0
            cocc_dim = Nint(cocc_dim*1.6)
            deallocate (rwork)

            allocate (iwork(icvir_dim), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            iwork = icvir
            deallocate (icvir)
            allocate (icvir(Nint(icvir_dim*1.6)), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            icvir(:icvir_dim) = iwork(:icvir_dim)
            icvir(icvir_dim+1:) = 0
            icvir_dim = Nint(icvir_dim*1.6)
            deallocate (iwork)

            allocate (rwork(cvir_dim), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            rwork = cvir
            deallocate (cvir)
            allocate (cvir(Nint(cvir_dim*1.6)), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            cvir(:cvir_dim) = rwork(:cvir_dim)
            cvir(cvir_dim+1:) = 0.0d0
            cvir_dim = Nint(cvir_dim*1.6)
            deallocate (rwork)
          end if
          moperr = .false.
        else
          exit
        end if
      end do
      do
        call mozyme_section_timer_begin('iter_tidy_virt', mozyme_timer)
        call tidy (nvirtual, nce, icvir, icvir_dim, cvir, cvir_dim, nnce, ncvir, lnv, mn, 2)
        call mozyme_section_timer_end('iter_tidy_virt', mozyme_timer)
        if (moperr) then
!  Delete old memory
!
          if (use_disk) then
            deallocate (icocc, cocc, icvir, cvir)

!
!  Re-allocate more memory
!
            icocc_dim = Nint(icocc_dim*1.6)
            cocc_dim = Nint(cocc_dim*1.6)
            icvir_dim = Nint(icvir_dim*1.6)
            cvir_dim = Nint(cvir_dim*1.6)
            allocate (icocc(icocc_dim), cocc(cocc_dim), &
                  & icvir(icvir_dim), cvir(cvir_dim), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
!
!  Read in old density
!
            call pinout (0, .false.)
          else
!
!  Re-allocate LMO memory without use of disk
!
            allocate (iwork(icocc_dim), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            iwork = icocc
            deallocate (icocc)
            allocate (icocc(Nint(icocc_dim*1.6)), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            icocc(:icocc_dim) = iwork(:icocc_dim)
            icocc(icocc_dim+1:) = 0
            icocc_dim = Nint(icocc_dim*1.6)
            deallocate (iwork)

            allocate (rwork(cocc_dim), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            rwork = cocc
            deallocate (cocc)
            allocate (cocc(Nint(cocc_dim*1.6)), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            cocc(:cocc_dim) = rwork(:cocc_dim)
            cocc(cocc_dim+1:) = 0.0d0
            cocc_dim = Nint(cocc_dim*1.6)
            deallocate (rwork)

            allocate (iwork(icvir_dim), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            iwork = icvir
            deallocate (icvir)
            allocate (icvir(Nint(icvir_dim*1.6)), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            icvir(:icvir_dim) = iwork(:icvir_dim)
            icvir(icvir_dim+1:) = 0
            icvir_dim = Nint(icvir_dim*1.6)
            deallocate (iwork)

            allocate (rwork(cvir_dim), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            rwork = cvir
            deallocate (cvir)
            allocate (cvir(Nint(cvir_dim*1.6)), stat = i)
            if (i /= 0) then
              call memory_error(" iter_for MOZYME")
              return
            end if
            cvir(:cvir_dim) = rwork(:cvir_dim)
            cvir(cvir_dim+1:) = 0.0d0
            cvir_dim = Nint(cvir_dim*1.6)
            deallocate (rwork)
          end if
          moperr = .false.
        else
          exit
        end if
      end do
!----------------
      nocc1 = nelred / 2
      nvir1 = norred - nocc1
      resident_early_probe_handled = mozyme_gpu_scf_early_probe(niter, nocc1, &
        nvir1, itrmax, selcon)
      if (resident_early_probe_handled) then
        if (resident_strict_required) then
          call mozyme_gpu_strict_abort('strict_early_probe_fallback', &
            'MOZYME GPU strict resident SCF does not support early probe fallback')
        end if
        call mozyme_section_timer_report_all()
        return
      end if
      resident_initial_setup_needed = (imol /= numcal .or. &
        (icalcn /= step_num .and. numat > numred+1))
      if (resident_initial_setup_needed) then
        if (nmol == numcal .and. numat > numred+1) then
          resident_fock_mode = 1
        else
          resident_fock_mode = 0
        end if
        ! Fallback initial setup after CPU tidy if the pre-tidy resident
        ! attempt did not handle the case.
        resident_loop_control_needed = (.not. bigscf .and. numcal == 1+numcal0)
        call mozyme_section_timer_begin('iter_resident_scf_boundary', mozyme_timer)
        resident_scf_complete = .false.
        resident_isitsc_done = .false.
        resident_isitsc_okscf = .false.
        resident_gpu_handled = .false.
        if (.not. resident_loop_control_needed) then
          resident_gpu_handled = mozyme_gpu_scf_try(ee, niter, nocc1, nvir1, &
            itrmax, selcon, resident_fock_mode, idiagg, nhb, resident_fock_mode, &
            resident_scf_complete, eold, iemin, iemax, lstart, &
            initial_setup=.true., block_on_failure=.true., &
            final_reorth=resident_strict_required .and. &
              resident_final_reorth_due, &
            final_reorth_done=resident_final_reorth_done)
        end if
        if (resident_gpu_handled) then
          resident_isitsc_done = .true.
          resident_isitsc_okscf = resident_scf_complete
          escf = (ee+enuclr) * fpc_9 + atheat
          if (useps) then
                escf = escf + solv_energy * fpc_9
          end if
          icalcn = step_num
          imol = numcal
          call mozyme_section_timer_end('iter_resident_scf_boundary', mozyme_timer)
          if (resident_scf_complete) then
            energy_diff = escf - eold
            eold = escf
            okscf = .true.
            resident_density_current = .true.
            exit
          end if
          goto 700
        end if
        call mozyme_section_timer_end('iter_resident_scf_boundary', mozyme_timer)
        if (resident_strict_required) then
          call mozyme_gpu_strict_abort('strict_post_tidy_failed', &
            'MOZYME GPU strict resident SCF failed after CPU tidy')
          return
        end if
      end if
      !
      !   REMOVE ELECTRON DENSITY DUE TO LMO'S INVOLVED IN SCF FROM
      !   THE DENSITY MATRIX
      !
      if (imol /= numcal .or. icalcn /= step_num .and. numat > numred+1) then
        !---------------------------------------------------------
        !
        !   THIS PART IS ONLY RUN WHEN ICALCN IS INCREMENTED
        !
        call mozyme_section_timer_begin('iter_density_initial', mozyme_timer)
        call density_for_MOZYME (p, 0, noccupied, partp) ! Build the whole density matrix
        call mozyme_section_timer_end('iter_density_initial', mozyme_timer)
!
        if (times) call timer (" After DENSIT")
        if (prtden) then
          write (iw, "(' DENSITY MATRIX TO GO INTO PARTP')")
          call vecprt_for_MOZYME (p, norbs)
        end if
        call mozyme_section_timer_begin('iter_setupk', mozyme_timer)
        call setupk (nocc1) ! Work out the atom list to be used in the SCF
        call mozyme_section_timer_end('iter_setupk', mozyme_timer)
        if (times) call timer (" After SETUPK")
        if (imol == numcal .and. numat > numred+1) then
          call mozyme_section_timer_begin('iter_density_remove', mozyme_timer)
          call density_for_MOZYME (partp, -1, nocc1, p) ! Remove density due to atoms to be
                                                        ! used in the SCF
          call mozyme_section_timer_end('iter_density_remove', mozyme_timer)
          if (prtden) then
            write (iw, "(' DENSITY MATRIX IN PARTP')")
            call vecprt_for_MOZYME (partp, norbs)
          end if
        end if
        call mozyme_section_timer_begin('iter_buildf_initial', mozyme_timer)
        call buildf (f, partf, 0)
        call mozyme_section_timer_end('iter_buildf_initial', mozyme_timer)
        if (prtfok) then
          write (iw, "(' FOCK MATRIX AT START OF ITER')")
          call vecprt_for_MOZYME (f, norbs)
        end if
        if (icalcn /= step_num) then
          if (times) call timer (" After BUILDF")
          call mozyme_section_timer_begin('iter_helecz_initial', mozyme_timer)
          ee = helecz()
          call mozyme_section_timer_end('iter_helecz_initial', mozyme_timer)
          if (times) call timer (" After HELEC")
          escf = (ee+enuclr) * fpc_9 + atheat
          if (useps) then
                escf = escf + solv_energy * fpc_9
          end if
          if (prtpls)  write (iw, "(/,A,F16.6,A,/)") " PLS ESCF USING THE OLD LMOs:", escf, " KCAL/MOL"
          if (use_disk) then
            endfile (iw)
            backspace (iw)
          end if
        end if
        if (imol == numcal .and. numat > numred+1) then
          call mozyme_section_timer_begin('iter_buildf_partial', mozyme_timer)
          call buildf (partf, f, -1)
          call mozyme_section_timer_end('iter_buildf_partial', mozyme_timer)
        end if
        icalcn = step_num
        imol = numcal
      end if
      if (.not. resident_initial_setup_needed .and. niter <= 10) then
        resident_loop_control_needed = (.not. bigscf .and. numcal == 1+numcal0)
        if (nmol == numcal .and. numat > numred+1) then
          resident_fock_mode = 1
        else
          resident_fock_mode = 0
        end if
        call mozyme_section_timer_begin('iter_resident_scf_boundary', mozyme_timer)
        resident_scf_complete = .false.
        resident_isitsc_done = .false.
        resident_isitsc_okscf = .false.
        resident_gpu_handled = .false.
        if (.not. resident_loop_control_needed) then
          resident_gpu_handled = mozyme_gpu_scf_try(ee, niter, nocc1, nvir1, &
            itrmax, selcon, resident_fock_mode, idiagg, nhb, resident_fock_mode, &
            resident_scf_complete, eold, iemin, iemax, lstart, &
            final_reorth=resident_strict_required .and. &
              resident_final_reorth_due, &
            final_reorth_done=resident_final_reorth_done)
        end if
        if (resident_gpu_handled) then
          resident_isitsc_done = .true.
          resident_isitsc_okscf = resident_scf_complete
          escf = (ee+enuclr) * fpc_9 + atheat
          if (useps) then
                escf = escf + solv_energy * fpc_9
          end if
          call mozyme_section_timer_end('iter_resident_scf_boundary', mozyme_timer)
          if (resident_scf_complete) then
            energy_diff = escf - eold
            eold = escf
            okscf = .true.
            resident_density_current = .true.
            exit
          end if
          goto 700
        end if
        call mozyme_section_timer_end('iter_resident_scf_boundary', mozyme_timer)
        if (resident_strict_required) then
          call mozyme_gpu_strict_abort('strict_cpu_iteration_work', &
            'MOZYME GPU strict resident SCF failed before CPU iteration work')
          return
        end if
      end if
!
!  Correct any small errors in normalization
!
      if (resident_strict_required) then
        call mozyme_gpu_strict_abort('strict_cpu_lmo_check', &
          'MOZYME GPU strict resident SCF does not support CPU LMO check')
        return
      end if
      bad_occ = 0
      call mozyme_section_timer_begin('iter_check_occ', mozyme_timer)
#ifdef GPU
      if (mozyme_gpu .and. lgpu .and. mozyme_check_gpu) then
        gpu_error_occ = .false.
        call check_gpu(nocc1, nncf, ncf, icocc, icocc_dim, iorbs, ncocc, cocc, cocc_dim, gpu_error_occ, bad_occ)
        moperr = gpu_error_occ
      else
        call check(nocc1, nncf, ncf, icocc, icocc_dim, iorbs, ncocc, cocc, cocc_dim)
      end if
#else
      call check(nocc1, nncf, ncf, icocc, icocc_dim, iorbs, ncocc, cocc, cocc_dim)
#endif
      call mozyme_section_timer_end('iter_check_occ', mozyme_timer)
#ifdef GPU
      if (moperr) then
        if (lgpu .and. mozyme_gpu .and. mozyme_check_gpu) then
          if (bad_occ >= 1 .and. bad_occ <= size(gpu_occ_enabled)) then
            gpu_occ_enabled(bad_occ) = .false.
            if (bad_occ >= 1 .and. bad_occ <= size(nncf)-1 .and. nncf(bad_occ)+1 <= size(icocc)) then
              res_idx = at_res(icocc(nncf(bad_occ)+1))
              if (res_idx >= lbound(allres,1) .and. res_idx <= ubound(allres,1)) then
                res_name = allres(res_idx)
              else
                res_name = '    '
              end if
              write (iw, '(//,1x,a,1x,i6,1x,a,1x,a)') 'MOZYME GPU: disabling occupied LMO', bad_occ, 'for residue', trim(res_name)
            else
              write (iw, '(//,1x,a,1x,i6)') 'MOZYME GPU: disabling occupied LMO', bad_occ
            end if
          else
            write (iw, '(//,1x,a)') 'MOZYME GPU: CHECK failure detected; disabling MOZYME GPU globally.'
            mozyme_gpu = .false.
          end if
          moperr = .false.
          goto 80
        end if
      end if
#endif
      if (moperr) return
      bad_virt = 0
      call mozyme_section_timer_begin('iter_check_virt', mozyme_timer)
#ifdef GPU
      if (mozyme_gpu .and. lgpu .and. mozyme_check_gpu) then
        gpu_error_virt = .false.
        call check_gpu(nvir1, nnce, nce, icvir, icvir_dim, iorbs, ncvir, cvir, cvir_dim, gpu_error_virt, bad_virt)
        moperr = gpu_error_virt
      else
        call check(nvir1, nnce, nce, icvir, icvir_dim, iorbs, ncvir, cvir, cvir_dim)
      end if
#else
      call check(nvir1, nnce, nce, icvir, icvir_dim, iorbs, ncvir, cvir, cvir_dim)
#endif
      call mozyme_section_timer_end('iter_check_virt', mozyme_timer)
#ifdef GPU
      if (moperr) then
        if (lgpu .and. mozyme_gpu .and. mozyme_check_gpu) then
          if (bad_virt >= 1 .and. bad_virt <= size(gpu_virt_enabled)) then
            gpu_virt_enabled(bad_virt) = .false.
            if (bad_virt >= 1 .and. bad_virt <= size(nnce)-1 .and. nnce(bad_virt)+1 <= size(icvir)) then
              res_idx = at_res(icvir(nnce(bad_virt)+1))
              if (res_idx >= lbound(allres,1) .and. res_idx <= ubound(allres,1)) then
                res_name = allres(res_idx)
              else
                res_name = '    '
              end if
              write (iw, '(//,1x,a,1x,i6,1x,a,1x,a)') 'MOZYME GPU: disabling virtual LMO', bad_virt, 'for residue', trim(res_name)
            else
              write (iw, '(//,1x,a,1x,i6)') 'MOZYME GPU: disabling virtual LMO', bad_virt
            end if
          else
            write (iw, '(//,1x,a)') 'MOZYME GPU: CHECK failure (virtual) detected; disabling MOZYME GPU globally.'
            mozyme_gpu = .false.
          end if
          moperr = .false.
          goto 80
        end if
      end if
#endif
      if (moperr) return
      resident_loop_control_needed = .false.
      resident_pls_restart_needed = .false.
      if (niter > 10 .and. add_niter == 0) then
        if (resident_strict_required) then
          call mozyme_gpu_strict_abort('strict_cpu_pls_supervisor', &
            'MOZYME GPU strict resident SCF does not support CPU PLS supervisor')
          return
        end if
        call mozyme_section_timer_begin('iter_pls_faulty', mozyme_timer)
        resident_pls_restart_needed = PLS_faulty()
        if (resident_pls_restart_needed) then
          call mozyme_section_timer_end('iter_pls_faulty', mozyme_timer)
          if (resident_strict_required) then
            call mozyme_gpu_strict_abort('strict_cpu_pls_restart', &
              'MOZYME GPU strict resident SCF does not support CPU PLS restart')
            return
          end if
!
!  When some systems are run using MOZYME, the DIAGG1 - DIAGG2 combination fails to converge,
!  and the ovmax converges to a non-zero minimum.  If the job is stopped and a <file>.den
!  is generated, then on restarting the same job, the fault is automatically corrected.
!
!  PLS_faulty detects the conditions of the failure, at run time, and silently writes out
!  the <file>.den, then after reading in the same file, it re-runs the SCF calculation.
!  This corrects the fault.
!
          numcal = numcal + 1
          add_niter = niter
          if (use_disk) call pinout(1, .false.)
          call l_control("OLDEN", len_trim("OLDEN"), 1)
          call l_control("SILENT", len_trim("SILENT"), 1)
          nscf = nscf - 1
          goto 80
        end if
        call mozyme_section_timer_end('iter_pls_faulty', mozyme_timer)
      end if
      if (.not. bigscf .and. numcal == 1+numcal0) then
        resident_loop_control_needed = .true.
        if (resident_strict_required) then
          call mozyme_gpu_strict_abort('strict_cpu_loop_control', &
            'MOZYME GPU strict resident SCF does not support CPU loop-control step')
          return
        end if
      end if
      ! Experimental resident-SCF boundary.  Strict resident mode resolves
      ! PLS restart and ADDHB inside the CUDA resident-control contract.
      if (nmol == numcal .and. numat > numred+1) then
        resident_fock_mode = 1
      else
        resident_fock_mode = 0
      end if
      call mozyme_section_timer_begin('iter_resident_scf_boundary', mozyme_timer)
      resident_scf_complete = .false.
      resident_isitsc_done = .false.
      resident_isitsc_okscf = .false.
      resident_gpu_handled = .false.
      if (.not. resident_loop_control_needed) then
        resident_gpu_handled = mozyme_gpu_scf_try(ee, niter, nocc1, nvir1, &
          itrmax, selcon, resident_fock_mode, idiagg, nhb, resident_fock_mode, &
          resident_scf_complete, eold, iemin, iemax, lstart, &
          final_reorth=resident_strict_required .and. resident_final_reorth_due, &
          final_reorth_done=resident_final_reorth_done)
      end if
      if (resident_gpu_handled) then
        resident_isitsc_done = .true.
        resident_isitsc_okscf = resident_scf_complete
        escf = (ee+enuclr) * fpc_9 + atheat
        if (useps) then
              escf = escf + solv_energy * fpc_9
        end if
        call mozyme_section_timer_end('iter_resident_scf_boundary', mozyme_timer)
        if (resident_scf_complete) then
          energy_diff = escf - eold
          eold = escf
          okscf = .true.
          resident_density_current = .true.
          exit
        end if
        goto 700
      end if
      call mozyme_section_timer_end('iter_resident_scf_boundary', mozyme_timer)
      if (resident_strict_required) then
        call mozyme_gpu_strict_abort('strict_cpu_scf_body', &
          'MOZYME GPU strict resident SCF failed before CPU SCF body')
        return
      end if
      if (Mod(niter+1, idnout) == 0 .and. use_disk) then
        write (iw, "(A)") " .den FILE TO BE WRITTEN OUT"
        endfile (iw)
        backspace (iw)
        call pinout (1, .true.)
        write (iw, "(A)") " .den FILE WRITTEN OUT"
        endfile (iw)
        backspace (iw)
      end if
      call mozyme_section_timer_begin('iter_eimp', mozyme_timer)
      call eimp ()
      call mozyme_section_timer_end('iter_eimp', mozyme_timer)
      if (nmol == numcal .and. numat > numred+1) then
        indi = 1
      else
        indi = 0
      end if
      if (bigscf .or. numcal /= 1+numcal0) then
          call mozyme_section_timer_begin('iter_diagg', mozyme_timer)
          call diagg (f, nocc1, nvir1,  idiagg,  partp, indi)
          call mozyme_section_timer_end('iter_diagg', mozyme_timer)
        idiagg = idiagg + 1
      else
        call mozyme_section_timer_begin('iter_density_iter', mozyme_timer)
        call density_for_MOZYME (p, mode, nocc1,  partp)
        call mozyme_section_timer_end('iter_density_iter', mozyme_timer)
        bigscf = .true.
      end if
      niter = niter + 1
      if (Mod(niter, 3) == 0 .and. nhb < 4) then
        nhb = nhb + 1
         !
         !   Check for missed hydrogen bonds and other unusual bonds
         !
          call mozyme_section_timer_begin('iter_addhb', mozyme_timer)
          call addhb (nocc1, nvir1, idiagg, nij, nhb)
          call mozyme_section_timer_end('iter_addhb', mozyme_timer)
         !
         !   If hydrogen bonds have been made, set IDIAGG even for DIAGG
         !   to make new interactions.
        if (nij /= 0 .and. Mod (idiagg, 2) == 1) then
          idiagg = idiagg + 1
        end if
      end if
      if (times) then
        if (i == 1) then
          call timer (" After DENS+1")
        else
          call timer (" After DENSIT")
        end if
      end if
      if (use_three_point_extrap) then
        call mozyme_section_timer_begin('iter_cnvgz', mozyme_timer)
        call cnvgz (p, pold, p1, p2, p3, niter, idiag)
        call mozyme_section_timer_end('iter_cnvgz', mozyme_timer)
      end if
      if (times) call timer (" After CNVG")
      if (prtden) then
        write (iw, "(' DENSITY MATRIX ON ITERATION',I4)") niter
        call vecprt_for_MOZYME (p, norbs)
      end if
      if (nmol == numcal .and. numat > numred+1) then
        call mozyme_section_timer_begin('iter_buildf_iter_partial', mozyme_timer)
        call buildf (f, partf, 1)  !
        call mozyme_section_timer_end('iter_buildf_iter_partial', mozyme_timer)
      else
        call mozyme_section_timer_begin('iter_buildf_iter_full', mozyme_timer)
        call buildf (f, partf, 0)
        call mozyme_section_timer_end('iter_buildf_iter_full', mozyme_timer)
      end if
      if (itrmax < 3) then
        call mozyme_section_timer_report_all()
        return
      end if
      if (times) call timer (" After BUILDF")
      if (prtfok) then
        write (iw, "(' FOCK    MATRIX ON ITERATION',I4)") niter
        call vecprt_for_MOZYME (f, norbs)
      end if
      call mozyme_section_timer_begin('iter_helecz_iter', mozyme_timer)
      ee = helecz()
      call mozyme_section_timer_end('iter_helecz_iter', mozyme_timer)
      escf = (ee+enuclr) * fpc_9 + atheat
      if (times) call timer (" After HELEC")
      if (useps) then
            escf = escf + solv_energy * fpc_9
      end if
700   continue
      energy_diff = escf - eold
      eold = escf
      if (Abs(ovmax) < 5.d0*selcon) then
        c_proc = 1.d0
      else
        c_proc = 5.d0*selcon/abs(ovmax)
      end if
      escf = max(-999999.d0, min(999999.d0, escf))
      if (abs(energy_diff) > 9999.D0) energy_diff = 0.D0
      if (prtpls .or. debug .and. niter > itrmax - 20) then
        write (line, "(' ITER.',i7,' PLS=', e10.3,10x,' ENERGY ',f13.5,' DELTAE',f13.7)")  &
        niter + add_niter, ovmax,   escf, energy_diff
        write(iw,"(a)")trim(line)
        call to_screen(line)
        if (use_disk) then
          endfile (iw)
          backspace (iw)
        end if
        if (debug) then
          write (iw, "(A,F9.6,A,F7.1,A,F9.6,A,F8.2,A,F11.3,A,I7)") "TINY:", &
               & tiny, " SUMT:", sumt, " OVMAX:", ovmax, " SUMB:", sumb, &
               & " DIFF:", energy_diff, " IJ:", ijc
        end if
      end if
      if (debug) then
        call chrge_for_MOZYME (p, ws)
        sum = 0.d0
        do i = 1, numat
          sum = sum + Abs (ws(i))
        end do
        write (iw, "(A,I4)") " Atomic Electron Population on Iteration:", &
               & niter
        write (iw, "(10F8.4)") (ws(i), i=1, numat)
        write (iw, "(A,F12.6)") " Variance:", sum
      end if
      if (use_disk) then
        endfile (iw)
        backspace (iw)
      end if
      if (resident_isitsc_done) then
        okscf = resident_isitsc_okscf
      else
        call mozyme_section_timer_begin('iter_isitsc', mozyme_timer)
        call isitsc (escf, selcon, emin, iemin, iemax, okscf, niter, itrmax)
        call mozyme_section_timer_end('iter_isitsc', mozyme_timer)
      end if
      if ( .not. bigscf .and. numcal == 1+numcal0) then
        exit
      else if (okscf .and. niter > 1 .and. (emin /= 0.d0 .or. niter > 3)) then
        exit
      end if
      if (use_three_point_extrap) then
        if (mod(niter,3) == 2 .and. Abs (energy_diff) < 0.1d0) then
          use_three_point_extrap = .false.
          lstart = niter
          shift = 0.0d0
        end if
      else
        if (energy_diff > 0.0d0 .and. shift < 11.0d0 .and. &
             & niter > lstart+2) then
          shift = shift + 2.0d0
          lstart = niter
        end if
      end if
    end do
!************************************************************
!
!   The SCF equations are now solved
!
!************************************************************
    if (istabl > 100000 .and. escf-emin > 200.d0 .and. panic) then
      write (iw, "(A)") " Something disastrous has happened.  " // &
                        & "The LMOs are probably corrupt."
      write (iw, "(A)") " The job should be restarted using RESTART but" // &
             & " OLDENS should NOT be used."
      write (iw, "(/10X,A,F16.6,/10X,A,F16.6)") &
             & " Value of previous Heat of Formation:", emin, &
             & " Value of current Heat of Formation: ", escf
        !
        !          Start the SHUT command.
        !
      inquire (file=end_fn, opened=opend)
      if (opend) then
        rewind (iend)
      else
        open (unit=iend, file=end_fn)
      end if
      write (iend, "(A)", err=1000) xchar
      go to 1010
        !
        !  The SHUT command is faulty.  Stop everything.
        !
1000  call geout (iw)
      call mopend ("Severe fault in RESTART")
1010  panic = .false.
    end if
    if (Abs (escf-emin) < 1.d0) then
      istabl = istabl + 1
    else
      istabl = 0
    end if
    if (escf < emin .or. emin == 0.d0) then
      emin = escf
    end if
    if (resident_density_current) then
      call mozyme_section_timer_begin('iter_density_final_resident', mozyme_timer)
      write(iw,'(1x,a)') '[MOZYME GPU SCF] final_density=current_resident'
      call flush(iw)
      call mozyme_section_timer_end('iter_density_final_resident', mozyme_timer)
    else if (resident_strict_required) then
      call mozyme_gpu_strict_abort('strict_missing_final_density', &
        'MOZYME GPU strict resident SCF ended without resident final density')
      return
    else if (.not. scf1) then
      if (numat > numred+1) then
        call mozyme_section_timer_begin('iter_density_final_partial', mozyme_timer)
        call density_for_MOZYME (p, 1, nocc1, partp)
        call mozyme_section_timer_end('iter_density_final_partial', mozyme_timer)
      else
        call mozyme_section_timer_begin('iter_density_final_full', mozyme_timer)
        call density_for_MOZYME (p, 0, nocc1, partp)
        call mozyme_section_timer_end('iter_density_final_full', mozyme_timer)
      end if
    end if
    icalcn = step_num
    imol = numcal
    if (resident_final_reorth_due) then
      if (resident_strict_required) then
        if (.not. resident_final_reorth_done) then
          call mozyme_gpu_strict_abort('strict_cpu_reorthogonalization', &
            'MOZYME GPU strict resident SCF did not complete resident reorthogonalization')
          return
        end if
      else
        call mozyme_section_timer_begin('iter_reorth', mozyme_timer)
        if (.not. mozyme_gpu_reorth_try()) then
          call reorth (ws)             !   Re-orthogonalize the LMO's
        end if
        call mozyme_section_timer_end('iter_reorth', mozyme_timer)
        call mozyme_section_timer_begin('iter_density_reorth', mozyme_timer)
        call density_for_MOZYME (p, 0, noccupied, partp)
        call mozyme_section_timer_end('iter_density_reorth', mozyme_timer)
        call mozyme_section_timer_begin('iter_buildf_reorth', mozyme_timer)
        call buildf (f, partf, 0)
        call mozyme_section_timer_end('iter_buildf_reorth', mozyme_timer)
        call mozyme_section_timer_begin('iter_helecz_reorth', mozyme_timer)
        ee = helecz ()
        call mozyme_section_timer_end('iter_helecz_reorth', mozyme_timer)
      end if
      escf = (ee+enuclr) * fpc_9 + atheat
      if (useps) then
            escf = escf + solv_energy * fpc_9
      end if
    end if
    nmol = numcal
    call mozyme_section_timer_report_all()
    return
    end subroutine iter_for_MOZYME

subroutine mozyme_gpu_strict_fallback_marker(reason)
  use chanel_C, only: iw
  implicit none
  character(len=*), intent(in) :: reason

  write(iw,'(1x,a,a)') '[MOZYME GPU SCF] status=strict_abort reason=', &
    trim(reason)
  call flush(iw)
end subroutine mozyme_gpu_strict_fallback_marker

subroutine mozyme_gpu_strict_abort(reason, message)
  use mozyme_section_timers, only : mozyme_section_timer_report_all
  implicit none
  character(len=*), intent(in) :: reason, message
  external :: mopend, mozyme_gpu_strict_fallback_marker

  call mozyme_gpu_strict_fallback_marker(reason)
  call mopend(message)
  call mozyme_section_timer_report_all()
  error stop 'MOZYME GPU strict resident SCF abort'
end subroutine mozyme_gpu_strict_abort

subroutine mozyme_gpu_grow_lmo_storage(status)
  use MOZYME_C, only : icocc, cocc, icvir, cvir, icocc_dim, cocc_dim, &
    icvir_dim, cvir_dim
  implicit none
  integer, intent(out) :: status
  integer :: new_icocc_dim, new_cocc_dim, new_icvir_dim, new_cvir_dim
  integer, allocatable :: new_icocc(:), new_icvir(:)
  double precision, allocatable :: new_cocc(:), new_cvir(:)

  status = 0
  if (.not. allocated(icocc) .or. .not. allocated(cocc) .or. &
      .not. allocated(icvir) .or. .not. allocated(cvir)) then
    status = 1
    return
  end if

  new_icocc_dim = max(icocc_dim + 1, nint(dble(icocc_dim) * 1.6d0))
  new_cocc_dim = max(cocc_dim + 1, nint(dble(cocc_dim) * 1.6d0))
  new_icvir_dim = max(icvir_dim + 1, nint(dble(icvir_dim) * 1.6d0))
  new_cvir_dim = max(cvir_dim + 1, nint(dble(cvir_dim) * 1.6d0))

  allocate(new_icocc(new_icocc_dim), new_cocc(new_cocc_dim), &
    new_icvir(new_icvir_dim), new_cvir(new_cvir_dim), stat=status)
  if (status /= 0) return

  new_icocc = 0
  new_cocc = 0.0d0
  new_icvir = 0
  new_cvir = 0.0d0
  new_icocc(:icocc_dim) = icocc(:icocc_dim)
  new_cocc(:cocc_dim) = cocc(:cocc_dim)
  new_icvir(:icvir_dim) = icvir(:icvir_dim)
  new_cvir(:cvir_dim) = cvir(:cvir_dim)

  call move_alloc(new_icocc, icocc)
  call move_alloc(new_cocc, cocc)
  call move_alloc(new_icvir, icvir)
  call move_alloc(new_cvir, cvir)
  icocc_dim = new_icocc_dim
  cocc_dim = new_cocc_dim
  icvir_dim = new_icvir_dim
  cvir_dim = new_cvir_dim
end subroutine mozyme_gpu_grow_lmo_storage
