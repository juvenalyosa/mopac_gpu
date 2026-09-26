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

subroutine diagg1 (fao, nocc, nvir, eigv, ws, latoms, ifmo, fmo, fmo_dim, nij, idiagg,  avir, aocc, aov)
   !**********************************************************************
   !
   !  FAO:      FOCK MATRIX OVER ATOMIC ORBITALS
   !  FMO:      FOCK MATRIX OVER MOLECULAR ORBITALS
   !  VECTOR:   EIGENVECTORS OF M.O.S
   !  FILLED:   OCCUPIED M.O.S
   !  eigs:     EIGENVALUES OF OCCUPIED M.O.S
   !  EMPTY:    VIRTUAL M.O.S
   !  EIGV:     EIGENVALUES OF VIRTUAL M.O.S
   !  NOCC:     NUMBER OF OCCUPIED M.O.S
   !  EIG:      ALL THE EIGENVALUES
   !  N:        NUMBER OF M.O.S = NUMBER OF A.O.S*
   !
   !**********************************************************************
   !
   !  ALL OCCUPIED AND UNOCCUPIED LMO ENERGY LEVELS ARE CALCULATED, AS WELL
   !  AS ALL MATRIX ELEMENTS WHICH CAN BE NON ZERO CONNECTING THE OCCUPIED
   !  AND VIRTUAL SETS OF LMO'S.
   !
   !**********************************************************************
   !
    use molkst_C, only: numat, norbs, mpack, numcal, keywrd
    use MOZYME_C, only : nvirtual, icocc_dim, &
     & nfmo, lijbo, nijbo, &
       tiny, sumt, ijc, ovmax, ncf, nce, nncf, nnce, ncocc, ncvir, &
     & iorbs, icocc, icvir, cocc, cvir, cocc_dim, cvir_dim, icvir_dim
    use common_arrays_C, only : eigs, nfirst, nlast, p
    use mozyme_diagg1_state, only: nf => diagg1_nf, &
     & icalcn => diagg1_icalcn, mydisp => diagg1_mydisp, &
     & ij0 => diagg1_ij0, fref => diagg1_fref, &
     & oldlim => diagg1_oldlim, safety => diagg1_safety
#ifdef GPU
    use iso_c_binding, only: c_int, c_double
    use chanel_C, only: iw
    use mozyme_gpu_int_utils, only: mozyme_c_int_checked, &
      mozyme_c_int_nonnegative_or_zero, mozyme_c_int_positive_or_zero
    use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required
    use mod_vars_cuda, only: lgpu, mozyme_gpu
#endif
    implicit none
    integer, intent (in) :: idiagg, nocc, nvir, fmo_dim
    integer, intent (inout) :: nij
    logical, dimension (numat), intent (out) :: latoms
    integer, dimension (2, fmo_dim), intent (inout) :: ifmo
    double precision, dimension (fmo_dim), intent (out) :: fmo
    double precision, dimension (numat), intent (out) :: aov
    double precision, dimension (norbs), intent (out) :: avir, ws
    double precision, dimension (mpack), intent (in) :: fao
    double precision, dimension (nvirtual), intent (inout) :: eigv
    double precision, dimension (icocc_dim), intent (out) :: aocc
    integer :: i, i1, i2, i4, ii, j, j1, j2, j4, &
         & jj, jl, jx, k, k1, kk, kl, l, loopi, loopj, kj, i1j1, &
         & i1j2, i2j1, i2j2, k1j1
    logical :: lij
    integer, dimension (:), allocatable :: nbfirst, nbatom, nboff
    double precision, dimension (:), allocatable :: nbp
    logical, save :: times
    double precision :: cutlim, flim
    double precision :: cutoff, sum, sum1
    integer, external :: ijbo
#ifdef GPU
    logical :: aocc_gpu_done, avir_gpu_done
    integer(c_int) :: gpu_code, gpu_updates, gpu_nij, gpu_ijc, gpu_nf
    integer :: gpu_alloc_stat
    real(c_double) :: gpu_sumt, gpu_tiny, gpu_fref, gpu_oldlim
    real(c_double) :: gpu_safety, gpu_wall_ms
    double precision, allocatable :: avir_cache(:)

    interface
      function mopac_cuda_mozyme_diagg1_construct(nocc_c, nvir_c, numat_c, &
          norbs_c, mpack_c, icocc_dim_c, icvir_dim_c, cocc_dim_c, &
          cvir_dim_c, fmo_dim_c, idiagg_c, mydisp_c, fao_c, p_c, &
          nfirst_c, nlast_c, ncf_c, nce_c, nncf_c, nnce_c, ncocc_c, &
          ncvir_c, icocc_c, icvir_c, iorbs_c, nijbo_c, cocc_c, cvir_c, &
          cutoff_c, flim_c, oldlim_c, safety_c, nf_c, eigs_c, eigv_c, &
          nfmo_c, ifmo_c, fmo_c, nij_c, ijc_c, nf_out_c, sumt_c, tiny_c, &
          fref_c, oldlim_out_c, safety_out_c, wall_ms_c) &
          bind(C,name='mopac_cuda_mozyme_diagg1_construct') result(code)
        import :: c_int, c_double
        integer(c_int), value :: nocc_c, nvir_c, numat_c, norbs_c, mpack_c
        integer(c_int), value :: icocc_dim_c, icvir_dim_c
        integer(c_int), value :: cocc_dim_c, cvir_dim_c, fmo_dim_c
        integer(c_int), value :: idiagg_c, mydisp_c, nf_c
        real(c_double), intent(in) :: fao_c(*), p_c(*)
        integer(c_int), intent(in) :: nfirst_c(*), nlast_c(*)
        integer(c_int), intent(in) :: ncf_c(*), nce_c(*), nncf_c(*), nnce_c(*)
        integer(c_int), intent(in) :: ncocc_c(*), ncvir_c(*)
        integer(c_int), intent(in) :: icocc_c(*), icvir_c(*), iorbs_c(*)
        integer(c_int), intent(in) :: nijbo_c(*)
        real(c_double), intent(in) :: cocc_c(*), cvir_c(*)
        real(c_double), value :: cutoff_c, flim_c, oldlim_c, safety_c
        real(c_double) :: eigs_c(*), eigv_c(*), fmo_c(*)
        integer(c_int) :: nfmo_c(*), ifmo_c(*)
        integer(c_int) :: nij_c, ijc_c, nf_out_c
        real(c_double) :: sumt_c, tiny_c, fref_c, oldlim_out_c
        real(c_double) :: safety_out_c, wall_ms_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_diagg1_construct

      function mopac_cuda_mozyme_diagg1_aocc(nocc_c, icocc_dim_c, cocc_dim_c, &
          numat_c, ncf_c, nncf_c, ncocc_c, icocc_c, iorbs_c, cocc_c, aocc_c, &
          updated_terms_c, wall_ms_c) bind(C,name='mopac_cuda_mozyme_diagg1_aocc') result(code)
        import :: c_int, c_double
        integer(c_int), value :: nocc_c, icocc_dim_c, cocc_dim_c, numat_c
        integer(c_int), intent(in) :: ncf_c(*), nncf_c(*), ncocc_c(*), icocc_c(*), iorbs_c(*)
        real(c_double), intent(in) :: cocc_c(*)
        real(c_double) :: aocc_c(*), wall_ms_c
        integer(c_int) :: updated_terms_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_diagg1_aocc

      function mopac_cuda_mozyme_diagg1_avir(nvir_c, icvir_dim_c, cvir_dim_c, &
          numat_c, nce_c, nnce_c, ncvir_c, icvir_c, iorbs_c, cvir_c, avir_cache_c, &
          updated_terms_c, wall_ms_c) bind(C,name='mopac_cuda_mozyme_diagg1_avir') result(code)
        import :: c_int, c_double
        integer(c_int), value :: nvir_c, icvir_dim_c, cvir_dim_c, numat_c
        integer(c_int), intent(in) :: nce_c(*), nnce_c(*), ncvir_c(*), icvir_c(*), iorbs_c(*)
        real(c_double), intent(in) :: cvir_c(*)
        real(c_double) :: avir_cache_c(*), wall_ms_c
        integer(c_int) :: updated_terms_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_diagg1_avir
    end interface
#endif
    times = (Index (keywrd, " TIMES") /= 0)
    if (numcal /= icalcn) then
      icalcn = numcal
      fref = 10.0d0
      safety = 1.0d0
      oldlim = 0.0d0
      nf = 0
      if (Index (keywrd, " OLDENS") /= 0) then
        fref = 0.d0
      end if
    end if
    !
    !    CUTLIM    PRECISION OF PL
    !
    !    1.D-6      0.004
    !    1.D-7      0.00004
    !
    !   A TERM OF SIZE SQRT(CUTOFF) IS LEFT OUT OF EACH OCCUPIED - VIRTUAL
    !   INTERACTION.  NEAR SELF-CONSISTENCY THAT MUST STAY WELL BELOW THE
    !   LARGEST INTERACTION (OVMAX, FROM THE LAST ITERATION), OR THE LEFT-OUT
    !   TERMS, WHICH CHANGE AS THE LMOS ROTATE, SET A FLOOR TO THE SCF.
    !
    cutlim = 1.d-8
    if (ovmax > 0.d0) cutlim = Min (cutlim, Max ((1.d-2*ovmax)**2, 1.d-20))
    cutoff = Max (cutlim, tiny*10.d0*cutlim)
    flim = Min (3.d0, fref*0.5d0)
    fref = 0.d0
    if (idiagg <= 5) then
      cutoff = cutlim
    end if
#ifdef GPU
    if (mozyme_diagg1_construct_gpu_enabled()) then
      if (.not. lijbo) then
        if (mozyme_diagg1_aocc_trace()) then
          write(iw,'(1x,a," fallback_cpu code=",i0," reason=no_nijbo")') &
            '[MOZYME GPU diagg1_construct]', -2
          call flush(iw)
        end if
      else
        gpu_nij = mozyme_c_int_nonnegative_or_zero(nij)
        gpu_ijc = 0_c_int
        gpu_nf = mozyme_c_int_nonnegative_or_zero(nf)
        gpu_sumt = 0.0_c_double
        gpu_tiny = 0.0_c_double
        gpu_fref = 0.0_c_double
        gpu_oldlim = oldlim
        gpu_safety = safety
        gpu_wall_ms = 0.0_c_double
        gpu_code = mopac_cuda_mozyme_diagg1_construct( &
          mozyme_c_int_nonnegative_or_zero(nocc), &
          mozyme_c_int_nonnegative_or_zero(nvir), &
          mozyme_c_int_positive_or_zero(numat), &
          mozyme_c_int_positive_or_zero(norbs), &
          mozyme_c_int_positive_or_zero(mpack), &
          mozyme_c_int_positive_or_zero(icocc_dim), &
          mozyme_c_int_positive_or_zero(icvir_dim), &
          mozyme_c_int_positive_or_zero(cocc_dim), &
          mozyme_c_int_positive_or_zero(cvir_dim), &
          mozyme_c_int_positive_or_zero(fmo_dim), &
          mozyme_c_int_checked(idiagg), mozyme_c_int_checked(mydisp), &
          fao, p, nfirst, nlast, &
          ncf, nce, nncf, nnce, ncocc, ncvir, icocc, icvir, iorbs, nijbo, &
          cocc, cvir, cutoff, flim, oldlim, safety, &
          mozyme_c_int_nonnegative_or_zero(nf), eigs, &
          eigv, nfmo, ifmo, fmo, gpu_nij, gpu_ijc, gpu_nf, gpu_sumt, &
          gpu_tiny, gpu_fref, gpu_oldlim, gpu_safety, gpu_wall_ms)
        if (gpu_code == 0_c_int) then
          nij = int(gpu_nij)
          ijc = int(gpu_ijc)
          nf = int(gpu_nf)
          sumt = gpu_sumt
          tiny = gpu_tiny
          ovmax = tiny
          fref = gpu_fref
          oldlim = gpu_oldlim
          safety = gpu_safety
          if (mozyme_diagg1_aocc_trace()) then
            write(iw,'(1x,a," success code=",i0," nij=",i0," sumt=",es13.6,&
              &" tiny=",es13.6," ms=",f12.6)') &
              '[MOZYME GPU diagg1_construct]', int(gpu_code), nij, sumt, &
              tiny, gpu_wall_ms
            call flush(iw)
          end if
          if (times) then
            call timer (" AFTER DIAGG1 IN ITER")
          end if
          return
        else if (mozyme_diagg1_aocc_trace()) then
          write(iw,'(1x,a," fallback_cpu code=",i0)') &
            '[MOZYME GPU diagg1_construct]', int(gpu_code)
          call flush(iw)
        end if
      end if
    end if
    if (mozyme_gpu_scf_no_fallback_required()) then
      write(iw,'(1x,a)') &
        '[MOZYME GPU SCF] status=strict_abort reason=strict_diagg1_cpu_fallback'
      call flush(iw)
      error stop 'MOZYME GPU strict diagg1 abort'
    end if
#endif
    !
    !
    aocc(:) = 0.d0
#ifdef GPU
    aocc_gpu_done = .false.
    if (mozyme_diagg1_aocc_gpu_enabled()) then
      gpu_updates = 0_c_int
      gpu_code = mopac_cuda_mozyme_diagg1_aocc( &
        mozyme_c_int_nonnegative_or_zero(nocc), &
        mozyme_c_int_positive_or_zero(icocc_dim), &
        mozyme_c_int_positive_or_zero(cocc_dim), &
        mozyme_c_int_positive_or_zero(numat), ncf, nncf, ncocc, icocc, iorbs, &
        cocc, aocc, gpu_updates, gpu_wall_ms)
      if (gpu_code == 0_c_int) then
        aocc_gpu_done = .true.
        if (mozyme_diagg1_aocc_trace()) then
          write(iw,'(1x,a," success code=",i0," terms=",i0," ms=",f10.3)') &
            '[MOZYME GPU diagg1_aocc]', int(gpu_code), int(gpu_updates), gpu_wall_ms
          call flush(iw)
        end if
      else if (mozyme_diagg1_aocc_trace()) then
        write(iw,'(1x,a," fallback_cpu code=",i0)') '[MOZYME GPU diagg1_aocc]', int(gpu_code)
        call flush(iw)
      end if
    end if
    if (.not. aocc_gpu_done) then
#endif
    !
    !   IF THE CONTRIBUTION OF AN ATOM IN AN OCCUPIED LMO IS VERY SMALL
    !   THEN DO NOT USE THAT ATOM IN CALCULATING THE OCCUPIED-VIRTUAL
    !   INTERACTION.  PUT THE CONTRIBUTIONS INTO AN ARRAY 'AOCC'.
    !
    do j = 1, nocc
        loopj = ncocc(j)
        kl = 0
        do kk = nncf(j) + 1, nncf(j) + ncf(j)
          k1 = icocc(kk)
          sum = 0.d0
          do k = nfirst(k1), nlast(k1)
            kl = kl + 1
            sum = sum + cocc(kl+loopj) ** 2
          end do
          !
          !   AOCC(KK) HOLDS THE SQUARE OF THE CONTRIBUTION OF THE KK'TH ATOM
          !   IN THE OCCUPIED SET.  NOTE:  THIS IS NOT ATOM KK.
          !
          aocc(kk) = sum
          !
        end do
      !
    end do
#ifdef GPU
    end if
    avir_gpu_done = .false.
    if (mozyme_diagg1_avir_gpu_enabled()) then
      allocate(avir_cache(max(1, icvir_dim)), stat=gpu_alloc_stat)
      if (gpu_alloc_stat == 0) then
        avir_cache(:) = 0.0d0
        gpu_updates = 0_c_int
        gpu_code = mopac_cuda_mozyme_diagg1_avir( &
          mozyme_c_int_nonnegative_or_zero(nvir), &
          mozyme_c_int_positive_or_zero(icvir_dim), &
          mozyme_c_int_positive_or_zero(cvir_dim), &
          mozyme_c_int_positive_or_zero(numat), nce, nnce, ncvir, icvir, iorbs, &
          cvir, avir_cache, gpu_updates, gpu_wall_ms)
        if (gpu_code == 0_c_int) then
          avir_gpu_done = .true.
          if (mozyme_diagg1_aocc_trace()) then
            write(iw,'(1x,a," success code=",i0," terms=",i0," ms=",f10.3)') &
              '[MOZYME GPU diagg1_avir]', int(gpu_code), int(gpu_updates), gpu_wall_ms
            call flush(iw)
          end if
        else
          if (mozyme_diagg1_aocc_trace()) then
            write(iw,'(1x,a," fallback_cpu code=",i0)') '[MOZYME GPU diagg1_avir]', int(gpu_code)
            call flush(iw)
          end if
          deallocate(avir_cache)
        end if
      end if
    end if
#endif
    sumt = 0.d0
    ijc = 0
    tiny = 0.d0
    !
    !   ATOM NEIGHBOUR LISTS, SORTED BY THE SIZE OF THE FOCK BLOCK
    !
    allocate (nbfirst(numat+1), nbatom(1), nboff(1), nbp(1))
    call diagg1_neighbours (0, cutoff, nbfirst, nbatom, nboff, nbp)
    k = Max (nbfirst(numat+1) - 1, 1)
    deallocate (nbatom, nboff, nbp)
    allocate (nbatom(k), nboff(k), nbp(k))
    call diagg1_neighbours (1, cutoff, nbfirst, nbatom, nboff, nbp)
    !
    do i = 1, nvir
      !
      loopi = ncvir(i)
      latoms(:) = .false.
      l = 0
      do j = nnce(i) + 1, nnce(i) + nce(i)
        j1 = icvir(j)
#ifdef GPU
        if (avir_gpu_done) then
          sum = avir_cache(j)
        else
#endif
        sum = 0.d0
        do k = l + 1, l + iorbs(j1)
          sum = sum + cvir(k+loopi) ** 2
        end do
#ifdef GPU
        end if
#endif
        l = l + iorbs(j1)
        !
        !   AVIR(J1) HOLDS THE SQUARE OF THE CONTRIBUTION OF THE ATOM J1
        !   IN THE VIRTUAL LMO 'I'.
        !
        avir(j1) = sum
        latoms(icvir(j)) = .true.
      end do
      !
      !
      !   WS = FAO * (VIRTUAL LMO I), OVER THE ATOMS OF LMO I AND OVER EVERY
      !   ATOM OUTSIDE IT THAT HAS A SIGNIFICANT FOCK BLOCK WITH AN ATOM OF
      !   LMO I.  THE OCCUPIED - VIRTUAL INTERACTION <I|F|J> RUNS OVER EVERY
      !   ATOM OF THE OCCUPIED LMO J: IF THE ATOMS OUTSIDE LMO I WERE LEFT
      !   OUT, THE INTERACTION WOULD DEPEND ON WHETHER A BOUNDARY ATOM OF
      !   LMO J HAPPENED TO BE IN LMO I, WHICH CHANGES WHEN DIAGG2 ADDS THE
      !   ATOM AND TIDY REMOVES IT AGAIN, AND THE SCF WOULD CYCLE WITHOUT
      !   CONVERGING.  LATOMS MARKS THE ATOMS WHERE WS IS SET.
      !
      do jj = nnce(i) + 1, nnce(i) + nce(i)
        j1 = icvir(jj)
        do jx = nfirst(j1), nlast(j1)
          ws(jx) = 0.d0
        end do
      end do
      kl = loopi
      do kk = nnce(i) + 1, nnce(i) + nce(i)
        k1 = icvir(kk)
        call diagg1_ws (fao, k1, kl, avir(k1), cutoff, nbfirst, nbatom, nboff, nbp, ws, latoms)
        kl = kl + iorbs(k1)
      end do
      !
      do j = 1, numat
        if (latoms(j)) then
          sum = 0.d0
          do k = nfirst(j), nlast(j)
            sum = sum + ws(k) ** 2
          end do
          !
          !   AOV(J) HOLDS THE SQUARE OF THE ENERGY CONTRIBUTION OF THE
          !   ATOM J IN THE VIRTUAL LMO 'I'.
          !
          aov(j) = sum
        else
          aov(j) = 0.d0
        end if
      end do
      !
      !   EVALUATE THE VIRTUAL ENERGY LEVELS
      !
      sum = 0.d0
      kl = loopi
      do kk = nnce(i) + 1, nnce(i) + nce(i)
        k1 = icvir(kk)
        if (aov(k1)*avir(k1) > cutoff) then
          do k = nfirst(k1), nlast(k1)
            kl = kl + 1
            sum = sum + ws(k) * cvir(kl)
          end do
        else
          kl = kl + iorbs(k1)
        end if
      end do
      !
      eigv(i) = sum
      !
      !  EVALUATE THE OCCUPIED-VIRTUAL LMO INTERACTION ENERGIES
      !
      if (idiagg <= 5 .or. Mod (idiagg, 2) == 0) then
        nf = 0
        i1 = icvir(nnce(i)+1)
        if (nce(i) > 1) then
          i2 = icvir(nnce(i)+2)
        else
          i2 = i1
        end if
        if (lijbo) then
          do j = 1, nocc
            if (ijc /= nij) then
              !
              !  FAST TEST TO SEE IF THE INTEGRAL IS WORTH EVALUATING
              !
              j1 = icocc(nncf(j)+1)
              if (ncf(j) > 1) then
                j2 = icocc(nncf(j)+2)
              else
                j2 = j1
              end if
              !
              i1j1 = nijbo(i1, j1)
              i1j2 = nijbo(i1, j2)
              i2j1 = nijbo(i2, j1)
              i2j2 = nijbo(i2, j2)
              !
              if (i1j1 >= 0 .or. i1j2 >= 0 .or. i2j1 >= 0 .or. i2j2 >= 0) then
                sum = 0.d0
                if (i1j1 >= 0) then
                  sum = Abs (fao(i1j1+1))
                end if
                if (i1j2 >= 0) then
                  sum = sum + Abs (fao(i1j2+1))
                end if
                if (i2j1 >= 0) then
                  sum = sum + Abs (fao(i2j1+1))
                end if
                if (i2j2 >= 0) then
                  sum = sum + Abs (fao(i2j2+1))
                end if
                if (sum >= flim) then
                  loopj = ncocc(j)
                  lij = .false.
                  sum = 0.d0
                  kl = 0
                  do kk = nncf(j) + 1, nncf(j) + ncf(j)
                    k1 = icocc(kk)
                    if (aocc(kk)*aov(k1) < cutoff .or. .not. latoms(k1)) then
                      kl = kl + iorbs(k1)
                    else
                      lij = .true.
                      do k = nfirst(k1), nlast(k1)
                        kl = kl + 1
                        sum = sum + ws(k) * cocc(kl+loopj)
                      end do
                    end if
                  end do
                  sumt = sumt + Abs (sum)
                  tiny = Max (tiny, Abs (sum))
                  if (lij) then
                    if (Abs (sum) > oldlim) then
                      nf = nf + 1
                      ijc = ijc + 1
                      ifmo(1, ijc) = i
                      ifmo(2, ijc) = j
                      fmo(ijc) = sum
                    end if
                  end if
                end if
              end if
            end if
          end do
        else
          do j = 1, nocc
            if (ijc /= nij) then
              !
              !  FAST TEST TO SEE IF THE INTEGRAL IS WORTH EVALUATING
              !
              j1 = icocc(nncf(j)+1)
              if (ncf(j) > 1) then
                j2 = icocc(nncf(j)+2)
              else
                j2 = j1
              end if
              !
              i1j1 = ijbo (i1, j1)
              i1j2 = ijbo (i1, j2)
              i2j1 = ijbo (i2, j1)
              i2j2 = ijbo (i2, j2)
              !
              if (i1j1 >= 0 .or. i1j2 >= 0 .or. i2j1 >= 0 .or. i2j2 >= 0) &
                   & then
                sum = 0.d0
                if (i1j1 >= 0) then
                  sum = Abs (fao(i1j1+1))
                end if
                if (i1j2 >= 0) then
                  sum = sum + Abs (fao(i1j2+1))
                end if
                if (i2j1 >= 0) then
                  sum = sum + Abs (fao(i2j1+1))
                end if
                if (i2j2 >= 0) then
                  sum = sum + Abs (fao(i2j2+1))
                end if
                if (sum >= flim) then
                  loopj = ncocc(j)
                  lij = .false.
                  sum = 0.d0
                  kl = 0
                  do kk = nncf(j) + 1, nncf(j) + ncf(j)
                    k1 = icocc(kk)
                    if (aocc(kk)*aov(k1) < cutoff .or. .not. latoms(k1)) then
                      kl = kl + iorbs(k1)
                    else
                      lij = .true.
                      do k = nfirst(k1), nlast(k1)
                        kl = kl + 1
                        sum = sum + ws(k) * cocc(kl+loopj)
                      end do
                    end if
                  end do
                  sumt = sumt + Abs (sum)
                  tiny = Max (tiny, Abs (sum))
                  if (lij) then
                    if (Abs (sum) > oldlim) then
                      nf = nf + 1
                      ijc = ijc + 1
                      ifmo(1, ijc) = i
                      ifmo(2, ijc) = j
                      fmo(ijc) = sum
                    end if
                  end if
                end if
              end if
            end if
          end do
        end if
        nfmo(i) = nf
      else
        ij0 = ijc + mydisp
        do jj = 1, nfmo(i)
          if (ijc+mydisp == nij) exit
          j = ifmo(2, ij0+jj)
          loopj = ncocc(j)
          sum = 0.d0
          kl = 0
          do kk = nncf(j) + 1, nncf(j) + ncf(j)
            k1 = icocc(kk)
            if (aov(k1)*aocc(kk) < cutoff .or. .not. latoms(k1)) then
              kl = kl + iorbs(k1)
            else
              do k = nfirst(k1), nlast(k1)
                kl = kl + 1
                sum = sum + ws(k) * cocc(kl+loopj)
              end do
            end if
          end do
          sumt = sumt + Abs (sum)
          tiny = Max (tiny, Abs (sum))
          ijc = ijc + 1
          fmo(ijc) = sum
        end do
      end if
    end do
    deallocate (nbfirst, nbatom, nboff, nbp)
    !
    if (ijc == nij) then
      !
      !   THERE WAS NOT ENOUGH STORAGE TO HOLD ALL THE INTEGRALS.
      !   THEREFORE, ON THE NEXT ITERATION, CALCULATE FEWER INTEGRALS.
      !
      safety = safety * 2.d0
    else
      !
      !  THERE IS ENOUGH STORAGE FOR ALL THE INTEGRALS.  IF NECESSARY,
      !  CALCULATE MORE INTEGRALS.
      !
      safety = Max (safety*0.5d0, 1.d0)
    end if

    nij = ijc
    if (idiagg > 2 .and. Mod (idiagg, 4) /= 0) then
      fref = tiny ** 4
      if (times) then
        call timer (" AFTER DIAGG1 IN ITER")
      end if
        !
    else
      !
      !  EVALUATE THE OCCUPIED ENERGY LEVELS
      !
        !
      do i = 1, nocc
        !
          loopi = ncocc(i)
          l = 0
          do j = nncf(i) + 1, nncf(i) + ncf(i)
            j1 = icocc(j)
            sum = 0.d0
            do k = l + 1, l + iorbs(j1)
              sum = sum + cocc(k+loopi) ** 2
            end do
            l = l + iorbs(j1)
            !
            !   AOCC(KK) HOLDS THE SQUARE OF THE CONTRIBUTION OF THE KK'TH ATOM
            !   IN THE OCCUPIED SET.  NOTE:  THIS IS NOT ATOM KK.
            !
            !   AVIR(J1) HOLDS THE SQUARE OF THE CONTRIBUTION OF ATOM J1
            !   IN THE OCCUPIED SET.  NOTE:  THIS IS NOT THE J1'th ATOM.
            !
            avir(j1) = sum
            !
            !   AOCC(KK) HOLDS THE SQUARE OF THE CONTRIBUTION OF THE KK'TH ATOM
            !   IN THE OCCUPIED SET.  NOTE:  THIS IS NOT ATOM KK.
            !
            !   AVIR(J1) HOLDS THE SQUARE OF THE CONTRIBUTION OF ATOM J1
            !   IN THE OCCUPIED SET.  NOTE:  THIS IS NOT THE J1'th ATOM.
            !
          end do
          sum = 0.d0
          jl = loopi
          !
          if (lijbo) then
            do j = nncf(i) + 1, nncf(i) + ncf(i)
              j1 = icocc(j)
              kl = loopi
              do k = nncf(i) + 1, nncf(i) + ncf(i)
                k1 = icocc(k)
                k1j1 = nijbo(k1, j1)
                if (k1j1 >= 0) then
                  if (avir(k1)*p(k1j1+1)*aocc(k) >= cutoff) then
                    !
                    !  EXTRACT THE ATOM-ATOM INTERSECTION OF FAO
                    !
                    if (k1 > j1) then
                      !
                      !   LOWER TRIANGLE
                      !
                      do jx = 1, iorbs(j1)
                        sum1 = 0.d0
                        do i4 = 1, iorbs(k1)
                          ii = k1j1 + (i4-1) * iorbs(j1) + jx
                          sum1 = sum1 + fao(ii) * cocc(kl+i4)
                        end do
                        sum = sum + cocc(jl+jx) * sum1
                      end do
                    else if (k1 < j1) then
                      !
                      !   UPPER TRIANGLE
                      !
                      do jx = 1, iorbs(j1)
                        sum1 = 0.d0
                        do i4 = 1, iorbs(k1)
                          ii = k1j1 + (jx-1) * iorbs(k1) + i4
                          sum1 = sum1 + fao(ii) * cocc(kl+i4)
                        end do
                        sum = sum + cocc(jl+jx) * sum1
                      end do
                    else
                      !
                      !   DIAGONAL TERM
                      !
                      do jx = 1, iorbs(j1)
                        sum1 = 0.d0
                        do j4 = 1, jx
                          ii = k1j1 + (jx*(jx-1)) / 2 + j4
                          sum1 = sum1 + fao(ii) * cocc(kl+j4)
                        end do
                        ii = k1j1 + (jx*(jx+1)) / 2
                        do i4 = jx + 1, iorbs(k1)
                          ii = k1j1 + (i4*(i4-1)) / 2 + jx
                          sum1 = sum1 + fao(ii) * cocc(kl+i4)
                        end do
                        sum = sum + cocc(jl+jx) * sum1
                      end do
                    end if
                  end if
                end if
                kl = kl + iorbs(k1)
              end do
              !
              jl = jl + iorbs(j1)
            end do
          else
            do j = nncf(i) + 1, nncf(i) + ncf(i)
              j1 = icocc(j)
              kl = loopi
              do k = nncf(i) + 1, nncf(i) + ncf(i)
                k1 = icocc(k)
                k1j1 = ijbo (k1, j1)
                if (k1j1 >= 0) then
                  if (avir(k1)*p(k1j1+1)*aocc(k) >= cutoff) then
                    !
                    !  EXTRACT THE ATOM-ATOM INTERSECTION OF FAO
                    !
                    if (k1 > j1) then
                      !
                      !   LOWER TRIANGLE
                      !
                      do jx = 1, iorbs(j1)
                        sum1 = 0.d0
                        do i4 = 1, iorbs(k1)
                          ii = k1j1 + (i4-1) * iorbs(j1) + jx
                          sum1 = sum1 + fao(ii) * cocc(kl+i4)
                        end do
                        sum = sum + cocc(jl+jx) * sum1
                      end do
                    else if (k1 < j1) then
                      !
                      !   UPPER TRIANGLE
                      !
                      do jx = 1, iorbs(j1)
                        sum1 = 0.d0
                        do i4 = 1, iorbs(k1)
                          ii = k1j1 + (jx-1) * iorbs(k1) + i4
                          sum1 = sum1 + fao(ii) * cocc(kl+i4)
                        end do
                        sum = sum + cocc(jl+jx) * sum1
                      end do
                    else
                      !
                      !   DIAGONAL TERM
                      !
                      do jx = 1, iorbs(j1)
                        sum1 = 0.d0
                        do j4 = 1, jx
                          ii = k1j1 + (jx*(jx-1)) / 2 + j4
                          sum1 = sum1 + fao(ii) * cocc(kl+j4)
                        end do
                        ii = k1j1 + (jx*(jx+1)) / 2
                        do i4 = jx + 1, iorbs(k1)
                          ii = k1j1 + (i4*(i4-1)) / 2 + jx
                          sum1 = sum1 + fao(ii) * cocc(kl+i4)
                        end do
                        sum = sum + cocc(jl+jx) * sum1
                      end do
                    end if
                  end if
                end if
                kl = kl + iorbs(k1)
              end do
              !
              jl = jl + iorbs(j1)
            end do
          end if
          !
          eigs(i) = sum
          !
      end do
            oldlim = tiny * safety * 1.d-3
      fref = tiny ** 4
      if (times) then
        call timer (" AFTER DIAGG1 IN ITER")
      end if
    end if

    ovmax = tiny
#ifdef GPU
    if (avir_gpu_done) deallocate(avir_cache)
#endif
#ifdef GPU
contains
  logical function mozyme_diagg1_construct_gpu_enabled()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_diagg1_construct_gpu_enabled = .false.
    if (.not. (lgpu .and. mozyme_gpu)) return
    env_value = ' '
    call get_environment_variable('MOPAC_MOZYME_DIAGG1_CONSTRUCT_GPU', &
      env_value, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0) then
      select case (trim(env_value))
      case ('0', 'off', 'OFF', 'false', 'FALSE', 'no', 'NO')
        mozyme_diagg1_construct_gpu_enabled = .false.
      case default
        mozyme_diagg1_construct_gpu_enabled = .true.
      end select
    end if
  end function mozyme_diagg1_construct_gpu_enabled

  logical function mozyme_diagg1_aocc_gpu_enabled()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_diagg1_aocc_gpu_enabled = .false.
    ! Opt-in (2026-09-18): these per-call helpers predate the resident SCF, are
    ! slower than the CPU code they replace and at least two of them (diagg1_avir
    ! and another one) give wrong energies in the CPU fallback loop after a
    ! resident hand-back (crambin: SCF failed / -2899.57 vs -2901.68).  The
    ! resident SCF has its own kernels; set the variable to 1 to use this helper.
    if (.not. (lgpu .and. mozyme_gpu)) return
    env_value = ' '
    call get_environment_variable('MOPAC_MOZYME_DIAGG1_AOCC_GPU', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0) then
      select case (trim(env_value))
      case ('1', 'on', 'ON', 'true', 'TRUE', 'yes', 'YES')
        mozyme_diagg1_aocc_gpu_enabled = .true.
      end select
    end if
  end function mozyme_diagg1_aocc_gpu_enabled

  logical function mozyme_diagg1_avir_gpu_enabled()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_diagg1_avir_gpu_enabled = .false.
    ! Opt-in (2026-09-18): these per-call helpers predate the resident SCF, are
    ! slower than the CPU code they replace and at least two of them (diagg1_avir
    ! and another one) give wrong energies in the CPU fallback loop after a
    ! resident hand-back (crambin: SCF failed / -2899.57 vs -2901.68).  The
    ! resident SCF has its own kernels; set the variable to 1 to use this helper.
    if (.not. (lgpu .and. mozyme_gpu)) return
    env_value = ' '
    call get_environment_variable('MOPAC_MOZYME_DIAGG1_AVIR_GPU', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0) then
      select case (trim(env_value))
      case ('1', 'on', 'ON', 'true', 'TRUE', 'yes', 'YES')
        mozyme_diagg1_avir_gpu_enabled = .true.
      end select
    end if
  end function mozyme_diagg1_avir_gpu_enabled

  logical function mozyme_diagg1_aocc_trace()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_diagg1_aocc_trace = .false.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_PROFILE', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') &
      mozyme_diagg1_aocc_trace = .true.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_VERBOSE', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') &
      mozyme_diagg1_aocc_trace = .true.
  end function mozyme_diagg1_aocc_trace
#endif
end subroutine diagg1
subroutine diagg1_neighbours (mode, cutoff, nbfirst, nbatom, nboff, nbp)
   !**********************************************************************
   !
   !  For every atom J, list the atoms K that share a Fock block with J whose
   !  square (from EIMP, in P) exceeds CUTOFF: NBATOM(NBFIRST(J):NBFIRST(J+1)-1),
   !  with the block address NBOFF (as IJBO) and the square NBP, sorted by
   !  decreasing NBP, so that a search for significant blocks stops early.
   !
   !  MODE = 0: only NBFIRST is set (the lists need NBFIRST(NUMAT+1) - 1
   !            elements); MODE = 1: the lists are filled.
   !
   !**********************************************************************
    use molkst_C, only: numat
    use MOZYME_C, only : iorbs
    use common_arrays_C, only : p
    implicit none
    integer, intent (in) :: mode
    double precision, intent (in) :: cutoff
    integer, dimension (numat+1), intent (inout) :: nbfirst
    integer, dimension (*), intent (inout) :: nbatom, nboff
    double precision, dimension (*), intent (inout) :: nbp
    integer :: gap, i, j, k, kj, m, n, ia, io
    integer, dimension (:), allocatable :: nfill
    double precision :: pa
    integer, external :: ijbo
    allocate (nfill(numat))
    nfill(:) = 0
    if (mode == 0) then
      do j = 1, numat
        do k = 1, j - 1
          kj = ijbo (j, k)
          if (kj >= 0) then
            if (iorbs(j)*iorbs(k) /= 0 .and. p(kj+1) > cutoff) then
              nfill(j) = nfill(j) + 1
              nfill(k) = nfill(k) + 1
            end if
          end if
        end do
      end do
      nbfirst(1) = 1
      do j = 1, numat
        nbfirst(j+1) = nbfirst(j) + nfill(j)
      end do
      deallocate (nfill)
      return
    end if
    do j = 1, numat
      do k = 1, j - 1
        kj = ijbo (j, k)
        if (kj >= 0) then
          if (iorbs(j)*iorbs(k) /= 0 .and. p(kj+1) > cutoff) then
            m = nbfirst(j) + nfill(j)
            nbatom(m) = k
            nboff(m) = kj
            nbp(m) = p(kj+1)
            nfill(j) = nfill(j) + 1
            m = nbfirst(k) + nfill(k)
            nbatom(m) = j
            nboff(m) = kj
            nbp(m) = p(kj+1)
            nfill(k) = nfill(k) + 1
          end if
        end if
      end do
    end do
    !
    !  Shell sort of each list, largest block first
    !
    do j = 1, numat
      n = nfill(j)
      gap = 1
      do while (gap < n/3)
        gap = 3*gap + 1
      end do
      do while (gap >= 1)
        do i = nbfirst(j) + gap, nbfirst(j) + n - 1
          pa = nbp(i)
          ia = nbatom(i)
          io = nboff(i)
          m = i
          do while (m - gap >= nbfirst(j))
            if (nbp(m-gap) >= pa) exit
            nbp(m) = nbp(m-gap)
            nbatom(m) = nbatom(m-gap)
            nboff(m) = nboff(m-gap)
            m = m - gap
          end do
          nbp(m) = pa
          nbatom(m) = ia
          nboff(m) = io
        end do
        gap = gap/3
      end do
    end do
    deallocate (nfill)
end subroutine diagg1_neighbours
subroutine diagg1_ws (fao, k1, kl, avirk, cutoff, nbfirst, nbatom, nboff, nbp, ws, latoms)
   !**********************************************************************
   !
   !  Add to WS the contribution of atom K1 of a virtual LMO: for every atom
   !  J1 (K1 itself and its neighbours), WS(J1) += FAO(J1,K1) * CVIR(K1), if
   !  AVIRK*P(J1,K1) exceeds CUTOFF (AVIRK is the square of the coefficients
   !  of K1, P holds the density on the diagonal block and the square of the
   !  Fock block, from EIMP, elsewhere).  KL is the address of K1 in CVIR.
   !  An atom J1 that is not yet in LATOMS is added, and its WS zeroed.
   !  The neighbour lists are sorted by decreasing P, so the search stops at
   !  the first block that fails the test.
   !
   !**********************************************************************
    use molkst_C, only: numat, norbs, mpack
    use MOZYME_C, only : iorbs, cvir
    use common_arrays_C, only : nfirst, nlast, p
    implicit none
    integer, intent (in) :: k1, kl
    double precision, intent (in) :: avirk, cutoff
    integer, dimension (numat+1), intent (in) :: nbfirst
    integer, dimension (*), intent (in) :: nbatom, nboff
    double precision, dimension (*), intent (in) :: nbp
    double precision, dimension (mpack), intent (in) :: fao
    double precision, dimension (norbs), intent (inout) :: ws
    logical, dimension (numat), intent (inout) :: latoms
    integer :: i4, ii, j1, jx, kj, m
    integer, external :: ijbo
    !
    !  Diagonal block
    !
    kj = ijbo (k1, k1)
    if (avirk*p(kj+1) > cutoff) then
      do jx = 1, iorbs(k1)
        do i4 = 1, iorbs(k1)
          if (i4 > jx) then
            ii = kj + (i4*(i4-1)) / 2 + jx
          else
            ii = kj + (jx*(jx-1)) / 2 + i4
          end if
          ws(nfirst(k1)+jx-1) = ws(nfirst(k1)+jx-1) + fao(ii) * cvir(kl+i4)
        end do
      end do
    end if
    do m = nbfirst(k1), nbfirst(k1+1) - 1
      if (avirk*nbp(m) <= cutoff) exit
      j1 = nbatom(m)
      if (.not. latoms(j1)) then
        latoms(j1) = .true.
        do jx = nfirst(j1), nlast(j1)
          ws(jx) = 0.d0
        end do
      end if
      ii = nboff(m)
      if (k1 > j1) then
        do i4 = 1, iorbs(k1)
          do jx = 1, iorbs(j1)
            ii = ii + 1
            ws(nfirst(j1)+jx-1) = ws(nfirst(j1)+jx-1) + fao(ii) * cvir(kl+i4)
          end do
        end do
      else
        do jx = 1, iorbs(j1)
          do i4 = 1, iorbs(k1)
            ii = ii + 1
            ws(nfirst(j1)+jx-1) = ws(nfirst(j1)+jx-1) + fao(ii) * cvir(kl+i4)
          end do
        end do
      end if
    end do
end subroutine diagg1_ws
