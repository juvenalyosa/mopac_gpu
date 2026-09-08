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

  subroutine diagg2 (nocc, nvir,  eigv, iused, latoms, &
     & nij, idiagg, storei, storej)
   !***********************************************************************
   !
   !   DIAGG2  PERFORMS A SIMPLE JACOBIAN ANNIHILATION OF THE ENERGY TERMS
   !   CONNECTING THE OCCUPIED LMOS AND THE VIRTUAL LMOS.  THE ENERGY TERMS
   !   ARE IN THE ARRAY FMO, AND THE INDICES OF THE LMOS ARE IN IFMO.
   !
   !***********************************************************************
    use molkst_C, only: numat, norbs, numcal, keywrd, id
    use MOZYME_C, only : nvirtual, icocc_dim, shift, &
       & icvir_dim, cocc_dim, cvir_dim, ipad2, thresh, tiny, sumb, &
       ncf, nce, nncf, nnce, ncocc, ncvir, iorbs, cocc, cvir, icocc, icvir, &
       ifmo, fmo
!
    use common_arrays_C, only : eigs, nat
    use parameters_C, only: main_group
    use chanel_C, only: iw
    use mozyme_diagg2_state, only: mozyme_diagg2_retry, &
       mozyme_diagg2_set_rejections
#ifdef GPU
    use iso_c_binding, only: c_int, c_double
    use mozyme_gpu_int_utils, only: mozyme_c_int_nonnegative_or_zero, &
      mozyme_c_int_positive_or_zero
    use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required
    use mod_vars_cuda, only: lgpu, mozyme_gpu
#endif
    implicit none
    integer, intent (in) :: idiagg, nij, nocc, nvir
    logical, dimension (numat), intent (out) :: latoms
    integer, dimension (numat), intent (out) :: iused
    double precision, dimension (norbs), intent (out) :: storei, storej
    double precision, dimension (nvirtual), intent (in) :: eigv
    logical :: bug = .false.
    logical :: retry
    logical, save :: debug, times
    integer :: i, ii, jur, l
    integer, save :: icalcn = 0
    integer :: ij, ilr, incv, iur, j, jlr, jncf, k, le, lf, lij, loopi, loopj, &
   & mie, mle, mlee, mlf, mlff, ncei, ncfj, nrej
    double precision :: a, alpha, b, beta, biglim, c, d, e, sum
    double precision, save :: const, eps, eta, bigeps
    double precision, external :: reada
#ifdef GPU
    logical :: rotprep_gpu_done
    integer(c_int) :: gpu_code, gpu_active, gpu_nrej
    integer :: gpu_alloc_stat
    real(c_double) :: gpu_sumb, gpu_wall_ms
    integer(c_int), allocatable :: rot_active(:)
    double precision, allocatable :: rot_alpha(:)

    interface
      function mopac_cuda_mozyme_diagg2_rotate(nij_c, nocc_c, nvir_c, &
          numat_c, norbs_c, icocc_dim_c, icvir_dim_c, cocc_dim_c, &
          cvir_dim_c, ifmo_c, fmo_c, eigs_c, eigv_c, nncf_c, ncf_c, &
          ncocc_c, icocc_c, nnce_c, nce_c, ncvir_c, icvir_c, iorbs_c, &
          cocc_c, cvir_c, shift_c, rot_const_c, tiny_c, biglim_c, &
          thresh_c, retry_c, sumb_c, nrej_c, wall_ms_c) &
          bind(C,name='mopac_cuda_mozyme_diagg2_rotate') result(code)
        import :: c_int, c_double
        integer(c_int), value :: nij_c, nocc_c, nvir_c, numat_c, norbs_c
        integer(c_int), value :: icocc_dim_c, icvir_dim_c
        integer(c_int), value :: cocc_dim_c, cvir_dim_c
        integer(c_int), intent(in) :: ifmo_c(*)
        real(c_double), intent(in) :: fmo_c(*), eigs_c(*), eigv_c(*)
        integer(c_int), intent(in) :: nncf_c(*), ncocc_c(*)
        integer(c_int), intent(in) :: nnce_c(*), ncvir_c(*), iorbs_c(*)
        integer(c_int) :: ncf_c(*), icocc_c(*), nce_c(*), icvir_c(*)
        real(c_double) :: cocc_c(*), cvir_c(*)
        real(c_double), value :: shift_c, rot_const_c, tiny_c, biglim_c
        real(c_double), value :: thresh_c
        integer(c_int), value :: retry_c
        real(c_double) :: sumb_c, wall_ms_c
        integer(c_int) :: nrej_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_diagg2_rotate

      function mopac_cuda_mozyme_diagg2_rotprep(nij_c, nocc_c, nvir_c, &
          ifmo_c, fmo_c, eigs_c, eigv_c, shift_c, rot_const_c, tiny_c, &
          biglim_c, active_c, alpha_c, active_count_c, wall_ms_c) &
          bind(C,name='mopac_cuda_mozyme_diagg2_rotprep') result(code)
        import :: c_int, c_double
        integer(c_int), value :: nij_c, nocc_c, nvir_c
        integer(c_int), intent(in) :: ifmo_c(*)
        real(c_double), intent(in) :: fmo_c(*), eigs_c(*), eigv_c(*)
        real(c_double), value :: shift_c, rot_const_c, tiny_c, biglim_c
        integer(c_int) :: active_c(*), active_count_c
        real(c_double) :: alpha_c(*), wall_ms_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_diagg2_rotprep
    end interface
#endif
    if (numcal /= icalcn) then
      icalcn = numcal
      times = (Index (keywrd, " TIMES") /= 0)
      debug = (Index (keywrd, " DIAGG2") /= 0)
      !
      !   IF THE SYSTEM IS A SOLID, THEN DAMP ROTATION OF VECTORS,
      !   IN AN ATTEMPT TO PREVENT AUTOREGENERATIVE CHARGE OSCILLATION.
      !
      i = Index (keywrd, " DAMP")
      if (i /= 0) then
        const = reada (keywrd, i+5)
      else if (id == 3) then
        const = 0.5d0
      else
        a = 1.d0               !  If a transition metal, set DAMP to 0.5d0
        do i = 1, numat        !
          if (.not. main_group(nat(i)))  a = 0.5d0
        end do
        const = a
      end if
      !
      !   EPS IS THE SMALLEST NUMBER WHICH, WHEN ADDED TO 1.D0, IS NOT
      !   EQUAL TO 1.D0
      call epseta (eps, eta)
      !
      !   INCREASE EPS TO ALLOW FOR A LOT OF ROUND-OFF
      !
      bigeps = 50.d0 * Sqrt (eps)
    end if
   !
   !  RETRY IS .TRUE. IF THE NUMBER OF REJECTED ANNIHILATIONS IS IN THE
   !         RANGE 1 TO 20  AND THE SAME ON TWO ITERATIONS.  THIS WILL
   !         OCCUR NEAR THE END OF A SCF CALCULATION, WHEN ONLY A FEW
   !         LMOS ARE BADLY BEHAVED.
   !
    retry = mozyme_diagg2_retry()
    if (Mod(idiagg, 5) == 0 .or. idiagg <= 5) then
      tiny = -1.d0
      biglim = -1.d0
    else
      tiny = 0.01d0 * tiny
      biglim = bigeps
    end if
   !***********************************************************************
   !
   !   DO A CRUDE 2 BY 2 ROTATION TO "ELIMINATE" SIGNIFICANT ELEMENTS
   !
   !***********************************************************************
    iused(:) = -1
    latoms(:) = .false.
    if (debug) then
      write (iw,*)
      write (iw,*) "            SIZE OF OCCUPIED ARRAYS IN DIAGG2"
      write (iw,*)
      write (iw,*) "    LMO    NNCF     NCF   SPACE  ", " NCOCC    SIZE   SPACE"
      do i = 1, nocc - 1
        l = ncocc(i)
        do j = nncf(i) + 1, nncf(i) + ncf(i)
          l = l + iorbs(icocc(j))
        end do
        write (iw, "(7I8)") i, nncf (i), ncf (i), nncf (i+1) - &
       & nncf(i) - ncf(i), ncocc(i), l - ncocc(i), ncocc(i+1) - l
      end do
      i = nocc
      l = ncocc(i)
      do j = nncf(i) + 1, nncf(i) + ncf(i)
        l = l + iorbs(icocc(j))
      end do
      write (iw, "(7I8)") i, nncf (i), ncf (i), icocc_dim - nncf (i) - ncf &
     & (i), ncocc(i), l - ncocc(i), cocc_dim - l
      write (iw,*)
      write (iw,*) "            SIZE OF VIRTUAL ARRAYS IN DIAGG2"
      write (iw,*)
      write (iw,*) "    LMO    NNCE     NCE   SPACE  ", " NCVIR    SIZE   SPACE"
      do i = 1, nvir - 1
        l = ncvir(i)
        do j = nnce(i) + 1, nnce(i) + nce(i)
          l = l + iorbs(icvir(j))
        end do
        write (iw, "(7I8)") i, nnce (i), nce (i), nnce (i+1) - nnce (i) - nce &
       & (i), ncvir(i), l - ncvir(i), ncvir(i+1) - l
      end do
      i = nvir
      l = ncvir(i)
      do j = nnce(i) + 1, nnce(i) + nce(i)
        l = l + iorbs(icvir(j))
      end do
      write (iw, "(7I8)") i, nnce (i), nce (i), icvir_dim - nnce (i) - nce (i), &
     & ncvir(i), l - ncvir(i), cvir_dim - l
      if (bug) then
        write (iw,*)
        write (iw, "(A,I3,A)") " THIS FAULT CAN PROBABLY BE " // &
                             & "CORRECTED BY USE OF KEYWORD 'NLMO=", &
                             & ipad2 + 50, "'"
        write (iw,*)
        call mopend ("VALUE OF NLMO IS TOO SMALL")
      end if
    end if
    sumb = 0.d0
    nrej = 0
    lij = 0
#ifdef GPU
    if (mozyme_diagg2_rotate_gpu_enabled()) then
      gpu_sumb = 0.0_c_double
      gpu_nrej = 0_c_int
      gpu_wall_ms = 0.0_c_double
      gpu_code = mopac_cuda_mozyme_diagg2_rotate( &
        mozyme_c_int_nonnegative_or_zero(nij), &
        mozyme_c_int_nonnegative_or_zero(nocc), &
        mozyme_c_int_nonnegative_or_zero(nvir), &
        mozyme_c_int_positive_or_zero(numat), &
        mozyme_c_int_positive_or_zero(norbs), &
        mozyme_c_int_positive_or_zero(icocc_dim), &
        mozyme_c_int_positive_or_zero(icvir_dim), &
        mozyme_c_int_positive_or_zero(cocc_dim), &
        mozyme_c_int_positive_or_zero(cvir_dim), ifmo, fmo, eigs, eigv, &
        nncf, ncf, ncocc, icocc, nnce, nce, ncvir, icvir, iorbs, cocc, &
        cvir, shift, const, tiny, biglim, thresh, merge(1_c_int, 0_c_int, &
        retry), gpu_sumb, gpu_nrej, gpu_wall_ms)
      if (gpu_code == 0_c_int) then
        sumb = gpu_sumb
        nrej = int(gpu_nrej)
        call mozyme_diagg2_set_rejections(nrej)
        if (mozyme_diagg2_rotprep_trace()) then
          write(iw,'(1x,a," success code=",i0," nrej=",i0," sumb=",es13.6," ms=",f12.6)') &
            '[MOZYME GPU diagg2_rotate]', int(gpu_code), &
            nrej, sumb, gpu_wall_ms
          call flush(iw)
        end if
        if (times) then
          call timer (" AFTER DIAGG2 IN ITER")
        end if
        return
      else if (mozyme_diagg2_rotprep_trace()) then
        write(iw,'(1x,a," fallback_cpu code=",i0)') &
          '[MOZYME GPU diagg2_rotate]', int(gpu_code)
        call flush(iw)
      end if
    end if
    if (mozyme_gpu_scf_no_fallback_required()) then
      write(iw,'(1x,a)') &
        '[MOZYME GPU SCF] status=strict_abort reason=strict_diagg2_cpu_fallback'
      call flush(iw)
      error stop 'MOZYME GPU strict diagg2 abort'
    end if
    rotprep_gpu_done = .false.
    if (mozyme_diagg2_rotprep_gpu_enabled() .and. nij > 0) then
      allocate(rot_active(nij), rot_alpha(nij), stat=gpu_alloc_stat)
      if (gpu_alloc_stat == 0) then
        gpu_active = 0_c_int
        gpu_wall_ms = 0.0_c_double
        gpu_code = mopac_cuda_mozyme_diagg2_rotprep( &
          mozyme_c_int_nonnegative_or_zero(nij), &
          mozyme_c_int_nonnegative_or_zero(nocc), &
          mozyme_c_int_nonnegative_or_zero(nvir), ifmo, fmo, eigs, eigv, shift, &
          const, tiny, biglim, rot_active, rot_alpha, gpu_active, gpu_wall_ms)
        if (gpu_code == 0_c_int) then
          rotprep_gpu_done = .true.
          if (mozyme_diagg2_rotprep_trace()) then
            write(iw,'(1x,a," success code=",i0," active=",i0," ms=",f12.6)') &
              '[MOZYME GPU diagg2_rotprep]', int(gpu_code), int(gpu_active), &
              gpu_wall_ms
            call flush(iw)
          end if
        else if (mozyme_diagg2_rotprep_trace()) then
          write(iw,'(1x,a," fallback_cpu code=",i0)') &
            '[MOZYME GPU diagg2_rotprep]', int(gpu_code)
          call flush(iw)
        end if
      else if (mozyme_diagg2_rotprep_trace()) then
        write(iw,'(1x,a," fallback_cpu code=",i0)') &
          '[MOZYME GPU diagg2_rotprep]', -2
        call flush(iw)
      end if
    end if
#endif
    outer_loop: do ij = 1, nij
      i = ifmo(1, ij)
      j = ifmo(2, ij)
      if (Abs (fmo(ij)) >= tiny) then
        c = fmo(ij) * const
        d = eigs(j) - eigv(i) - shift
        if (Abs (c/d) >= biglim) then
          ncfj = ncf(j)
          ncei = nce(i)
      !
      !  STORE LMOS FOR POSSIBLE REJECTION, IF LMOS EXPAND TOO MUCH.
      !
          jlr = ncocc(j) + 1
          if (j /= nocc) then
            jur = ncocc(j+1)
            jncf = nncf(j+1)
          else
            jur = cocc_dim
            jncf = icocc_dim
          end if
          jur = Min (jlr+norbs-1, jur)
          ilr = ncvir(i) + 1
          if (i /= nvir) then
            iur = ncvir(i+1)
            incv = nnce(i+1)
          else
            iur = cvir_dim
            incv = icvir_dim
          end if
          iur = Min (ilr+norbs-1, iur)
          l = 0
          do k = jlr, jur
            l = l + 1
            storej(l) = cocc(k)
          end do
          l = 0
          do k = ilr, iur
            l = l + 1
            storei(l) = cvir(k)
          end do
          !
          !   STORAGE DONE.
          !
          lij = lij + 1
          e = Sign (Sqrt(4.d0*c*c+d*d), d)
          alpha = Sqrt (0.5d0*(1.d0+d/e))
#ifdef GPU
          if (rotprep_gpu_done) then
            if (rot_active(ij) /= 0_c_int) alpha = rot_alpha(ij)
          end if
#endif
          do
            beta = -Sign (Sqrt(1.d0-alpha*alpha), c)
            sumb = sumb + Abs (beta)
            !
            ! IDENTIFY THE ATOMS IN THE OCCUPIED LMO.  ATOMS NOT USED ARE
            ! FLAGGED BY '-1' IN IUSED.
            !
            mlf = 0
            !
            do lf = nncf(j) + 1, nncf(j) + ncf(j)
              ii = icocc(lf)
              iused(ii) = mlf
              mlf = mlf + iorbs(ii)
            end do
            loopi = ncvir(i)
            loopj = ncocc(j)
            mle = 0
         !
         !      ROTATION OF PSEUDO-EIGENVECTORS
         !
            do le = nnce(i) + 1, nnce(i) + nce(i)
              mie = icvir(le)

              latoms(mie) = .true.
              mlff = iused(mie) + loopj
              if (iused(mie) >= 0) then
                !
                !  TWO BY TWO ROTATION OF ATOMS WHICH ARE COMMON
                !  TO OCCUPIED LMO J AND VIRTUAL LMO I
                !
                do mlee = mle + 1 + loopi, mle + iorbs(mie) + loopi
                  mlff = mlff + 1
                  a = cocc(mlff)
                  b = cvir(mlee)
                  cocc(mlff) = alpha * a + beta * b
                  cvir(mlee) = alpha * b - beta * a
                end do
              else
                !
                !   FILLED  LMO ATOM 'MIE' DOES NOT EXIST.
                !   CHECK IF IT SHOULD EXIST
                !
                sum = 0.d0
                do mlee = mle + 1 + loopi, mle + iorbs(mie) + loopi
                  sum = sum + (beta*cvir(mlee)) ** 2
                end do
                if (sum > thresh) then
                  !
                  if (nncf(j)+ncf(j) >= jncf) go to 1000
                  if (mlf+iorbs(mie)+loopj > jur) go to 1000
                  !
                  !  YES, OCCUPIED LMO ATOM 'MIE' SHOULD EXIST
                  !
                  ncf(j) = ncf(j) + 1
                  icocc(nncf(j)+ncf(j)) = mie
                  !
                  iused(mie) = mlf
                  mlf = mlf + iorbs(mie)
                  !
                  !   PUT INTENSITY INTO OCCUPIED LMO ATOM 'MIE'
                  !
                  mlff = iused(mie) + loopj
                  do mlee = mle + 1 + loopi, mle + iorbs(mie) + loopi
                    mlff = mlff + 1
                    cocc(mlff) = beta * cvir(mlee)
                    cvir(mlee) = alpha * cvir(mlee)
                  end do
                end if
              end if
              mle = mle + iorbs(mie)
            end do
            !
            !  NOW CHECK ALL ATOMS WHICH WERE IN THE OCCUPIED LMO
            !  WHICH ARE NOT IN THE VIRTUAL LMO, TO SEE IF THEY
            !  SHOULD BE IN THE VIRTUAL LMO.
            !
            do lf = nncf(j) + 1, nncf(j) + ncf(j)
              ii = icocc(lf)

              if ( .not. latoms(ii)) then
                sum = 0.d0
                do mlff = iused(ii) + loopj + 1, iused(ii) + loopj + &
                     & iorbs(ii)
                  sum = sum + (beta*cocc(mlff)) ** 2
                end do
                if (sum > thresh) then
                  if (nnce(i)+nce(i) >= incv) go to 1000
                  if (mle+iorbs(ii)+loopi > iur) go to 1000
                  !
                  !  YES, VIRTUAL  LMO ATOM 'II' SHOULD EXIST
                  !
                  nce(i) = nce(i) + 1
                  icvir(nnce(i)+nce(i)) = ii
                  latoms(ii) = .true.
                  !
                  !   PUT INTENSITY INTO VIRTUAL  LMO ATOM 'II'
                  !
                  mlff = iused(ii) + loopj
                  do mlee = mle + 1 + loopi, mle + iorbs(ii) + loopi
                    mlff = mlff + 1
                     !
                    cvir(mlee) = -beta * cocc(mlff)
                    cocc(mlff) = alpha * cocc(mlff)
                  end do
                  mle = mle + iorbs(ii)
                end if
              end if
            end do
          exit
      1000  continue
            nrej = nrej + 1
              !
              !   THE ARRAY BOUNDS WERE GOING TO BE EXCEEDED.
              !   TO PREVENT THIS, RESET THE LMOS.
              !
            l = 0
            do k = jlr, jur
              l = l + 1
              cocc(k) = storej(l)
            end do
            l = 0
            do k = ilr, iur
              l = l + 1
              cvir(k) = storei(l)
            end do
            ncf(j) = ncfj
            nce(i) = ncei
            do k = 1, numat
              iused(k) = -1
              latoms(k) = .false.
            end do
            if (retry) then
                !
                !   HALF THE ROTATION ANGLE.  WILL THIS PREVENT THE
                !   ARRAY BOUND FROM BEING EXCEEDED?
                !
              alpha = 0.5d0 * (alpha+1.d0)
            else
              cycle outer_loop
            end if
          end do
        !
        !  RESET COUNTERS WHICH HAVE BEEN SET.
        !
          do le = nnce(i) + 1, nnce(i) + nce(i)
            mie = icvir(le)
            latoms(mie) = .false.
          end do
        !
          do lf = nncf(j) + 1, nncf(j) + ncf(j)
            iused(icocc(lf)) = -1
          end do
        end if
      end if
    end do outer_loop
    call mozyme_diagg2_set_rejections(nrej)
#ifdef GPU
    if (allocated(rot_active)) deallocate(rot_active)
    if (allocated(rot_alpha)) deallocate(rot_alpha)
#endif
    if (times) then
      call timer (" AFTER DIAGG2 IN ITER")
    end if
#ifdef GPU
contains
  logical function mozyme_diagg2_rotate_gpu_enabled()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_diagg2_rotate_gpu_enabled = .false.
    if (.not. (lgpu .and. mozyme_gpu)) return
    env_value = ' '
    call get_environment_variable('MOPAC_MOZYME_DIAGG2_ROTATE_GPU', &
      env_value, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0) then
      select case (trim(env_value))
      case ('0', 'off', 'OFF', 'false', 'FALSE', 'no', 'NO')
        mozyme_diagg2_rotate_gpu_enabled = .false.
      case default
        mozyme_diagg2_rotate_gpu_enabled = .true.
      end select
    end if
  end function mozyme_diagg2_rotate_gpu_enabled

  logical function mozyme_diagg2_rotprep_gpu_enabled()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_diagg2_rotprep_gpu_enabled = lgpu .and. mozyme_gpu
    if (.not. mozyme_diagg2_rotprep_gpu_enabled) return
    env_value = ' '
    call get_environment_variable('MOPAC_MOZYME_DIAGG2_ROTPREP_GPU', &
      env_value, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0) then
      select case (trim(env_value))
      case ('0', 'off', 'OFF', 'false', 'FALSE', 'no', 'NO')
        mozyme_diagg2_rotprep_gpu_enabled = .false.
      case default
        mozyme_diagg2_rotprep_gpu_enabled = .true.
      end select
    end if
  end function mozyme_diagg2_rotprep_gpu_enabled

  logical function mozyme_diagg2_rotprep_trace()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value

    mozyme_diagg2_rotprep_trace = .false.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_PROFILE', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') &
      mozyme_diagg2_rotprep_trace = .true.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_VERBOSE', env_value, &
      length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') &
      mozyme_diagg2_rotprep_trace = .true.
  end function mozyme_diagg2_rotprep_trace
#endif
  end subroutine diagg2
