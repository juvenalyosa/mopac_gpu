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

      subroutine pulay(f, p, n, fppf, fock, emat, lfock, nfock, msize, start, pl)
      use chanel_C, only : iw
      use molkst_C, only : numcal, keywrd, mpack, npulay
      use iter_C, only : pulay_work1, pulay_work2, pulay_work3
#ifdef GPU
      use mod_vars_cuda, only : lgpu
      use gpu_diis_interfaces
      use gpu_bmat_interfaces
      use gpu_small_solve_interfaces
      use overlap_build, only : build_overlap_packed
#endif
      implicit none
      integer  :: n
      integer , intent(inout) :: lfock
      integer , intent(inout) :: nfock
      integer , intent(in) :: msize
      double precision , intent(out) :: pl
      logical , intent(inout) :: start
      double precision  :: f(mpack)
      double precision  :: p(mpack)
      double precision  :: fppf(*)
      double precision , intent(inout) :: fock(*)
      double precision , intent(inout) :: emat(npulay+1,npulay+1)
!
      integer :: icalcn, maxlim, linear, mfock, lbase, i, nfock1, j, l, il, ii
      double precision, dimension((npulay+1)**2) :: evec
      double precision, dimension(npulay) :: coeffs
      double precision :: const, d, sum
      logical :: debug
      logical :: coeffs_ready
      logical :: use_gpu_diis, use_gpu_buffer, use_gpu_bfull, use_gpu_bcol, use_gpu_solve, use_gen_resid
      integer :: info_s
#ifdef GPU
      character(len=32) :: env
      integer :: istat_env, iopc_sel
      double precision, allocatable :: spack(:), tmp1(:), tmp2(:)
      double precision, allocatable :: bfull(:,:), bcol(:), amat(:,:), rhs(:)
#endif
      double precision, external :: ddot
!
      save icalcn, maxlim, debug, linear, mfock
!-----------------------------------------------
!***********************************************************************
!
!   PULAY USES DR. PETER PULAY'S METHOD FOR CONVERGENCE.
!         A MATHEMATICAL DESCRIPTION CAN BE FOUND IN
!         "P. PULAY, J. COMP. CHEM. 3, 556 (1982).
!
! ARGUMENTS:-
!         ON INPUT F      = FOCK MATRIX, PACKED, LOWER HALF TRIANGLE.
!                  P      = DENSITY MATRIX, PACKED, LOWER HALF TRIANGLE.
!                  N      = NUMBER OF ORBITALS.
!                  FPPF   = WORKSTORE OF SIZE MSIZE, CONTENTS WILL BE
!                           OVERWRITTEN.
!                  FOCK   =      "       "              "         "
!                  EMAT   = WORKSTORE OF AT LEAST NPULAY**2 ELEMENTS.
!                  START  = LOGICAL, = TRUE TO START PULAY.
!                  PL     = UNDEFINED ELEMENT.
!      ON OUTPUT   F      = "BEST" FOCK MATRIX, = LINEAR COMBINATION
!                           OF KNOWN FOCK MATRICES.
!                  START  = FALSE
!                  PL     = MEASURE OF NON-SELF-CONSISTENCY
!                         = [F*P] = F*P - P*F.
!
!***********************************************************************
      data icalcn/ 0/
      use_gpu_diis = .false.
      use_gpu_buffer = .false.
      use_gpu_bfull = .false.
      use_gpu_bcol = .false.
      use_gpu_solve = .false.
      use_gen_resid = .false.
      coeffs_ready = .false.
      info_s = 0
#ifdef GPU
      if (lgpu) use_gpu_diis = .true.
#endif
      if (icalcn /= numcal) then
        icalcn = numcal
        maxlim = npulay
        debug = index(keywrd,'DEBUGPULAY') /= 0
      end if
      if (start) then
        linear = (n*(n + 1))/2
        mfock = msize/linear
        mfock = min0(maxlim,mfock)
        if (debug) write (iw, '('' MAXIMUM SIZE:'',I5)') mfock
        nfock = 1
        lfock = 1
        start = .FALSE.
#ifdef GPU
        if (use_gpu_diis) call mopac_cuda_diis_init(linear, mfock)
#endif
      else
        if (nfock < mfock) nfock = nfock + 1
        if (lfock /= mfock) then
          lfock = lfock + 1
        else
          lfock = 1
        end if
      end if
      lbase = (lfock - 1)*linear
#ifdef GPU
      if (use_gpu_diis) then
        env = '' ; istat_env = 1
        call get_environment_variable('MOPAC_DIIS_GEN', env, status=istat_env)
        if (istat_env == 0) use_gen_resid = (trim(adjustl(env)) /= '')
        env = '' ; istat_env = 1
        call get_environment_variable('MOPAC_DIIS_GPU_BUF', env, status=istat_env)
        if (istat_env == 0) use_gpu_buffer = (trim(adjustl(env)) /= '')
        env = '' ; istat_env = 1
        call get_environment_variable('MOPAC_DIIS_GPU_BFULL', env, status=istat_env)
        if (istat_env == 0) use_gpu_bfull = (trim(adjustl(env)) /= '')
        env = '' ; istat_env = 1
        call get_environment_variable('MOPAC_DIIS_GPU_BMAT', env, status=istat_env)
        if (istat_env == 0) use_gpu_bcol = (trim(adjustl(env)) /= '')
        env = '' ; istat_env = 1
        call get_environment_variable('MOPAC_DIIS_GPU', env, status=istat_env)
        if (istat_env == 0) use_gpu_solve = (trim(adjustl(env)) /= '')
        if (use_gpu_bfull) use_gpu_bcol = .false.
      end if
#endif
!
!   FIRST, STORE FOCK MATRIX FOR FUTURE REFERENCE.
!
      fock(lfock:(linear-1)*mfock+lfock:mfock) = f(:linear)
!
!   NOW FORM /FOCK*DENSITY-DENSITY*FOCK/, AND STORE THIS IN FPPF
!
!      call mamult (p, f, fppf(lbase+1), n, 0.D0)
!      call mamult (f, p, fppf(lbase+1), n, -1.D0)
#ifdef GPU
      if (use_gpu_diis .and. use_gen_resid) then
        allocate(spack(linear), tmp1(linear), tmp2(linear), stat=i)
        if (i /= 0) then
          call memory_error('Pulay generalized residual alloc')
          return
        end if
        call build_overlap_packed(spack)
        iopc_sel = 4
        call mult_symm_AB(p, spack, 1.d0, n, linear, tmp1, 0.d0, iopc_sel)
        call mult_symm_AB(f, tmp1, 1.d0, n, linear, fppf(lbase+1:lbase+linear), 0.d0, iopc_sel)
        call mult_symm_AB(p, f, 1.d0, n, linear, tmp2, 0.d0, iopc_sel)
        call mult_symm_AB(spack, tmp2, 1.d0, n, linear, tmp1, 0.d0, iopc_sel)
        fppf(lbase+1:lbase+linear) = fppf(lbase+1:lbase+linear) - tmp1(:linear)
        deallocate(spack, tmp1, tmp2, stat=i)
      else
#endif
      call unpack_matrix(p, pulay_work1, n)
      call unpack_matrix(f, pulay_work2, n)
      call sym_commute(pulay_work1, pulay_work2, pulay_work3, n)
      call pack_matrix(pulay_work3, fppf(lbase+1), n)
#ifdef GPU
      end if
#endif
!
!   FPPF NOW CONTAINS THE RESULT OF FP - PF.
!
      nfock1 = nfock + 1
#ifdef GPU
      if (use_gpu_diis) then
        if (use_gpu_buffer) call mopac_cuda_diis_store(linear, lfock, fppf(lbase+1:lbase+linear))
        if (use_gpu_bfull) then
          allocate(bfull(nfock, nfock), stat=i)
          if (i /= 0) then
            call memory_error('Pulay GPU Bfull alloc')
            return
          end if
          if (use_gpu_buffer) then
            call mopac_cuda_bfull_from_device(linear, nfock, bfull)
          else
            call mopac_cuda_bfull_from_host(linear, nfock, fppf, bfull)
          end if
          emat(1:nfock,1:nfock) = bfull(1:nfock,1:nfock)
          do i = 1, nfock
            emat(nfock1,i) = -1.D0
            emat(i,nfock1) = -1.D0
          end do
          deallocate(bfull, stat=i)
        else if (use_gpu_bcol) then
          allocate(bcol(nfock), stat=i)
          if (i /= 0) then
            call memory_error('Pulay GPU Bcol alloc')
            return
          end if
          if (use_gpu_buffer) then
            call mopac_cuda_diis_bcol(linear, nfock, lfock, bcol)
          else
            call mopac_cuda_bcol_from_residuals(linear, nfock, fppf, lfock, bcol)
          end if
          do i = 1, nfock
            emat(nfock1,i) = -1.D0
            emat(i,nfock1) = -1.D0
            emat(lfock,i) = bcol(i)
            emat(i,lfock) = emat(lfock,i)
          end do
          deallocate(bcol, stat=i)
        else
          do i = 1, nfock
            emat(nfock1,i) = -1.D0
            emat(i,nfock1) = -1.D0
            emat(lfock,i) = ddot(linear,fppf((i-1)*linear+1),1,fppf(lbase+1),1)
            emat(i,lfock) = emat(lfock,i)
          end do
        end if
      else
#endif
      do i = 1, nfock
        emat(nfock1,i) = -1.D0
        emat(i,nfock1) = -1.D0
        emat(lfock,i) = ddot(linear,fppf((i-1)*linear+1),1,fppf(lbase+1),1)
        emat(i,lfock) = emat(lfock,i)
      end do
#ifdef GPU
      end if
#endif
      pl = emat(lfock,lfock)/linear
      emat(nfock1,nfock1) = 0.D0
      if (emat(lfock, lfock) < 1.d-20) return
      const = 1.D0/emat(lfock,lfock)
      emat(:nfock,:nfock) = emat(:nfock,:nfock)*const
      if (debug) then
        write (iw, '('' EMAT'')')
        do i = 1, nfock1
          write (iw, '(6E13.6)') (emat(j,i),j=1,nfock1)
        end do
      end if
      l = 0
      do i = 1, nfock1
        evec(l+1:nfock1+l) = emat(i,:nfock1)
        l = nfock1 + l
      end do
      const = 1.D0/const
      emat(:nfock,:nfock) = emat(:nfock,:nfock)*const
      coeffs_ready = .false.
#ifdef GPU
      if (use_gpu_diis .and. use_gpu_solve) then
        allocate(amat(nfock1,nfock1), rhs(nfock1), stat=i)
        if (i /= 0) then
          call memory_error('Pulay GPU DIIS solve alloc')
          return
        end if
        amat(:,:) = 0.D0
        amat(:nfock,:nfock) = emat(:nfock,:nfock)
        amat(:nfock,nfock1) = emat(:nfock,nfock1)
        amat(nfock1,:nfock) = emat(nfock1,:nfock)
        amat(nfock1,nfock1) = emat(nfock1,nfock1)
        const = 1.D0/emat(lfock,lfock)
        amat(:nfock,:nfock) = amat(:nfock,:nfock) * const
        rhs(:) = 0.D0
        rhs(nfock1) = 1.D0
        call mopac_cuda_solve_linear(nfock1, amat, nfock1, rhs, info_s)
        if (info_s == 0) then
          if (nfock < 2) then
            deallocate(amat, rhs, stat=i)
            return
          end if
          coeffs(:nfock) = -rhs(:nfock)
          coeffs_ready = .true.
        end if
        deallocate(amat, rhs, stat=i)
      end if
#endif
!********************************************************************
!   THE MATRIX EMAT SHOULD HAVE FORM
!
!      |<E(1)*E(1)>  <E(1)*E(2)> ...   -1.0|
!      |<E(2)*E(1)>  <E(2)*E(2)> ...   -1.0|
!      |<E(3)*E(1)>  <E(3)*E(2)> ...   -1.0|
!      |<E(4)*E(1)>  <E(4)*E(2)> ...   -1.0|
!      |     .            .      ...     . |
!      |   -1.0         -1.0     ...    0. |
!
!   WHERE <E(I)*E(J)> IS THE SCALAR PRODUCT OF [F*P] FOR ITERATION I
!   TIMES [F*P] FOR ITERATION J.
!
!********************************************************************
      if (.not. coeffs_ready) then
        call osinv (evec, nfock1, d)
        if (abs(d) < 1.D-6) then
          start = .TRUE.
          return
        end if
        if (nfock < 2) return
        il = nfock*nfock1
        coeffs(:nfock) = -evec(1+il:nfock+il)
        coeffs_ready = .true.
      end if
      if (debug) then
        write (iw, '('' EVEC'')')
        write (iw, '(6F12.6)') (coeffs(i),i=1,nfock)
        write (iw, '(''    LAGRANGIAN MULTIPLIER (ERROR) =''                          ,F13.6)') evec(nfock1*nfock1)
      end if
      do i = 1, linear
        sum = 0.D0
        l = 0
        ii = (i - 1)*mfock
        do j = 1, nfock
          sum = sum + coeffs(j)*fock(j+ii)
        end do
        f(i) = sum
      end do
      return
      end subroutine pulay

      subroutine pack_matrix(unpacked, packed, size)
        implicit none
        integer :: info
        integer , intent(in) :: size
        double precision , intent(in) :: unpacked(size, size)
        double precision , intent(out) :: packed(*)
  !-----------------------------------------------
  !***********************************************************************
  !
  !   CONVERT UNPACKED SYMMETRIC MATRIX INTO A PACKED UPPER TRIANGLE
  !   (LAPACK DTRTTP CALL)
  !
  ! ARGUMENTS:-
  !         ON INPUT UNPACKED = UNPACKED SYMMETRIC MATRIX
  !                  SIZE     = DIMENSION OF MATRIX
  !      ON OUTPUT   PACKED   = PACKED UPPER TRIANGLE MATRIX
  !
  !***********************************************************************
        call dtrttp( 'U', size, unpacked, size, packed, info )
        if (info /= 0) stop 'error in dtrttp'
        return
        end subroutine pack_matrix

        subroutine unpack_matrix(packed, unpacked, size)
          implicit none
          integer :: info, i, j
          integer , intent(in) :: size
          double precision , intent(in) :: packed(*)
          double precision , intent(out) :: unpacked(size, size)
    !-----------------------------------------------
    !***********************************************************************
    !
    !   CONVERT PACKED UPPER TRIANGLE INTO AN UNPACKED SYMMETRIC MATRIX
    !   (LAPACK DTPTTR CALLS & FILLING IN THE REST BY HAND)
    !
    ! ARGUMENTS:-
    !         ON INPUT PACKED   = PACKED UPPER TRIANGLE MATRIX
    !                  SIZE     = DIMENSION OF MATRIX
    !      ON OUTPUT   UNPACKED = UNPACKED SYMMETRIC MATRIX
    !
    !***********************************************************************
          call dtpttr( 'U', size, packed, unpacked, size, info )
          if (info /= 0) stop 'error in dtpttr'
          do i = 1, size
            do j = i+1, size
              unpacked(j,i) = unpacked(i,j)
            end do
          end do
          return
          end subroutine unpack_matrix

          subroutine sym_commute(mat1, mat2, mat3, size)
            implicit none
            integer :: i, j
            integer , intent(in) :: size
            double precision , intent(in) :: mat1(size, size)
            double precision , intent(in) :: mat2(size, size)
            double precision , intent(out) :: mat3(size, size)
    !-----------------------------------------------
    !***********************************************************************
    !
    !   COMPUTE THE COMMUTATOR BETWEEN TWO SYMMETRIC MATRICES
    !   (DGEMM & AN IN-PLACE EVALUATION OF THE SECOND TERM)
    !
    ! ARGUMENTS:-
    !         ON INPUT MAT1   = FIRST SYMMETRIC MATRIX IN COMMUTATOR
    !                  MAT2   = SECOND SYMMETRIC MATRIX IN COMMUTATOR
    !                  SIZE   = DIMENSION OF MATRIX
    !      ON OUTPUT   MAT3   = MAT1*MAT2 - MAT2*MAT1
    !
    !***********************************************************************
            call dsymm('L', 'U', size, size, 1.D0, mat1, size, mat2, size, 0.D0, mat3, size)
            do i = 1, size
              do j = i, size
                mat3(i,j) = mat3(i,j) - mat3(j,i)
                mat3(j,i) = -mat3(i,j)
              end do
            end do
            return
            end subroutine sym_commute
