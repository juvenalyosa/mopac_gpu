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

subroutine reorth (ws)
    use molkst_C, only: numat, norbs
    use MOZYME_C, only : nvirtual, noccupied, icocc_dim, &
       & icvir_dim, cocc_dim, cvir_dim, ncf, nce, nncf, nnce, ncocc, ncvir, iorbs, &
       & icocc, icvir, cocc, cvir
    use common_arrays_C, only : nfirst
    use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required
    use mozyme_section_timers, only: mozyme_section_timers_enabled
    use chanel_C, only: iw
    implicit none
    double precision, dimension (norbs) :: ws
!
    integer :: i, ii, j, j1, jj, jx, loopi, loopii, alloc_stat
    double precision :: sum, sumtot
    double precision :: maxov   ! largest |<LMO_i|LMO_ii>| found before re-orthogonalizing
    logical, dimension (:), allocatable :: latom_loc
    integer, dimension (:), allocatable :: iused
    external :: mozyme_gpu_strict_abort
    if (mozyme_gpu_scf_no_fallback_required()) then
      call mozyme_gpu_strict_abort('strict_reorth_cpu_fallback', &
        'MOZYME GPU strict resident SCF does not support CPU reorthogonalization')
      return
    end if
!
    allocate (latom_loc(numat), iused(numat), stat=alloc_stat)
    if (alloc_stat /= 0) then
      call memory_error ("reorth")
      go to 1100
    end if
    sumtot = 0.d0
    maxov = 0.d0
    do i = 1, nvirtual
      loopi = ncvir(i)
      do j = 1, numat
        latom_loc(j) = .false.
      end do
      do jj = nnce(i) + 1, nnce(i) + nce(i)
        j1 = icvir(jj)
        latom_loc(j1) = .true.
        j = nfirst(j1) - 1
        do jx = 1, iorbs(j1)
          loopi = loopi + 1
          j = j + 1
          ws(j) = cvir(loopi)
        end do
      end do
      !
      !    WS holds the LMO coefficients, in atom number order.
      !
      do ii = i + 1, nvirtual
        j1 = icvir(nnce(ii)+1)
        sum = 0.d0
        loopii = ncvir(ii)
        do jj = nnce(ii) + 1, nnce(ii) + nce(ii)
          j1 = icvir(jj)
          if (latom_loc(j1)) then
            j = nfirst(j1) - 1
            do jx = 1, iorbs(j1)
              loopii = loopii + 1
              j = j + 1
              sum = sum + ws(j) * cvir(loopii)
            end do
          else
            loopii = loopii + iorbs(j1)
          end if
        end do
        maxov = max(maxov, abs(sum))
        call adjvec (cvir, cvir_dim, icvir, icvir_dim, nnce, nce, nvirtual, &
             & ncvir, i, iorbs, cvir, cvir_dim, icvir, icvir_dim, nnce, &
             & nce, nvirtual, ncvir, ii, sum, iused, sumtot)
      end do
      !
      !   EVALUATE THE VIRTUAL-OCCUPIED LMO OVERLAPS
      !
      do ii = 1, noccupied
        j1 = icocc(nncf(ii)+1)
        sum = 0.d0
        loopii = ncocc(ii)
        do jj = nncf(ii) + 1, nncf(ii) + ncf(ii)
          j1 = icocc(jj)
          if (latom_loc(j1)) then
            j = nfirst(j1) - 1
            do jx = 1, iorbs(j1)
              loopii = loopii + 1
              j = j + 1
              sum = sum + ws(j) * cocc(loopii)
            end do
          else
            loopii = loopii + iorbs(j1)
          end if
        end do
        maxov = max(maxov, abs(sum))
        call adjvec (cvir, cvir_dim, icvir, icvir_dim, nnce, nce, nvirtual, &
             & ncvir, i, iorbs, cocc, cocc_dim, icocc, icocc_dim, nncf, &
             & ncf, noccupied, ncocc, ii, sum, iused, sumtot)
      end do
    end do
    do i = 1, noccupied
      loopi = ncocc(i)
      do j = 1, numat
        latom_loc(j) = .false.
      end do
      do jj = nncf(i) + 1, nncf(i) + ncf(i)
        j1 = icocc(jj)
        latom_loc(j1) = .true.
        j = nfirst(j1) - 1
        do jx = 1, iorbs(j1)
          loopi = loopi + 1
          j = j + 1
          ws(j) = cocc(loopi)
        end do
      end do
      !
      !    WS holds the LMO coefficients, in atom number order.
      !
      !  EVALUATE THE OCCUPIED-OCCUPIED LMO OVERLAPS
      !
      do ii = i + 1, noccupied
        j1 = icocc(nncf(ii)+1)
        sum = 0.d0
        loopii = ncocc(ii)
        do jj = nncf(ii) + 1, nncf(ii) + ncf(ii)
          j1 = icocc(jj)
          if (latom_loc(j1)) then
            j = nfirst(j1) - 1
            do jx = 1, iorbs(j1)
              loopii = loopii + 1
              j = j + 1
              sum = sum + ws(j) * cocc(loopii)
            end do
          else
            loopii = loopii + iorbs(j1)
          end if
        end do
        maxov = max(maxov, abs(sum))
        call adjvec (cocc, cocc_dim, icocc, icocc_dim, nncf, ncf, &
             & noccupied, ncocc, i, iorbs, cocc, cocc_dim, icocc, &
             & icocc_dim, nncf, ncf, noccupied, ncocc, ii, sum, &
			 & iused, sumtot)
      end do
    end do
    if (mozyme_section_timers_enabled()) then
      ! Profile aid: how far the LMOs had drifted from orthogonality before this pass.
      write (iw, '(1x,a,es12.4,a,es12.4)') '[MOZYME reorth] max_overlap=', maxov, &
        ' sumtot=', sumtot
      call flush(iw)
    end if
    deallocate (latom_loc, iused)
1100 continue
end subroutine reorth

! Profile aid: how orthonormal the LMOs are at the end of an SCF (CPU or GPU):
! the largest |<i|j>| over all LMO pairs that share atoms and the largest
! |1 - <i|i>|.  Same array conventions as reorth; nothing is modified.
subroutine mozyme_lmo_orthogonality_report()
    use molkst_C, only: numat, norbs
    use MOZYME_C, only : nvirtual, noccupied, ncf, nce, nncf, nnce, ncocc, ncvir, &
       & iorbs, icocc, icvir, cocc, cvir
    use common_arrays_C, only : nfirst
    use chanel_C, only: iw
    implicit none
    double precision, allocatable :: ws(:)
    logical, allocatable :: latom(:)
    integer :: i, alloc_stat
    double precision :: maxoff, maxnorm, norm
    allocate (ws(norbs), latom(numat), stat=alloc_stat)
    if (alloc_stat /= 0) return
    ws = 0.d0
    latom = .false.
    maxoff = 0.d0
    maxnorm = 0.d0
    do i = 1, noccupied
      call scatter(ncf(i), nncf(i), ncocc(i), icocc, cocc, .true.)
      norm = overlap_with(ncf(i), nncf(i), ncocc(i), icocc, cocc)
      maxnorm = max(maxnorm, abs(1.d0 - norm))
      call scan_set(i + 1, noccupied, ncf, nncf, ncocc, icocc, cocc)
      call scan_set(1, nvirtual, nce, nnce, ncvir, icvir, cvir)
      call scatter(ncf(i), nncf(i), ncocc(i), icocc, cocc, .false.)
    end do
    do i = 1, nvirtual
      call scatter(nce(i), nnce(i), ncvir(i), icvir, cvir, .true.)
      norm = overlap_with(nce(i), nnce(i), ncvir(i), icvir, cvir)
      maxnorm = max(maxnorm, abs(1.d0 - norm))
      call scan_set(i + 1, nvirtual, nce, nnce, ncvir, icvir, cvir)
      call scatter(nce(i), nnce(i), ncvir(i), icvir, cvir, .false.)
    end do
    write (iw, '(1x,a,es12.4,a,es12.4)') '[MOZYME LMO orthogonality] max_offdiag=', maxoff, &
      ' max_norm_err=', maxnorm
    call flush(iw)
    deallocate (ws, latom)
contains
    ! Put (or remove) LMO (nc atoms from base_ic, coefficients from base_c) into ws.
    subroutine scatter(nc, base_ic, base_c, ic, c, on)
      integer, intent(in) :: nc, base_ic, base_c
      integer, intent(in) :: ic(*)
      double precision, intent(in) :: c(*)
      logical, intent(in) :: on
      integer :: jj, j1, jx, j, loop
      loop = base_c
      do jj = base_ic + 1, base_ic + nc
        j1 = ic(jj)
        j = nfirst(j1) - 1
        do jx = 1, iorbs(j1)
          loop = loop + 1
          j = j + 1
          if (on) then
            ws(j) = c(loop)
          else
            ws(j) = 0.d0
          end if
        end do
        latom(j1) = on
      end do
    end subroutine scatter

    double precision function overlap_with(nc, base_ic, base_c, ic, c)
      integer, intent(in) :: nc, base_ic, base_c
      integer, intent(in) :: ic(*)
      double precision, intent(in) :: c(*)
      integer :: jj, j1, jx, j, loop
      overlap_with = 0.d0
      loop = base_c
      do jj = base_ic + 1, base_ic + nc
        j1 = ic(jj)
        if (latom(j1)) then
          j = nfirst(j1) - 1
          do jx = 1, iorbs(j1)
            loop = loop + 1
            j = j + 1
            overlap_with = overlap_with + ws(j) * c(loop)
          end do
        else
          loop = loop + iorbs(j1)
        end if
      end do
    end function overlap_with

    subroutine scan_set(first, last, nc, nnc, ncmo, ic, c)
      integer, intent(in) :: first, last
      integer, intent(in) :: nc(*), nnc(*), ncmo(*), ic(*)
      double precision, intent(in) :: c(*)
      integer :: ii
      double precision :: s
      do ii = first, last
        s = overlap_with(nc(ii), nnc(ii), ncmo(ii), ic, c)
        maxoff = max(maxoff, abs(s))
      end do
    end subroutine scan_set
end subroutine mozyme_lmo_orthogonality_report
