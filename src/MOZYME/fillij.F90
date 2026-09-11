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

! Changed entries of a fillij update pass (promotions of atom pairs), sent to
! the device copy of nijbo as a patch in GPU builds.  Kept in a module (not an
! internal procedure of fillij) so the pair loop is not pessimised by host
! association.
module fillij_changes_C
  implicit none
  ! Fixed capacity: 1M entries (12 MB); an update pass that changes more
  ! than that falls back to a full re-upload of nijbo (chg_overflow).
  integer, parameter :: nchg_cap = 1048576
  integer, allocatable, save :: chg_i(:), chg_j(:), chg_v(:)
  integer, save :: nchg = 0
  logical, save :: chg_overflow = .false.
contains
  subroutine fillij_changes_reset()
    implicit none
    integer :: stat
    nchg = 0
    chg_overflow = .false.
    if (.not. allocated(chg_i)) then
      allocate (chg_i(nchg_cap), chg_j(nchg_cap), chg_v(nchg_cap), stat=stat)
      if (stat /= 0) then
        chg_overflow = .true.
        if (allocated(chg_i)) deallocate(chg_i)
        if (allocated(chg_j)) deallocate(chg_j)
        if (allocated(chg_v)) deallocate(chg_v)
      end if
    end if
  end subroutine fillij_changes_reset
end module fillij_changes_C

  subroutine fillij (count)
#ifdef GPU
      use iso_c_binding, only: c_int, c_double
#endif
      use molkst_C, only: numat, natoms, cutofp, id, n2elec, l1u, l2u, l3u, keywrd, mpack, &
        ispd, line
      use common_arrays_C, only : tvec, coord
      use chanel_C, only: iw
      use MOZYME_C, only : cutofs, direct, semidr, &
        nijbo, lijbo, iijj, iij, ij_dim, ijall, numij, morb, iorbs
      use overlaps_C, only : cutof1, cutof2
      use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required
      use mozyme_section_timers, only: mozyme_section_timer_begin, mozyme_section_timer_end
      use fillij_changes_C, only: fillij_changes_reset, nchg_cap, &
        nchg, chg_overflow, chg_i, chg_j, chg_v
!
      implicit none
      !
      !***********************************************************************
      !
      ! fillij creates either the array ijbo, if there is enough memory, or
      ! the arrays iijj, ijall, and iij, if there is not enough memory.
      !
      !***********************************************************************
      logical :: count
!
      integer :: i, ii, iloop, io, ip, j, jloop, jo, jp, &
           & kp, lp, ix
      double precision :: r, rmin, rr, x1, x2, x3
      save :: ix
      logical :: first
      integer :: ib, jb
      integer, parameter :: nblk = 64
      double precision :: fillij_timer
      logical, save :: strict_nijbo_marker_written = .false.
      double precision, dimension (3) :: xj
      double precision, external :: reada
      external :: mozyme_gpu_strict_abort
#ifdef GPU
      integer(c_int) :: gpu_code, gpu_mpack, gpu_n2elec, gpu_ij_dim
      interface
        function mopac_cuda_mozyme_fillij_count(numat_c, id_c, l1u_c, l2u_c, l3u_c, ispd_c, &
            direct_c, semidr_c, cutof1_c, cutof2_c, coord_c, tvec_c, iorbs_c, mpack_c, &
            n2elec_c, ij_dim_c) bind(C,name='mopac_cuda_mozyme_fillij_count') result(code)
          use iso_c_binding, only: c_int, c_double
          integer(c_int), value :: numat_c, id_c, l1u_c, l2u_c, l3u_c, ispd_c
          integer(c_int), value :: direct_c, semidr_c
          real(c_double), value :: cutof1_c, cutof2_c
          real(c_double) :: coord_c(3,*), tvec_c(3,*)
          integer(c_int) :: iorbs_c(*), mpack_c, n2elec_c, ij_dim_c
          integer(c_int) :: code
        end function mopac_cuda_mozyme_fillij_count

        function mopac_cuda_mozyme_fillij_nijbo(numat_c, id_c, l1u_c, l2u_c, l3u_c, ispd_c, &
            direct_c, semidr_c, cutof1_c, cutof2_c, coord_c, tvec_c, iorbs_c, nijbo_c, &
            mpack_c, n2elec_c, ij_dim_c) bind(C,name='mopac_cuda_mozyme_fillij_nijbo') result(code)
          use iso_c_binding, only: c_int, c_double
          integer(c_int), value :: numat_c, id_c, l1u_c, l2u_c, l3u_c, ispd_c
          integer(c_int), value :: direct_c, semidr_c
          real(c_double), value :: cutof1_c, cutof2_c
          real(c_double) :: coord_c(3,*), tvec_c(3,*)
          integer(c_int) :: iorbs_c(*), nijbo_c(numat_c,*), mpack_c, n2elec_c, ij_dim_c
          integer(c_int) :: code
        end function mopac_cuda_mozyme_fillij_nijbo
        ! Announces a (re)filled host nijbo to the shared device cache
        ! (cuda_wrappers.cu), so hcore / Fock plan / resident SCF re-upload it
        ! only after a fillij pass instead of on every use.
        subroutine mopac_cuda_mozyme_nijbo_touch(nijbo_c, numat_c) bind(C, name='mopac_cuda_mozyme_nijbo_touch')
          use iso_c_binding, only: c_int
          integer(c_int), intent(in) :: nijbo_c(*)
          integer(c_int), value :: numat_c
        end subroutine mopac_cuda_mozyme_nijbo_touch
        ! Update pass: only n entries changed; patch the device copy instead.
        subroutine mopac_cuda_mozyme_nijbo_patch(nijbo_c, numat_c, n_c, pi_c, pj_c, pv_c) &
            bind(C, name='mopac_cuda_mozyme_nijbo_patch')
          use iso_c_binding, only: c_int
          integer(c_int), intent(in) :: nijbo_c(*), pi_c(*), pj_c(*), pv_c(*)
          integer(c_int), value :: numat_c, n_c
        end subroutine mopac_cuda_mozyme_nijbo_patch
      end interface
#endif
!
      if (.not. allocated(nijbo)) then
        ix = 0
        n2elec = 0
        mpack = 0
        first = .true.
        morb = 4
        if (ispd > 0) morb = 9
      else
!
!   Array nijbo was already allocated, therefore it was filled.  Do not reset mpack1, instead add any new interactions
!   to array nijbo
!
        first = .false.
      end if
!
      if (.not. count) then
        ! Allocate either nijbo or iijj and ijall
        if ( .not. allocated(nijbo)) then
          i = 1
          if (lijbo) then
!
!  Try to allocate memory for a simple square array
!
            allocate (nijbo(numat, numat), stat = i)
            if (i == 0) nijbo(:, :) = 0
          end if
          if (i /= 0) then

!
!  Array nijbo could not be created.  Therefore set lijbo false and
!  create smaller, but more CPU intensive, arrays
!
            if (mozyme_gpu_scf_no_fallback_required()) then
              call mozyme_gpu_strict_abort('strict_nijbo_alloc_failed', &
                "MOZYME GPU strict resident SCF requires nijbo allocation")
              return
            end if
            lijbo = .false.
            allocate (iijj(ij_dim), ijall(ij_dim), iij(natoms), numij(natoms), &
               & stat = i)
            if (i /= 0) then
              call mopend("There is not enough memory to create the ijbo-type arrays")
              return
            end if
            iijj(:) = 0
            ijall(:) = 0
            iij(:) = 0
            numij(:) = 0
          end if
        end if
      else
        lijbo = .true.
      end if
      if (mozyme_gpu_scf_no_fallback_required() .and. allocated(nijbo) .and. &
          lijbo .and. .not. strict_nijbo_marker_written) then
        write(iw,'(1x,a,1x,a,1x,i0)') &
          '[MOZYME GPU SCF]', 'compact_index_route=0 use_nijbo=', 1
        call flush(iw)
        strict_nijbo_marker_written = .true.
      end if
      !
      !   Set CUTOF values  CUTOF2 = First cutoff (NDDO-dipolar)
      !                     CUTOF1 = Second cutoff (dipolar-point charge)
      !

      cutof1 = 10.d0 ** 2
      if (cutofp < 100.d0) cutof1 = cutofp ** 2 + 1.d-4
      if (numat < 30)      cutof1 = 1.d6 + 10
      cutof2 = 9.9d0 ** 2
      if (cutofp < 100.d0) cutof2 = cutofp ** 2
      if (numat < 30)      cutof2 = 1.d6
      line = trim(keywrd)
      if (index(line," GEO_DAT") /= 0) then
        i = index(line," GEO_DAT") + 9
        j = index(line(i + 10:),'" ') + i + 9
        line(i:j) = " "
      end if

      i = Index (line, " CUTOFF=")
      if (i /= 0) then
        cutof1 = reada (line, i+8)
        cutof2 = (cutof1 - 1.d-1)**2
        cutof1 = cutof1**2
      else
        i = Index (line, " CUTOF1=")
        if (i /= 0) cutof1 = reada (keywrd, i+8) ** 2
        i = Index (line, " CUTOF2=")
        if (i /= 0) cutof2 = reada (keywrd, i+8) ** 2
      end if
      !
      i = Index (line, " CUTOFS=")
      if (i /= 0) then
        cutofs = reada (line, i+8) ** 2
      else
        cutofs = 7.d0 ** 2
      end if
      !
      !   Check that CUTOF1 is greater than CUTOF2
      !
      if (cutof1 < cutof2) then
        cutof1 = cutof2 + 1.d-4
        write (iw, "(/,A,F12.4,/)") " CUTOF1 WAS SET SMALLER THAN CUTOF2," // &
             & " RESET TO", Sqrt (cutof1)
      end if
      !#aab
      if ((Index (keywrd, " NODIRECT") /= 0) .or. (id /= 0) ) then
        direct = .false.
        semidr = .false.
      else if (Index (keywrd, " SEMIDIRECT") /= 0) then
        direct = .false.
        semidr = .true.
      else
        direct = .true.
        semidr = .true.
      end if
#ifdef GPU
      if (mozyme_gpu_scf_no_fallback_required()) then
        gpu_mpack = 0_c_int
        gpu_n2elec = 0_c_int
        gpu_ij_dim = 0_c_int
        if (count) then
          gpu_code = mopac_cuda_mozyme_fillij_count( &
            int(numat, c_int), int(id, c_int), int(l1u, c_int), &
            int(l2u, c_int), int(l3u, c_int), int(ispd, c_int), &
            merge(1_c_int, 0_c_int, direct), merge(1_c_int, 0_c_int, semidr), &
            cutof1, cutof2, coord, tvec, iorbs, gpu_mpack, gpu_n2elec, gpu_ij_dim)
        else
          if (.not. allocated(nijbo) .or. .not. lijbo) then
            call mozyme_gpu_strict_abort('strict_fillij_nijbo_missing', &
              'MOZYME GPU strict resident SCF requires GPU fillij with nijbo')
            return
          end if
          gpu_code = mopac_cuda_mozyme_fillij_nijbo( &
            int(numat, c_int), int(id, c_int), int(l1u, c_int), &
            int(l2u, c_int), int(l3u, c_int), int(ispd, c_int), &
            merge(1_c_int, 0_c_int, direct), merge(1_c_int, 0_c_int, semidr), &
            cutof1, cutof2, coord, tvec, iorbs, nijbo, gpu_mpack, gpu_n2elec, gpu_ij_dim)
        end if
        if (gpu_code /= 0_c_int) then
          write(iw,'(1x,a,1x,a,1x,i0)') &
            '[MOZYME GPU SCF]', 'fillij_gpu_failed code=', gpu_code
          call flush(iw)
          call mozyme_gpu_strict_abort('strict_fillij_gpu_failed', &
            'MOZYME GPU strict resident SCF could not build fillij on GPU')
          return
        end if
        mpack = int(gpu_mpack)
        n2elec = int(gpu_n2elec)
        if (count) ij_dim = int(gpu_ij_dim)
        write(iw,'(1x,a,1x,a,1x,l1,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
          '[MOZYME GPU SCF]', 'fillij_gpu=1 count=', count, 'mpack=', mpack, &
          'n2elec=', n2elec, 'ij_dim=', int(gpu_ij_dim)
        call flush(iw)
        if (.not. count .and. lijbo .and. allocated(nijbo)) &
          call mopac_cuda_mozyme_nijbo_touch(nijbo, int(numat, c_int))
        return
      end if
#else
      if (mozyme_gpu_scf_no_fallback_required()) then
        call mozyme_gpu_strict_abort('strict_fillij_gpu_unavailable', &
          'MOZYME GPU strict resident SCF requires a GPU fillij builder')
        return
      end if
#endif
      !#aab - end
      !
      rmin = 100.d0
      !
      !     Identify atom pairs involved in calculation.  This is done based
      !     on interatomic distance.
      !
      call mozyme_section_timer_begin('fillij_pairs', fillij_timer)
      call fillij_changes_reset()
      i = 0
      do iloop = 1, numat
          io = iorbs(iloop)
          i = i + 1
          x1 = coord(1, i)
          x2 = coord(2, i)
          x3 = coord(3, i)
          ii = (io*(io+1)) / 2
          if (first) n2elec = n2elec + ii * ii ! for one-centre two-electron integrals
          if (.not. lijbo .and. .not. count) then
            iij(i) = ix + 1
          end if
          j = 0
          do jloop = 1, iloop - 1
              jo = iorbs(jloop)
              j = j + 1
              if (id == 0) then
                r = (x1-coord(1, j)) ** 2 + (x2-coord(2, j)) ** 2 &
                     & + (x3-coord(3, j)) ** 2
              else
                r = 1.d10
                do ip = -l1u, l1u
                  do jp = -l2u, l2u
                    do kp = -l3u, l3u
                      do lp = 1, 3
                        xj(lp) = coord(lp, j) + tvec(lp, 1) * ip &
                             & + tvec(lp, 2) * jp + tvec(lp, 3) * kp
                      end do
                      rr = (x1-xj(1)) ** 2 + (x2-xj(2)) ** 2 + (x3-xj(3)) ** 2
                      r = Min (r, rr)
                    end do
                  end do
                end do
              end if

              if (r < cutof2) then
                ix = ix + 1
                if (.not. count) then
                  if (lijbo) then
                    !  nijbo is symmetric: read/write nijbo(j, i) (contiguous in the
                    !  inner j loop).  On the first pass only the lower triangle is
                    !  written here and the upper one is filled by the tiled copy
                    !  after the loop; strided nijbo(i, j) stores cost ~0.3 s per
                    !  pass for 7000 atoms.
                    if (first) then
                      nijbo(j, i) = mpack
                    else if (nijbo(j, i) < 0) then
                      nijbo(i, j) = mpack
                      nijbo(j, i) = mpack
                      ! record the promotion inline (no call inside the pair loop)
                      if (nchg < nchg_cap .and. .not. chg_overflow) then
                        nchg = nchg + 1
                        chg_i(nchg) = i
                        chg_j(nchg) = j
                        chg_v(nchg) = mpack
                      else
                        chg_overflow = .true.
                      end if
                    else
                      mpack = mpack - io * jo  !  prevent mpack from being incremented - the element of
                                                 !  nijbo was already set.
                    end if
                  else
                    iijj(ix) = mpack
                    ijall(ix) = j
                  end if
                end if
                mpack = mpack + io * jo
                !
                if ( .not. direct) then
                  n2elec = n2elec + (jo*(jo+1)) / 2 * ii
                end if
                if (r < rmin) then
                  rmin = r
                end if
              else if (r < cutof1) then
                !
                !   USE CHARGES AND DIPOLES
                !
                if ( .not. semidr) then
                  !
                  if (io > 1) then
                    if (jo > 1) then
                      n2elec = n2elec + 7
                    else
                      n2elec = n2elec + 4
                    end if
                  else if (jo > 1) then
                    n2elec = n2elec + 4
                  else
                    n2elec = n2elec + 1
                  end if
                end if !#aab - end
                !
                if (lijbo .and. (.not. count)) then
                  if (first) then
                    nijbo(j, i) = -2
                  else if (nijbo(j, i) == -1) then
                    nijbo(i, j) = -2
                    nijbo(j, i) = -2
                    if (nchg < nchg_cap .and. .not. chg_overflow) then
                      nchg = nchg + 1
                      chg_i(nchg) = i
                      chg_j(nchg) = j
                      chg_v(nchg) = -2
                    else
                      chg_overflow = .true.
                    end if
                  end if
                end if
                !#aab
              else
                !
                !  USE POINT CHARGE APPROXIMATION
                !
                if ( .not. semidr) then
                  n2elec = n2elec + 1
                end if
                !
                if (lijbo .and. (.not. count)) then
                  if (first) nijbo(j, i) = -1
                end if
                !#aab
            end if
          end do
          !
          ix = ix + 1
          if (.not. count) then
            if (lijbo) then
              if (first) then
                nijbo(i, i) = mpack
              else
                mpack = mpack - (io*(io+1)) / 2  !  prevent mpack from being incremented - the element of
                                                   !  nijbo was already set.
              end if
            else
              iijj(ix) = mpack
              ijall(ix) = i
              numij(i) = ix
            end if
          end if
          if (id /= 0) then
            n2elec = n2elec + ((io*(io+1))/2) ** 2
          end if
          mpack = mpack + (io*(io+1)) / 2
      end do
      if (first .and. lijbo .and. .not. count .and. numat > 1) then
        !
        !  Fill the upper triangle nijbo(i, j) = nijbo(j, i), i > j, in cache-sized
        !  tiles (contiguous stores along i).
        !
        do jb = 1, numat, nblk
          do ib = jb, numat, nblk
            do j = jb, min(jb + nblk - 1, numat)
              do i = max(ib, j + 1), min(ib + nblk - 1, numat)
                nijbo(i, j) = nijbo(j, i)
              end do
            end do
          end do
        end do
      end if
      call mozyme_section_timer_end('fillij_pairs', fillij_timer)
#ifdef GPU
      if (.not. count .and. lijbo .and. allocated(nijbo)) then
        if (first .or. chg_overflow) then
          call mopac_cuda_mozyme_nijbo_touch(nijbo, int(numat, c_int))
        else if (nchg > 0) then
          call mopac_cuda_mozyme_nijbo_patch(nijbo, int(numat, c_int), int(nchg, c_int), &
            chg_i, chg_j, chg_v)
        else
          call mopac_cuda_mozyme_nijbo_patch(nijbo, int(numat, c_int), 0_c_int, &
            chg_i, chg_j, chg_v)
        end if
      end if
#endif
      !
      if (count) then
        ij_dim = ix
      end if
      !
      !  Leave room for dummy atom integrals and for gradients
      !
      if (n2elec < 2025) then
        n2elec = 2025
      end if
      if (first) then
        n2elec = n2elec + 10
        if (direct .and. ispd == 0) n2elec = n2elec + 100
        if (direct .and. ispd /= 0) n2elec = n2elec + 2025
      end if
      if (id /= 0) then
        n2elec = n2elec * 2
      end if
    end subroutine fillij
