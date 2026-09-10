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

subroutine hcore_for_MOZYME ()
  use molkst_C, only: numat, norbs, id, numcal, n2elec, moperr, &
     & enuclr, l1u, l2u, l3u, keywrd, efield, mpack, cutofp
  use cosmo_C, only : useps,  phinet, qscnet, qdenet
  use linear_cosmo, only : addnucz
  use chanel_C, only: iw
  use funcon_C, only: a0, ev
  use overlaps_C, only : cutof1, cutof2
  use parameters_C, only: tore, dd, natorb
  use common_arrays_C, only: h, coord, nat, w, wj => w, wk, uspd, tvec
  use MOZYME_C, only : semidr, direct, cutofs, parth, &
    iorbs, jopt, mode, numred, refnuc
  use mozyme_section_timers, only : mozyme_section_timer_begin, mozyme_section_timer_end
  use mozyme_gpu_gradient, only : mozyme_gpu_hcore_enabled, mozyme_gpu_hcore_check_enabled, &
    mozyme_gpu_hcore_run, mozyme_gpu_device_pair
  implicit none
  double precision :: hcore_timer
  ! GPU evaluation of the sp-sp block pairs (h1elec + rotate); the CPU loop
  ! below skips them when gpu_block_pairs is true and the device adds them
  ! afterwards.  gpu_check keeps a CPU reference for those pairs and reports
  ! the difference.
  logical :: gpu_block_pairs, gpu_point_pairs, gpu_check, gpu_verbose
  double precision, allocatable :: h_ref(:)
  double precision :: gpu_enuc, gpu_ms, enuc_ref, h_diff
  integer :: gpu_code, gpu_pairs, gpu_d_pairs, env_stat
  character(len=8) :: env_dbg
!
  character (len=248) :: tmpkey
  logical :: calci, calcij, calcj, fldon
  logical, save :: debug
  integer :: i1, i2, ii, im1, io1, ione, ired, j, j1, jj, jo1, jred, k, &
 & krmax, kro, ks, mm, nj, kr, i, ni, itemp, imol=0
  double precision :: const, enuc, fldcon, fnuc, half, hterme, xf, yf, zf, xj(3)
  double precision, parameter :: eps = 1.d-10
  double precision, dimension (45) :: e1b, e2a
  double precision, dimension (2025) :: wjd, wkd
  double precision, dimension (9, 9) :: di, dibits
  double precision, external :: reada
  integer, external :: ijbo
  fldcon = 0.d0
  fnuc = 0.d0
  debug = (Index (keywrd, " HCORE") /= 0)
  call mozyme_section_timer_begin('hcore_add_more_interactions', hcore_timer)
  call add_more_interactions()
  call mozyme_section_timer_end('hcore_add_more_interactions', hcore_timer)
  if (moperr) return
  if (imol /= numcal) then
    if (index(keywrd, " SILENT") == 0 .and. (debug .or. id /= 0)) then
      write (iw, "(A,31X,F12.6,A)") " Overlap Cutoff Distance:", Sqrt(cutofs), " CUTOFS"
      write (iw, "(A,F12.6,A)") &
       & " Cutoff for quadrupolar and higher 2-electron integrals:", Sqrt(cutof2), " CUTOF2"
      write (iw, "(A,26X,F12.6,A)") " Cutoff for dipolar integrals:", Sqrt(cutof1), " CUTOF1"
      if (id /= 0) &
          write (iw, "(A,39X,F12.6,A)") " Madelung cutoff:", cutofp, " CUTOFP"
    end if
    imol = numcal
    xf = 0.d0
    yf = 0.d0
    zf = 0.d0
    tmpkey = trim(keywrd)
    i = Index (tmpkey, " FIELD(") + Index (tmpkey, " FIELD=(")
    if (i /= 0) then
       !
       !   ERASE ALL TEXT FROM TMPKEY EXCEPT FIELD DATA
       !
      tmpkey (:i) = " "
      itemp = Index (tmpkey, ")")
      tmpkey (itemp:) = " "
       !
       !   READ IN THE EFFECTIVE FIELD IN X,Y,Z COORDINATES
       !
      xf = reada (tmpkey, i)
      i = Index (tmpkey, ",")
      if (i /= 0) then
        tmpkey (i:i) = " "
        yf = reada (tmpkey, i)
        i = Index (tmpkey, ",")
        if (i /= 0) then
          tmpkey (i:i) = " "
          zf = reada (tmpkey, i)
        end if
      end if
      write (iw, "(/10X,'THE ELECTRIC FIELD IS',3F10.5, ' VOLTS/ANGSTROM',&
       &/)") xf, yf, zf
    end if
    !
    !        CONST = Ao/(8h)  (h=Hartree = eV/atomic unit)
    !
    const = a0 / ev
    !
    efield(1) = xf * const
    efield(2) = yf * const
    efield(3) = zf * const
  end if
  ione = 1
  if (id /= 0) ione = 0
  if (mode == -1) then
    h(:mpack) = -h(:mpack)
    enuclr = -enuclr
  else if (mode == 0) then
    enuclr = 0.d0
  else
    h = parth
    enuclr = refnuc
  end if
  krmax = n2elec + 101
  fldon = .false.
  if (Abs (efield(1)) > eps .or. Abs (efield(2)) > eps .or. Abs (efield(3)) > eps) then
    fldcon = eV/a0 ! = 51.42
    fldon = .true.
  end if
 !
  kr = 1
 !
  mm = 0
  ired = 1
  if (mode == 0) then
    !
    !   ZERO OUT THE H MATRIX
    !
    h(1:mpack) = 0.d0
  end if
  gpu_block_pairs = (id == 0 .and. mode == 0 .and. .not. fldon .and. mozyme_gpu_hcore_enabled())
  ! Point/dipole pairs (outer1/outer2) store no W in direct mode, so they can
  ! go to the device too: only e1b/e2a and enuc come back.
  gpu_point_pairs = gpu_block_pairs .and. semidr
  gpu_check = gpu_block_pairs .and. mozyme_gpu_hcore_check_enabled()
  call mozyme_section_timer_begin('hcore_pair_loop', hcore_timer)
  do i = 1, numat

    calci = (jopt(ired) == i)
    if (calci .and. ired < numred) then
      ired = ired + 1
    end if
    ni = nat(i)
    if (mode == 0) then
        !
        ! FILL THE DIAGONALS, AND OFF-DIAGONALS ON THE SAME ATOM
        !
      i2 = ijbo (i, i)
      do i1 = 1, iorbs(i)
        do j1 = 1, i1
          i2 = i2 + 1
          h(i2) = 0.d0
          if (fldon) then
            io1 = i1 - 1
            jo1 = j1 - 1
            if ((jo1 == 0) .and. (io1 == 1)) then
              hterme = -a0 * dd(ni) * efield(1) * fldcon
              h(i2) = hterme
            end if
            if ((jo1 == 0) .and. (io1 == 2)) then
              hterme = -a0 * dd(ni) * efield(2) * fldcon
              h(i2) = hterme
            end if
            if ((jo1 == 0) .and. (io1 == 3)) then
              hterme = -a0 * dd(ni) * efield(3) * fldcon
              h(i2) = hterme
            end if
          end if
        end do
        mm = mm + 1
        h(i2) = uspd(mm)
        if (fldon) then
          fnuc = -(efield(1)*coord(1, i) + efield(2)*coord(2, i) + &
               & efield(3)*coord(3, i)) * fldcon
          h(i2) = h(i2) + fnuc
        end if
      end do
    end if
    if (fldon) then
      enuclr = enuclr - fnuc * tore(nat(i))
    end if
    !
    !   FILL THE ATOM-OTHER ATOM ONE-ELECTRON MATRIX<PSI(LAMBDA)|PSI(SIGMA)>
    !
    jred = 1
    im1 = i - ione
    do j = 1, im1
      half = 1.d0
      if (i == 46 .and. j < 888) then
    enuclr = enuclr
   end if
      if (i == j) half = 0.5d0
      calcj = (jopt(jred) == j)
      if (calcj .and. jred < numred) jred = jred + 1
      calcij = (calci .or. calcj .or. mode == 0)
      nj = nat(j)
      if (id == 0) then
        !
        !   Molecular system
        !
        if (ijbo(i, j) >= 0) then
          if (calcij .and. gpu_block_pairs .and. mozyme_gpu_device_pair(iorbs(i), iorbs(j))) then
            ! Evaluated on the device after the loop (kr is not advanced for
            ! block pairs, so nothing else changes here).  The diagonal-block
            ! update below still runs with these zeros (up to 45 entries for d atoms).
            e1b(1:45) = 0.d0
            e2a(1:45) = 0.d0
          else if (calcij) then
            call h1elec (ni, nj, coord(1, i), coord(1, j), di)
            ii = ijbo (i, j)
            if (i == j) then
              do i1 = 1, iorbs(i)
                do j1 = 1, i1
                  ii = ii + 1
                  h(ii) = h(ii) + di(i1, j1)
                end do
              end do
            else
              do i1 = 1, iorbs(i)
                do j1 = 1, iorbs(j)
                  ii = ii + 1
                  h(ii) = h(ii) + di(i1, j1)
                end do
              end do
            end if
            !
            ! CALCULATE THE TWO-ELECTRON INTEGRALS, W; THE ELECTRON
            ! NUCLEAR TERMS E1B AND E2A; AND THE NUCLEAR-NUCLEAR TERM ENUC.
            !
            call rotate (ni, nj, coord(1, i), coord(1, j), w(kr), i1, e1b, e2a, enuc)
            enuclr = enuclr + enuc
           !
          else if ( .not. direct) then
            kr = kr + (natorb(ni)*(natorb(ni)+1))/2 * (natorb(nj)*(natorb(nj)+1))/2
          end if
        else if (ijbo(i, j) ==-2) then
          if (calcij .and. gpu_point_pairs) then
            e1b(1:45) = 0.d0
            e2a(1:45) = 0.d0
          else if (calcij) then
            call outer2 (ni, nj, coord(1, i), coord(1, j), w(kr), kr, e1b, e2a, enuc, id, semidr)
            enuclr = enuclr + enuc
          else if ( .not. semidr) then
            if (natorb(ni)*natorb(nj) > 0) then
              if (natorb(ni) > 1) then
                if (natorb(nj) > 1) then
                  kr = kr + 7
                else
                  kr = kr + 4
                end if
              else if (natorb(nj) > 1) then
                kr = kr + 4
              else
                kr = kr + 1
              end if
            end if
          end if
        else if (calcij .and. gpu_point_pairs) then
          e1b(1:45) = 0.d0
          e2a(1:45) = 0.d0
        else if (calcij) then
          call outer1 (ni, nj, coord(1, i), coord(1, j), w(kr), kr, e1b, e2a, enuc, 0, semidr)
          enuclr = enuclr + enuc
        else if ( .not. semidr) then
          if (natorb(ni)*natorb(nj) > 0) kr = kr + 1
        end if
        if (calcij) then
          !
          !   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM I.
          !
          ii = ijbo (i, i)
          j1 = (iorbs(i)*(iorbs(i)+1)) / 2
          do i1 = 1, j1
            ii = ii + 1
            h(ii) = h(ii) + e1b(i1) * half
          end do
          !
          !   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM J.
          !
          ii = ijbo (j, j)
          j1 = (iorbs(j)*(iorbs(j)+1)) / 2
          do i1 = 1, j1
            ii = ii + 1
            h(ii) = h(ii) + e2a(i1) * half
          end do
        end if
        if (kr > krmax-100 .and. i /= numat) then
          write (iw,*) kr, krmax
          write (iw, "(' Running out of storage for W in HCORE ')")
          write (iw,*) " NUMBER OF ATOMS CALCULATED FOR 'W':", i
          write (iw,*) " NUMBER OF ATOMS IN SYSTEM:", numat
          call mopend ("Running out of storage for W in HCORE")
          return
        end if
      else
        !
        !   Solid-state system
        !
        if (ijbo(i, j) >= 0) then
          if (calcij) then
            di = 0.D0
            do ii = -l1u, l1u
              do jj = -l2u, l2u
                do k = -l3u, l3u
                  xj = coord(:,j) + tvec(:,1)*ii + tvec(:,2)*jj + tvec(:,3)*k
                  call h1elec (ni, nj, coord(1,i), xj, dibits)
                  di = di + dibits
                end do
              end do
            end do
            ii = ijbo (i, j)
            if (i == j) then
              do i1 = 1, iorbs(i)
                do j1 = 1, i1
                  ii = ii + 1
                  h(ii) = h(ii) + di(i1, j1)
                end do
              end do
            else
              do i1 = 1, iorbs(i)
                do j1 = 1, iorbs(j)
                  ii = ii + 1
                  h(ii) = h(ii) + di(i1, j1)
                end do
              end do
            end if
            !
            ! CALCULATE THE TWO-ELECTRON INTEGRALS, W;
            ! THE ELECTRON NUCLEAR TERMS E1B AND E2A;
            ! AND THE NUCLEAR-NUCLEAR TERM ENUC.
            !
            kro = kr
            call solrot (ni, nj, coord(1,i), coord(1,j), wjd, wkd, kr, e1b, e2a, enuc)
            w(kro:kr - 1) = wjd(:kr-kro)
            wk(kro:kr - 1) = wkd(:kr-kro)
            enuclr = enuclr + enuc
          else if (natorb(ni) == 1) then
            if (natorb(nj) == 1) then
              kr = kr + 1
            else
              kr = kr + 10
            end if
          else if (natorb(nj) == 1) then
            kr = kr + 10
          else
            kr = kr + 100
          end if
        else if (ijbo(i, j) ==-2) then
          if (calcij) then
            ks = kr
            call outer2 (ni, nj, coord(1, i), coord(1, j), wj(kr), kr, &
                 & e1b, e2a, enuc, id, semidr)
            do k = ks, kr - 1
              wk(k) = 0.d0
            end do
            enuclr = enuclr + enuc
          else if (natorb(ni)*natorb(nj) > 0) then
            if (natorb(ni) > 1) then
              if (natorb(nj) > 1) then
                kr = kr + 7
              else
                kr = kr + 4
              end if
            else if (natorb(nj) > 1) then
              kr = kr + 4
            else
              kr = kr + 1
            end if
          end if
        else if (calcij) then
          wk(kr) = 0.d0
          call outer1 (ni, nj, coord(1, i), coord(1, j), wj(kr), kr, &
               & e1b, e2a, enuc, id, semidr)
          enuclr = enuclr + enuc
        else if (natorb(ni)*natorb(nj) > 0) then
          kr = kr + 1
        end if
        if (calcij) then
           !
           !   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM I.
           !
          ii = ijbo (i, i)
          j1 = (iorbs(i)*(iorbs(i)+1)) / 2
          do i1 = 1, j1
            ii = ii + 1
            h(ii) = h(ii) + e1b(i1) * half
          end do
           !
           !   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM J.
           !
          ii = ijbo (j, j)
          j1 = (iorbs(j)*(iorbs(j)+1)) / 2
          do i1 = 1, j1
            ii = ii + 1
            h(ii) = h(ii) + e2a(i1) * half
          end do
        end if
        if (kr > krmax-100 .and. i /= numat) then
          write (iw,*) kr, krmax
          write (iw, "(' Running out of storage for W in HCORE ')")
          write (iw,*) " NUMBER OF ATOMS CALCULATED FOR 'W':", i
          write (iw,*) " NUMBER OF ATOMS IN SYSTEM:", numat
          call mopend ("Running out of storage for W in HCORE")
        end if
      end if
    end do
    ii = iorbs(i)
    ii = (ii*(ii+1)) / 2
    if (id /= 0) then
      do i1 = kr, kr + ii * ii - 1
        wk(i1) = 0.d0
      end do
    end if
    if (ii /= 0) then
      call wstore (w(kr), kr, ni, ii)
    end if
  end do
  call mozyme_section_timer_end('hcore_pair_loop', hcore_timer)
  if (gpu_block_pairs) then
    env_dbg = ' '
    call get_environment_variable('MOPAC_GPU_DEBUG', env_dbg, status=env_stat)
    gpu_verbose = (env_stat == 0 .and. len_trim(env_dbg) > 0 .and. env_dbg(1:1) /= '0')
    if (gpu_check) then
      allocate(h_ref(mpack))
      h_ref(1:mpack) = h(1:mpack)
      call cpu_sp_block_pairs(h_ref, enuc_ref)
    end if
    call mozyme_section_timer_begin('hcore_gpu_pairs', hcore_timer)
    call mozyme_gpu_hcore_run(numat, coord, h, gpu_enuc, gpu_point_pairs, gpu_code, gpu_ms, gpu_pairs, gpu_d_pairs)
    call mozyme_section_timer_end('hcore_gpu_pairs', hcore_timer)
    if (gpu_code == 0) then
      enuclr = enuclr + gpu_enuc
      if (gpu_check) then
        h_diff = maxval(abs(h(1:mpack) - h_ref(1:mpack)))
        write (iw, '(1x,a,i0,a,i0,a,f10.3,a,es12.4,a,es12.4,a)') '[MOZYME GPU hcore] check pairs=', &
          gpu_pairs, ' d_pairs_cpu=', gpu_d_pairs, ' ms=', gpu_ms, ' max_abs_dh=', h_diff, &
          ' denuc=', gpu_enuc - enuc_ref, ' eV'
        call flush(iw)
      else if (gpu_verbose) then
        write (iw, '(1x,a,i0,a,i0,a,f10.3)') '[MOZYME GPU hcore] success pairs=', gpu_pairs, &
          ' d_pairs_cpu=', gpu_d_pairs, ' ms=', gpu_ms
        call flush(iw)
      end if
    else
      if (gpu_verbose) then
        write (iw, '(1x,a,i0)') '[MOZYME GPU hcore] fallback_cpu code=', gpu_code
        call flush(iw)
      end if
      call cpu_sp_block_pairs(h, enuc_ref)
      enuclr = enuclr + enuc_ref
    end if
    if (allocated(h_ref)) deallocate(h_ref)
  end if
 !
 !
  if (mode == -1) then
    parth(:mpack) = -h(:mpack)
    refnuc = -enuclr
  end if
  if (useps) then
    ! In the following routine the dielectric correction to the core-core-
    ! interaction is added to ENUCLR (just set arrays to zero)

    call addnucz (phinet, qscnet, qdenet)

  end if
  kr = kr - 1
 !
 !
  if (debug) then
    write (iw, "(//10X,'ONE-ELECTRON MATRIX FROM HCORE')")
    if (mode ==-1) then
      write (iw, "(10X,A)") " AFTER REMOVAL OF TERMS FOR MOVING ATOMS"
    end if
    if (mode == 1) then
      write (iw, "(10X,A)") " AFTER ADDITION OF TERMS FOR MOVING ATOMS"
    end if
    call vecprt_for_MOZYME (h, norbs)
    if (kr > 2000) then
      write (iw,*) " THE TWO-ELECTRON MATRIX IS TOO LARGE TO PRINT"
    end if
    j = Min (kr, 2000)
    if (id == 0) then
      write (iw, "(//10X,'TWO-ELECTRON MATRIX IN HCORE'/)")
10000   format (10 f8.4)
      write (iw, 10000) (w(i), i=1, j)
    else
      write (iw, "(//10X,'TWO-ELECTRON J MATRIX IN HCORE'/)")
      write (iw, 10000) (wj(i), i=1, j)
      write (iw, "(//10X,'TWO-ELECTRON K MATRIX IN HCORE'/)")
      write (iw, 10000) (wk(i), i=1, j)
    end if
  end if
contains

  ! CPU reference / fallback for the device block pairs skipped in the main loop
  ! (mode == 0, id == 0): h1elec into the off-diagonal block, rotate's e1b/e2a
  ! into the diagonal blocks, enuc summed into enuc_sum.
  subroutine cpu_sp_block_pairs(hh, enuc_sum)
    implicit none
    double precision, intent(inout) :: hh(*)
    double precision, intent(out) :: enuc_sum
    integer :: ia, ja, na, nb, ka, i1, j1, kdum
    double precision :: e1b_l(45), e2a_l(45), enuc_l, di_l(9, 9), w_l(2025)
    enuc_sum = 0.d0
    do ia = 2, numat
      na = nat(ia)
      do ja = 1, ia - 1
        ka = ijbo(ia, ja)
        nb = nat(ja)
        if (ka < 0) then
          if (.not. gpu_point_pairs) cycle
          kdum = 1
          if (ka == -2) then
            call outer2 (na, nb, coord(1, ia), coord(1, ja), w_l, kdum, e1b_l, e2a_l, enuc_l, id, semidr)
          else
            call outer1 (na, nb, coord(1, ia), coord(1, ja), w_l, kdum, e1b_l, e2a_l, enuc_l, 0, semidr)
          end if
          enuc_sum = enuc_sum + enuc_l
          ka = ijbo(ia, ia)
          do i1 = 1, (iorbs(ia)*(iorbs(ia)+1))/2
            hh(ka + i1) = hh(ka + i1) + e1b_l(i1)
          end do
          ka = ijbo(ja, ja)
          do i1 = 1, (iorbs(ja)*(iorbs(ja)+1))/2
            hh(ka + i1) = hh(ka + i1) + e2a_l(i1)
          end do
          cycle
        end if
        if (.not. mozyme_gpu_device_pair(iorbs(ia), iorbs(ja))) cycle
        call h1elec (na, nb, coord(1, ia), coord(1, ja), di_l)
        do i1 = 1, iorbs(ia)
          do j1 = 1, iorbs(ja)
            ka = ka + 1
            hh(ka) = hh(ka) + di_l(i1, j1)
          end do
        end do
        kdum = 1
        call rotate (na, nb, coord(1, ia), coord(1, ja), w_l, kdum, e1b_l, e2a_l, enuc_l)
        enuc_sum = enuc_sum + enuc_l
        ka = ijbo(ia, ia)
        do i1 = 1, (iorbs(ia)*(iorbs(ia)+1))/2
          hh(ka + i1) = hh(ka + i1) + e1b_l(i1)
        end do
        ka = ijbo(ja, ja)
        do i1 = 1, (iorbs(ja)*(iorbs(ja)+1))/2
          hh(ka + i1) = hh(ka + i1) + e2a_l(i1)
        end do
      end do
    end do
  end subroutine cpu_sp_block_pairs

end subroutine hcore_for_MOZYME
