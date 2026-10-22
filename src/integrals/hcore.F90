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

      subroutine hcore()
!-----------------------------------------------
!   M o d u l e s
!-----------------------------------------------
      USE molkst_C, only : numcal, numat, norbs, id, l1u, l2u, l3u, &
      & keywrd, enuclr, n2elec, efield, mpack, line
      use common_arrays_C, only : nfirst, nlast, nat, uspd, &
      & coord, h, w, wk, tvec, wk_image
      use cosmo_C, only : useps
      USE funcon_C, only : a0, ev, fpc_9
      USE parameters_C, only : tore, dd
      use MOZYME_C, only : cutofs
      use overlaps_C, only : cutof1, cutof2
      USE chanel_C, only : iw
      use mozyme_gpu_gradient, only : conv_gpu_hcore_enabled, conv_gpu_solrot_run
      use mozyme_section_timers, only : mozyme_section_timer_begin, mozyme_section_timer_end
#ifdef GPU
      use gpu_conv_fock, only : conv_w_generation
#endif

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------
      implicit none
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------
      integer :: icalcn, ione
      integer :: i, j, k, kr, ia, ib, ni, i1, i2, j1, io1, jo1, im1, &
        ja, jb, nj, ii, jj, kro, q_at, link_at, mol_at
      double precision, dimension(45) :: e1b(45), e2a(45), di(9,9), wjd(2025), wkd(2025), xj(3), dibits(9,9)
      double precision, allocatable:: vqc(:)
      double precision ::  xf, yf, zf, const, fldcon, hterme, fnuc, half, enuc
      logical :: fldon, first, debug, lmolaris_qmmm, exists
      character :: tmpkey*241
      double precision, external :: reada
!  Solids on the GPU: per-pair image sums from conv_gpu_solrot_run
      double precision, allocatable :: gdi(:,:), ge1b(:,:), ge2a(:,:), genuc(:)
      integer, allocatable :: gkb(:)
      logical :: use_gpu_pairs, gpu_check
      integer :: ip, gcode, gitems, kimg
      integer(8) :: t0, t1, rate
      double precision :: gms, dmax_h, dmax_w, dmax_wk, dmax_e, dmax_en, t_hcore
      save first, debug, icalcn, ione, vqc, lmolaris_qmmm
!***********************************************************************
!
!   HCORE GENERATES THE ONE-ELECTRON MATRIX AND TWO ELECTRON INTEGRALS
!         FOR A GIVEN MOLECULE WHOSE GEOMETRY IS DEFINED IN CARTESIAN
!         COORDINATES.
!
!  ON INPUT  COORD   = COORDINATES OF THE MOLECULE.
!
!  ON OUTPUT  H      = ONE-ELECTRON MATRIX.
!             W      = TWO-ELECTRON INTEGRALS.
!             ENUCLR = NUCLEAR ENERGY
!***********************************************************************
      data icalcn/ 0/
      fnuc = 0.d0
      fldcon = 0.d0
      first = icalcn /= numcal
      icalcn = numcal
      if (first) then
        ione = 1
        cutofs = 1.d4
        cutof2 = 1.D10
        cutof1 = 225.d0 ! (Cutoff distance)**2 for overlap integrals
        if (id /= 0) then
          if (allocated(wk)) deallocate (wk)
          allocate(wk(n2elec))
        end if
        if (id /= 0) ione = 0
        debug = (index(keywrd,'HCORE') /= 0)
        lmolaris_qmmm = (index(keywrd,'QMMM') /= 0)
        if (lmolaris_qmmm) then
!
!  Read in the energy in kcal/mol that an electron on each atom would have, arising from
!  the partial charges on all atoms.
!
          line="mol.in"
          call add_path(line)
          inquire (file=trim(line), exist = exists)
          if (.not. exists) then
             call mopend("A file named '"//trim(line)//"' was expected, but was not found.")
             return
          end if
          open(85, file=line, form='formatted')
          read(85, *, iostat = i)
          if (i /= 0) then
             call mopend("A file named '"//trim(line)//"' exists, but is empty.")
             return
          end if
          read(85, *, iostat = i)q_at, link_at
          if (i /= 0) then
             call mopend("A file named '"//trim(line)//"' exists, but is corrupt.")
             return
          end if
          mol_at = q_at + link_at
          if(mol_at /= numat) then
            write(line,'(a,i5,a,i5,a)') &
              " The number of atoms in 'mol.in':",mol_at," and in the MOPAC data set:",numat," are different."
            call mopend(trim(line))
            write(iw,'(10x,a)')" Correct fault and resubmit."
            return
          end if
          if (allocated(vqc)) deallocate(vqc)
          allocate(vqc(numat))
          do i = 1, numat
            read(85,*)line,line,line,line,vqc(i)
            if (debug) write(iw,'(a, i4, a, f9.3)')"ATOM No.",i," VQC(I)",vqc(i)
          end do
          close(85)
        end if
        xf = 0.D0
        yf = 0.D0
        zf = 0.D0
        tmpkey = trim(keywrd)
        i = index(tmpkey,' FIELD(') + index(tmpkey,' FIELD=(')
        if (i /= 0) then
!
!   ERASE ALL TEXT FROM TMPKEY EXCEPT FIELD DATA
!
          tmpkey(:i) = ' '
          tmpkey(index(tmpkey,')'):) = ' '
!
!   READ IN THE EFFECTIVE FIELD IN X,Y,Z COORDINATES
!
          xf = reada(tmpkey,i)
          i = index(tmpkey,',')
          if (i /= 0) then
            tmpkey(i:i) = ' '
            yf = reada(tmpkey,i)
            i = index(tmpkey,',')
            if (i /= 0) then
              tmpkey(i:i) = ' '
              zf = reada(tmpkey,i)
            end if
          end if
          write (iw, '(/10X,''THE ELECTRIC FIELD IS'',3F10.5,'' VOLTS/ANGSTROM'',/)') xf, yf, zf
        end if
        const = a0/ev
!
        efield(1) = xf*const
        efield(2) = yf*const
        efield(3) = zf*const
      end if
      fldon = .FALSE.
      if (efield(1) /= 0.0D00 .or. efield(2) /= 0.0D00 .or. efield(3) /= 0.0D00) then
!
!   FLDCON = h/Ao
!
        fldcon = ev/a0
        fldon = .TRUE.
      end if
      enuclr = 0.d0
      h(:mpack) = 0.d0
      kr = 1
      ip = 0
      call mozyme_section_timer_begin('hcore_total', t_hcore)
      use_gpu_pairs = .false.
      gpu_check = .false.
      if (id /= 0) then
        if (allocated(wk_image)) then
          if (size(wk_image) /= (numat*(numat + 1))/2) deallocate(wk_image)
        end if
        if (.not. allocated(wk_image)) allocate(wk_image((numat*(numat + 1))/2))
        wk_image = 0
        if (conv_gpu_hcore_enabled()) then
          call conv_gpu_solrot_run(gdi, ge1b, ge2a, genuc, gkb, gcode, gms, gitems)
          use_gpu_pairs = (gcode == 0)
          if (use_gpu_pairs) then
            call get_environment_variable('MOPAC_GPU_HCORE_CHECK', line, status=i)
            gpu_check = (i == 0 .and. len_trim(line) > 0 .and. line(1:1) /= '0')
            dmax_h = 0.d0
            dmax_w = 0.d0
            dmax_wk = 0.d0
            dmax_e = 0.d0
            dmax_en = 0.d0
            call system_clock(t0, rate)
          else
            write(iw,'(1x,a,i0)') '[GPU hcore] device path unavailable, CPU used, code=', gcode
          end if
        end if
      end if
      do i = 1, numat
        ia = nfirst(i)
        ib = nlast(i)
        ni = nat(i)
!
! FIRST WE FILL THE DIAGONALS, AND OFF-DIAGONALS ON THE SAME ATOM
!
        if (.not.fldon) then
          do i1 = ia, ib
            i2 = i1*(i1 - 1)/2 + ia - 1
            if (i1 - ia + 1 > 0) then
              h(i2+1:i1-ia+1+i2) = 0.D0
              i2 = i1 - ia + 1 + i2
            end if
            h(i2) = uspd(i1)
            if(lmolaris_qmmm) then
              if (debug) write(iw,'(''OLD 1e MATRIX ELEMENT '',i5,f12.5,'' i'',i5,&
            & '' vqc(i)'',f12.5)')i2,h(i2),i,-vqc(i)/fpc_9
              h(i2) = h(i2) - vqc(i)/fpc_9
              if (debug) write(iw,'(''UPD 1e MATRIX ELEMENT '',i5,f12.5)')i2,h(i2)
            end if
            cycle
          end do
        else
          do i1 = ia, ib
            i2 = i1*(i1 - 1)/2 + ia - 1
            do j1 = ia, i1
              i2 = i2 + 1
              h(i2) = 0.D0
              io1 = i1 - ia
              jo1 = j1 - ia
              if (jo1==0 .and. io1==1) then
                hterme = -a0*dd(ni)*efield(1)*fldcon
                h(i2) = hterme
              end if
              if (jo1==0 .and. io1==2) then
                hterme = -a0*dd(ni)*efield(2)*fldcon
                h(i2) = hterme
              end if
              if (jo1/=0 .or. io1/=3) cycle
              hterme = -a0*dd(ni)*efield(3)*fldcon
              h(i2) = hterme
            end do
            h(i2) = uspd(i1)
            fnuc = -(efield(1)*coord(1,i) + efield(2)*coord(2,i) + efield(3)*coord(3,i))*fldcon
            h(i2) = h(i2) + fnuc
          end do
        end if
        if (fldon) enuclr = enuclr - fnuc*tore(nat(i))
        if (lmolaris_qmmm) enuclr = enuclr + vqc(i)/fpc_9*tore(nat(i))
!
!   FILL THE ATOM-OTHER ATOM ONE-ELECTRON MATRIX<PSI(LAMBDA)|PSI(SIGMA)>
!
        im1 = i - ione
        do j = 1, im1
          half = 1.D0
          if (i == j) half = 0.5D0
          ja = nfirst(j)
          jb = nlast(j)
          nj = nat(j)
          if (use_gpu_pairs) ip = ip + 1
          if (id == 0) then
            call h1elec (ni, nj, coord(1,i), coord(1,j), di)
          else if (use_gpu_pairs .and. .not. gpu_check) then
            di = reshape(gdi(:,ip), [9,9])
          else
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
            if (gpu_check) then
              dibits = reshape(gdi(:,ip), [9,9])
              dmax_h = max(dmax_h, maxval(abs(di(:ib-ia+1,:jb-ja+1) - dibits(:ib-ia+1,:jb-ja+1))))
            end if
          end if
          i2 = 0
          do i1 = ia, ib
            ii = i1*(i1 - 1)/2 + ja - 1
            i2 = i2 + 1
            jj = min(i1,jb)
            h(ii+1:jj-ja+1+ii) = h(ii+1:jj-ja+1+ii) + di(i2,:jj-ja+1)
          end do
!
!   CALCULATE THE TWO-ELECTRON INTEGRALS, W; THE ELECTRON NUCLEAR TERMS
!   E1B AND E2A; AND THE NUCLEAR-NUCLEAR TERM ENUC.
!
          if (id == 0) then
            call rotate (ni, nj, coord(1,i), coord(1,j), w(kr), kr, e1b, e2a, enuc)
          else if (use_gpu_pairs .and. .not. gpu_check) then
            e1b = ge1b(:,ip)
            e2a = ge2a(:,ip)
            enuc = genuc(ip)
            kr = kr + gkb(ip)
          else
            kro = kr
            call solrot (ni, nj, coord(1,i), coord(1,j), wjd, wkd, kr, e1b, e2a, enuc, kimg)
            wk_image((i*(i - 1))/2 + j) = kimg
            if (gpu_check) then
              dmax_w = max(dmax_w, maxval(abs(w(kro:kr - 1) - wjd(:kr-kro))))
              dmax_wk = max(dmax_wk, maxval(abs(wk(kro:kr - 1) - wkd(:kr-kro))))
              k = ((ib - ia + 1)*(ib - ia + 2))/2
              dmax_e = max(dmax_e, maxval(abs(e1b(:k) - ge1b(:k,ip))))
              k = ((jb - ja + 1)*(jb - ja + 2))/2
              dmax_e = max(dmax_e, maxval(abs(e2a(:k) - ge2a(:k,ip))))
              dmax_en = max(dmax_en, abs(enuc - genuc(ip)))
            end if
            jj = 0
            w(kro:kr - 1) = wjd(:kr-kro)
            wk(kro:kr - 1) = wkd(:kr-kro)
          end if
          enuclr = enuclr + enuc
!
!   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM I.
!
          i2 = 0
          do i1 = ia, ib
            ii = i1*(i1 - 1)/2 + ia - 1
            if (i1 - ia + 1 > 0) then
              h(ii+1:i1-ia+1+ii) = h(ii+1:i1-ia+1+ii) + e1b(i2+1:i1-ia+1+i2)*half
              i2 = i1 - ia + 1 + i2
            end if
          end do
!
!   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM J.
!
          i2 = 0
          do i1 = ja, jb
            ii = i1*(i1 - 1)/2 + ja - 1
            if (i1 - ja + 1 > 0) then
              h(ii+1:i1-ja+1+ii) = h(ii+1:i1-ja+1+ii) + e2a(i2+1:i1-ja+1+i2)*half
              i2 = i1 - ja + 1 + i2
            end if
          end do
        end do
        ii = ib - ia + 1
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
      call mozyme_section_timer_end('hcore_total', t_hcore)
#ifdef GPU
      conv_w_generation = conv_w_generation + 1   ! new w / wk for the device Fock build
#endif
      if (use_gpu_pairs) then
        if (gpu_check) then
          call system_clock(t1)
          write(iw,'(1x,a,i0,a,i0,a,f10.1,a,f10.1,a,5es10.2,a)') '[GPU hcore] check pairs=', ip, ' items=', gitems, &
            ' ms=', gms, ' cpu_ms=', 1.d3*dble(t1 - t0)/dble(rate), ' max|d| h w wk e1b/e2a enuc=', &
            dmax_h, dmax_w, dmax_wk, dmax_e, dmax_en, ' (CPU result kept)'
        else if (debug) then
          write(iw,'(1x,a,i0,a,i0,a,f10.1)') '[GPU hcore] pairs=', ip, ' items=', gitems, ' ms=', gms
        end if
      end if
      if (useps) then
! In the following routine the dielectric correction to the core-core-
! interaction is added to ENUCLR
        call addnuc ()
! The following routine adds the dielectric correction for the electron-
! interaction to the diagonal elements of H
        call addhcr ()
      end if
! end of COSMO change
      if (debug) then
        write (iw, '(2/10X,''ONE-ELECTRON MATRIX FROM HCORE'')')
        call vecprt (h, norbs)
        j = min(400,kr - 1)
        j = kr - 1
        if (id == 0) then
          write (iw, '(2/10X,''TWO-ELECTRON MATRIX IN HCORE''/)')
          write (iw, 200) (w(i),i=1,j)
        else
          write (iw, '(2/10X,''TWO-ELECTRON J MATRIX IN HCORE''/)')
          write (iw, 200) (w(i),i=1,j)
          write (iw, '(2/10X,''TWO-ELECTRON K MATRIX IN HCORE''/)')
          write (iw, 200) (wk(i),i=1,j)
        end if
  200   format(10f8.4)
      end if
      return
      end subroutine hcore
