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

subroutine fock2z (f, q, qe, wj, wk, ptot2, mode, ione)
    use molkst_C, only: numat, mpack
    use MOZYME_C, only: lijbo, iorbs, kopt
    use common_arrays_C, only: coordinates => coord, p, nat, ifact
    use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required
    use mozyme_section_timers, only: mozyme_section_timer_begin, &
      mozyme_section_timer_end, mozyme_section_timer_report
    implicit none
    integer, intent (in) :: ione, mode
    double precision :: f(mpack)
    double precision, dimension (numat), intent (out) :: q, qe
    double precision, dimension (*), intent (in) :: wj, wk
    double precision, dimension (numat, 81), intent (out) :: ptot2
    double precision :: profile_token
    external :: mozyme_gpu_strict_abort
    if (mozyme_gpu_scf_no_fallback_required()) then
      call mozyme_gpu_strict_abort('strict_fock2z_cpu_fallback', &
        'MOZYME GPU strict resident SCF does not support CPU Fock construction')
      return
    end if
    call mozyme_section_timer_begin('fock2z', profile_token)
      if (lijbo) then
      call fz2n (f, p, iorbs, nat, ifact, q, qe, wj, wk, ptot2, mode, &
           & kopt, ione, coordinates)
    else
      call fz2 (f, p, iorbs, nat, ifact, q, qe, wj, wk, ptot2, mode, &
           & kopt, ione, coordinates)
    end if
    call mozyme_section_timer_end('fock2z', profile_token)
    call mozyme_section_timer_report('fock2z')
end subroutine fock2z
!
subroutine fz2 (f, ptot, iorbs, nat, ifact, q, qe, wj, wk, ptot2, mode, kopt, &
     & ione, coord)
    use molkst_C, only: numat, norbs, mpack, numcal, l_feather
    use cosmo_C, only : useps
    use parameters_C, only: am, dd, ad, tore
    use funcon_C, only: ev, a0
    use MOZYME_C, only : direct, semidr
    use mozyme_fock1_batch, only: mozyme_fock1_batch_begin, &
      mozyme_fock1_batch_record_or_run, mozyme_fock1_batch_flush
    use mozyme_fock2_4x1_batch, only: mozyme_fock2_4x1_batch_begin, &
      mozyme_fock2_4x1_batch_record, mozyme_fock2_4x1_batch_flush
    use mozyme_resident_fock, only: mozyme_resident_fock_try, mozyme_resident_pair_supported, &
      mozyme_resident_point_supported, mozyme_point_charge_advance_kr, resident_strict_requested, &
      strict_resident_fock_abort
    use mozyme_section_timers, only: mozyme_section_timer_begin, &
      mozyme_section_timer_end, mozyme_section_timer_report
    implicit none
   !***********************************************************************
   !
   ! FOCK2 FORMS THE TWO-ELECTRON TWO-CENTER REPULSION PART OF THE FOCK
   ! MATRIX
   ! ON INPUT  PTOT = TOTAL DENSITY MATRIX.
   !           W    = TWO-ELECTRON INTEGRAL MATRIX.
   !
   !  ON OUTPUT F   = PARTIAL FOCK MATRIX
   !***********************************************************************
    integer, intent (in) :: ione, mode
    integer, dimension (numat), intent (in) :: iorbs, kopt, nat
    integer, dimension (norbs), intent (in) :: ifact
    double precision, dimension (numat), intent (inout) :: q, qe
    double precision, dimension (mpack), intent (in) :: ptot
    double precision, dimension (mpack), intent (inout) :: f
    double precision, dimension (*), intent (in) :: wj, wk
    double precision, dimension (3, numat), intent (in) :: coord
    double precision, dimension (numat, 81), intent (inout) :: ptot2
   !
   !.. Local Scalars ..
    logical :: calci, calcj, batched_4x1, resident_fock_done
    integer, save :: icalcn = 0, krinc = 0
    integer :: i, i1, iab, ii, iim1, ij, ilim, ired, j, j1, jba, ji, jj, jk, &
   & jred, k, kj, kl, kr, l, li, lii, lij, lj, ljj, lk, m, ni, nj
    double precision :: sum, sumdia, sumoff, ade, aee, da, dx, dy, dz, r, r2, &
   & ri2, ri5, rm, rp, w1, w2, w3, w4, w5, w6, w7, enuc, point, const
    double precision :: profile_token
    integer, dimension (256), save :: jindex
    double precision, dimension (16) :: pja, pjb, wk4x1
    double precision, dimension (10) :: wj4x1
    double precision, dimension (45) :: e1b, e2a
    double precision, dimension (171) :: fdummy = 0.d0
    double precision, dimension (2025) :: wjloc = 0.d0
    integer, external :: ijbo
   !
    call mozyme_section_timer_begin('fz2', profile_token)
    dx = 0.d0
    dy = 0.d0
    dz = 0.d0
    r2 = 0.d0
    w2 = 0.d0
    w3 = 0.d0
    w4 = 0.d0
    w5 = 0.d0
    w6 = 0.d0
    w7 = 0.d0
    ni = 0
    nj = 0
   !
   !
    if (icalcn /= numcal) then
      icalcn = numcal
      !
      !   SET UP GATHER-SCATTER TYPE ARRAYS FOR USE WITH TWO-ELECTRON
      !   INTEGRALS.  JINDEX ARE THE INDICES OF THE J-INTEGRALS FOR ATOM I
      !               JJNDEX ARE THE INDICES OF THE J-INTEGRALS FOR ATOM J
      !
      m = 0
      do i = 1, 4
        do j = 1, 4
          ij = Min (i, j)
          ji = i + j - ij
          do k = 1, 4
            do l = 1, 4
              m = m + 1
              kl = Min (k, l)
              lk = k + l - kl
              jindex(m) = (ifact(ji)+ij) * 10 + ifact(lk) + kl - 10
            end do
          end do
        end do
      end do
    end if
   !
   !  Put P(A,A) density into array PTOT2
   !
    call chrge_for_MOZYME (ptot, qe)
    resident_fock_done = mozyme_resident_fock_try(f, ptot, qe, iorbs, nat, ifact, &
      wj, wk, mode, kopt, ione, coord, .false.)
    if (.not. resident_fock_done .and. resident_strict_requested()) then
      call strict_resident_fock_abort('strict_resident_fock_cpu_fallback', &
        'MOZYME GPU strict resident Fock did not complete before CPU Fock')
      return
    end if
   !
   !   MODE=1:   ADD TO EXISTING FOCK MATRIX
   !   MODE=0:   CALCULATE FOCK MATRIX STARTING WITH H MATRIX
   !   MODE=-1:  REMOVE TERMS FROM FULL FOCK MATRIX.
   !
    if (mode ==-1) then
      f(1:mpack) = -f(1:mpack)
    end if
    l = 0
    do ii = 1, numat
      q(ii) = tore(nat(ii)) - qe(ii)
      i = ijbo (ii, ii)
      iab = iorbs(ii)
      m = 0
      do j = 1, iab
        do k = 1, iab
          m = m + 1
          jk = Min (j, k)
          kj = k + j - jk
          ptot2(ii, m) = ptot(i+ (kj*(kj-1))/2+jk)
        end do
      end do
    end do

   !
    kr = 0
    call mozyme_fock1_batch_begin(numat)
    call mozyme_fock2_4x1_batch_begin(max(1, numat * min(numat, 64)))
    ired = 1
    do ii = 1, numat
      calci = (kopt(ired) == ii)
      if (calci .and. ired < numat) then
        ired = ired + 1
      end if
      !
      !
        if (mode == 0) then
          calci = .true.
        end if
        iab = iorbs(ii)
        if (iab /= 0) then
          jred = 1
          iim1 = ii - ione
          do jj = 1, iim1
            calcj = (kopt(jred) == jj)
            if (calcj .and. jred < numat) then
              jred = jred + 1
            end if
            jba = iorbs(jj)
            if (ijbo(ii, jj) >= 0) then
              if (resident_fock_done .and. mozyme_resident_pair_supported(iab, jba)) then
                if (.not. direct) then
                  if ((iab == 4 .and. jba == 1) .or. (iab == 1 .and. jba == 4)) then
                    kr = kr + 10
                  else
                    kr = kr + ((iab*(iab+1)) / 2) * ((jba*(jba+1)) / 2)
                  end if
                end if
                cycle
              end if
              if (direct .and. (calci .or. calcj)) then
                call rotate(nat(ii), nat(jj), coord(1, ii), coord(1, jj), wjloc, kr, e1b, e2a, enuc)
              end if
               !
              if (iab > 5 .or. jba > 5) then
                if ( .not. (calci .or. calcj)) then
                  if ( .not. direct) then
                    kr = kr + (iab*(iab+1)) / 2 * (jba*(jba+1)) / 2
                  end if
                else
                  !
                  !   Use MNDOD specific code
                  !
                  i = ijbo (ii, ii) + 1
                  j = ijbo (jj, jj) + 1
                  ij = ijbo (ii, jj) + 1
                     !
                  if (direct) then
                    call focd2z (iab, jba, f(i), f(j), f(ij), ptot(i), &
                   & ptot(j), ptot(ij), wjloc, wjloc, i == j, krinc)
                  else
                    call focd2z (iab, jba, f(i), f(j), f(ij), ptot(i), &
                   & ptot(j), ptot(ij), wj(kr+1), wk(kr+1), i == j, kr)
                  end if
                end if
              else if (iab >= 3 .and. jba >= 3) then
                !
                !                         HEAVY-ATOM  - HEAVY-ATOM
                !
                !   EXTRACT COULOMB TERMS
                !
                if (calci .or. calcj) then
                  do i = 1, 16
                    pja(i) = ptot2(ii, i)
                    pjb(i) = ptot2(jj, i)
                  end do
                  !
                  !  COULOMB TERMS
                  !
                  if (ii == jj) then
                    if (direct) then
                      call jab_for_MOZYME (1, 1, pja, pjb, wjloc, f(ijbo(ii, ii)+1), &
                     & fdummy)
                    else
                       call jab_for_MOZYME (1, 1, pja, pjb, wj(kr+1), f(ijbo(ii, ii)+1), &
                     & fdummy)
                    end if
                  else if (direct) then
                    call jab_for_MOZYME (1, 1, pja, pjb, wjloc, f(ijbo(ii, ii)+1), &
                   & f(ijbo(jj, jj)+1))
                  else
                    call jab_for_MOZYME (1, 1, pja, pjb, wj(kr+1), f(ijbo(ii, ii)+1), &
                   & f(ijbo(jj, jj)+1))
                  end if
                  !
                  !  EXCHANGE TERMS
                  !
                  !  EXTRACT INTERSECTION OF ATOMS II AND JJ IN THE SPIN
                  !  DENSITY MATRIX
                  !
                  l = ijbo (ii, jj) + 1
                  if (ii /= jj) then

                    if (direct) then
                      call kab_for_MOZYME (0, 0, ptot(l), wjloc, f(l))
                    else
                      call kab_for_MOZYME (0, 0, ptot(l), wk(kr+1), f(l))
                    end if
                  end if
                end if

                if ( .not. direct) then
                  kr = kr + 100
                end if

              else if (iab >= 3 .and. jba == 1) then
                if (calci .or. calcj) then
                  batched_4x1 = .false.
                  if (iab == 4 .and. jba == 1) then
                    if (direct) then
                      do m = 1, 10
                        wj4x1(m) = wjloc(m)
                      end do
                      do m = 1, 16
                        wk4x1(m) = wjloc(jindex(m))
                      end do
                    else
                      do m = 1, 10
                        wj4x1(m) = wj(kr + m)
                      end do
                      do m = 1, 16
                        wk4x1(m) = wk(kr + jindex(m))
                      end do
                    end if
                    batched_4x1 = mozyme_fock2_4x1_batch_record(f, ptot, &
                      ijbo(ii, ii) + 1, ijbo(jj, jj) + 1, ijbo(ii, jj) + 1, wj4x1, wk4x1)
                  end if
                  if (.not. batched_4x1) then
                  !
                  !                         LIGHT-ATOM  - HEAVY-ATOM
                  !
                  !   COULOMB TERMS
                  !
                  sumdia = 0.d0
                  sumoff = 0.d0
                  l = ijbo (ii, jj)
                  lj = ijbo (jj, jj) + 1
                  li = ijbo (ii, ii)
                  k = 0


                  if (direct) then


                    do i = 1, 4
                      do j = 1, i - 1
                        li = li + 1
                        k = k + 1
                        f(li) = f(li) + ptot(lj) * wjloc(k)
                        sumoff = sumoff + ptot(li) * wjloc(k)
                      end do
                      li = li + 1
                      k = k + 1
                      f(li) = f(li) + ptot(lj) * wjloc(k)
                      sumdia = sumdia + ptot(li) * wjloc(k)
                    end do
                    f(lj) = f(lj) + sumoff * 2.d0 + sumdia
                    !
                    !  EXCHANGE TERMS
                    !
                    !  EXTRACT INTERSECTION OF ATOMS II AND JJ IN THE SPIN
                    !  DENSITY MATRIX
                    !
                    l = ijbo (ii, jj)
                    k = 0
                    do i = 1, 4
                      sum = 0.d0
                      do j = 1, 4
                        k = k + 1
                        sum = sum + ptot(l+j) * wjloc(jindex(k))
                      end do
                      f(l+i) = f(l+i) - sum * 0.5d0
                    end do
                  else
                        !
                    do i = 1, 4
                      do j = 1, i - 1
                        li = li + 1
                        k = k + 1
                        f(li) = f(li) + ptot(lj) * wj(kr+k)
                        sumoff = sumoff + ptot(li) * wj(kr+k)
                      end do
                      li = li + 1
                      k = k + 1
                      f(li) = f(li) + ptot(lj) * wj(kr+k)
                      sumdia = sumdia + ptot(li) * wj(kr+k)
                    end do
                    f(lj) = f(lj) + sumoff * 2.d0 + sumdia
                    !
                    !  EXCHANGE TERMS
                    !
                    !  EXTRACT INTERSECTION OF ATOMS II AND JJ IN THE SPIN
                    !  DENSITY MATRIX
                    !
                    l = ijbo (ii, jj)
                    k = 0
                    do i = 1, 4
                      sum = 0.d0
                      do j = 1, 4
                        k = k + 1
                        sum = sum + ptot(l+j) * wk(kr+jindex(k))
                      end do
                      f(l+i) = f(l+i) - sum * 0.5d0
                    end do
                  end if
                  end if
                end if


                if ( .not. direct) then
                  kr = kr + 10
                end if


              else if (jba >= 3 .and. iab == 1) then
                if (calci .or. calcj) then
                  batched_4x1 = .false.
                  if (jba == 4 .and. iab == 1) then
                    if (direct) then
                      do m = 1, 10
                        wj4x1(m) = wjloc(m)
                      end do
                      do m = 1, 16
                        wk4x1(m) = wjloc(jindex(m))
                      end do
                    else
                      do m = 1, 10
                        wj4x1(m) = wj(kr + m)
                      end do
                      do m = 1, 16
                        wk4x1(m) = wk(kr + jindex(m))
                      end do
                    end if
                    batched_4x1 = mozyme_fock2_4x1_batch_record(f, ptot, &
                      ijbo(jj, jj) + 1, ijbo(ii, ii) + 1, ijbo(ii, jj) + 1, wj4x1, wk4x1)
                  end if
                  if (.not. batched_4x1) then
                  !
                  !                         HEAVY-ATOM - LIGHT-ATOM
                  !
                  !   COULOMB TERMS
                  !
                  sumdia = 0.d0
                  sumoff = 0.d0
                  l = ijbo (ii, jj)
                  lj = ijbo (jj, jj)
                  li = ijbo (ii, ii) + 1
                  k = 0

                  if (direct) then

                    do i = 1, 4
                      do j = 1, i - 1
                        k = k + 1
                        lj = lj + 1
                        f(lj) = f(lj) + ptot(li) * wjloc(k)
                        sumoff = sumoff + ptot(lj) * wjloc(k)
                      end do
                      lj = lj + 1
                      k = k + 1
                      f(lj) = f(lj) + ptot(li) * wjloc(k)
                      sumdia = sumdia + ptot(lj) * wjloc(k)
                    end do !
                    f(li) = f(li) + sumoff * 2.d0 + sumdia
                    !
                    !  EXCHANGE TERMS
                    !
                    !  EXTRACT INTERSECTION OF ATOMS II AND JJ IN THE SPIN
                    !  DENSITY MATRIX
                    !
                    l = ijbo (ii, jj)
                    k = 0
                    do i = 1, 4
                      sum = 0.d0
                      do j = 1, 4
                        k = k + 1
                        sum = sum + ptot(l+j) * wjloc(jindex(k))
                      end do
                      f(l+i) = f(l+i) - sum * 0.5d0
                    end do
                  else


                    do i = 1, 4
                      do j = 1, i - 1
                        k = k + 1
                        lj = lj + 1
                        f(lj) = f(lj) + ptot(li) * wj(kr+k)
                        sumoff = sumoff + ptot(lj) * wj(kr+k)
                      end do
                      lj = lj + 1
                      k = k + 1
                      f(lj) = f(lj) + ptot(li) * wj(kr+k)
                      sumdia = sumdia + ptot(lj) * wj(kr+k)
                    end do
                    f(li) = f(li) + sumoff * 2.d0 + sumdia
                    !
                    !  EXCHANGE TERMS
                    !
                    !  EXTRACT INTERSECTION OF ATOMS II AND JJ IN THE SPIN
                    !  DENSITY MATRIX
                    !
                    l = ijbo (ii, jj)
                    k = 0
                    do i = 1, 4
                      sum = 0.d0
                      do j = 1, 4
                        k = k + 1
                        sum = sum + ptot(l+j) * wk(kr+jindex(k))
                      end do
                      f(l+i) = f(l+i) - sum * 0.5d0
                    end do
                  end if
                  end if
                end if

                if ( .not. direct) then
                  kr = kr + 10
                end if

              else if (iab == 1 .and. jba == 1) then
                if (calci .or. calcj) then
                  !
                  !                         LIGHT-ATOM - LIGHT-ATOM
                  !
                  lii = ijbo (ii, ii) + 1
                  lij = ijbo (ii, jj) + 1
                  ljj = ijbo (jj, jj) + 1
                     !
                  if (direct) then
                        !
                    f (lii) = f (lii) + ptot (ljj) * wjloc (1)
                    if (ii /= jj) then
                      f(ljj) = f(ljj) + ptot(lii) * wjloc(1)
                      f(lij) = f(lij) - ptot(lij) * wjloc(1) * 0.5d0
                    end if
                  else

                    f(lii) = f(lii) + ptot(ljj) * wj(kr+1)
                    if (ii /= jj) then
                      f(ljj) = f(ljj) + ptot(lii) * wj(kr+1)
                      f(lij) = f(lij) - ptot(lij) * wk(kr+1) * 0.5d0
                    end if
                  end if
                end if
                if ( .not. direct) then
                  kr = kr + 1
                end if
              end if
            else
              !
              !   Use point-charge approximation
              !
              if (resident_fock_done .and. &
                  mozyme_resident_point_supported(iab, jba, ijbo(ii, jj))) then
                call mozyme_point_charge_advance_kr(iab, jba, ijbo(ii, jj), kr)
                cycle
              end if
              i1 = ijbo (ii, ii)
              j1 = ijbo (jj, jj)

              if (iorbs(ii)*iorbs(jj) > 0) then

                if (calci .or. calcj) then

                  if (semidr) then
                    dx = coord(1, ii) - coord(1, jj)
                    dy = coord(2, ii) - coord(2, jj)
                    dz = coord(3, ii) - coord(3, jj)

                    r2 = dx * dx + dy * dy + dz * dz

                    ni = nat(ii)
                    nj = nat(jj)

                    aee = 0.5d0 / am(ni) + 0.5d0 / am(nj)
                    w1 = ev / Sqrt (r2/(a0**2)+aee**2)
                    if (l_feather) then
                      call to_point(sqrt(r2), point, const)
                      w1 = w1*const + (1.d0 - const)*point
                    end if
                  else
                    w1 = wj(kr+1)
                  end if
                  !
                  !   MONOPOLE
                  !

                  do i = 1, iorbs(ii)
                    i1 = i1 + i
                    f(i1) = f(i1) + qe(jj) * w1
                  end do
                  do j = 1, iorbs(jj)
                    j1 = j1 + j
                    f(j1) = f(j1) + qe(ii) * w1
                  end do
                  i1 = ijbo (ii, ii)
                  j1 = ijbo (jj, jj)
                end if

                if ( .not. semidr) then
                  kr = kr + 1
                end if
                  !
                if (ijbo(ii, jj) ==-2) then
                  !
                  !   DIPOLE, IF NEEDED
                  !
                  if ((calci .or. calcj) .and. semidr) then
                    r = Sqrt (r2)
                    dx = dx / r
                    dy = dy / r
                    dz = dz / r
                    if (Abs (dz) > 0.99999999d0) then
                      dz = Sign (1.d0, dz)
                    end if

                    if (iorbs(ii) > 1) then
                      da = dd(ni)
                      ade = 0.5d0 / ad(ni) + 0.5d0 / am(nj)
                      rp = Sqrt ((r/a0+da)**2+ade**2)
                      rm = Sqrt ((r/a0-da)**2+ade**2)
                      ri2 = ev * (0.5d0/rp-0.5d0/rm)
                      if (l_feather) then
                        call to_point(r, point, const)
                        ri2 = ri2*const
                      end if

                      w5 = ri2 * dx
                      w6 = ri2 * dy
                      w7 = ri2 * dz
                    end if

                    if (iorbs(jj) > 1) then

                      da = dd(nj)
                      ade = 0.5d0 / am(ni) + 0.5d0 / ad(nj)
                      rp = Sqrt ((r/a0+da)**2+ade**2)
                      rm = Sqrt ((r/a0-da)**2+ade**2)
                      ri5 = -ev * (0.5d0/rp-0.5d0/rm)
                      if (l_feather) then
                        call to_point(r, point, const)
                        ri5 = ri5*const
                      end if

                      w2 = ri5 * dx
                      w3 = ri5 * dy
                      w4 = ri5 * dz
                    end if
                  end if

                  if (iorbs(ii) > 1) then
                    if (iorbs(jj) > 1) then
                      if (calci .or. calcj) then
                        if ( .not. semidr) then
                          w2 = wj(kr+1)
                          w3 = wj(kr+2)
                          w4 = wj(kr+3)
                          w5 = wj(kr+4)
                          w6 = wj(kr+5)
                          w7 = wj(kr+6)
                        end if

                        f(j1+2) = f(j1+2) + qe(ii) * w2
                        f(j1+4) = f(j1+4) + qe(ii) * w3
                        f(j1+7) = f(j1+7) + qe(ii) * w4
                        f(i1+2) = f(i1+2) + qe(jj) * w5
                        f(i1+4) = f(i1+4) + qe(jj) * w6
                        f(i1+7) = f(i1+7) + qe(jj) * w7

                        f(j1 +1) = f(j1 +1) + ptot(i1+2) * w5 *2
                        f(j1 +3) = f(j1 +3) + ptot(i1+2) * w5 *2
                        f(j1 +6) = f(j1 +6) + ptot(i1+2) * w5 *2
                        f(j1+10) = f(j1+10) + ptot(i1+2) * w5 *2

                        f(j1 +1) = f(j1 +1) + ptot(i1+4) * w6 *2
                        f(j1 +3) = f(j1 +3) + ptot(i1+4) * w6 *2
                        f(j1 +6) = f(j1 +6) + ptot(i1+4) * w6 *2
                        f(j1+10) = f(j1+10) + ptot(i1+4) * w6 *2

                        f(j1 +1) = f(j1 +1) + ptot(i1+7) * w7 *2
                        f(j1 +3) = f(j1 +3) + ptot(i1+7) * w7 *2
                        f(j1 +6) = f(j1 +6) + ptot(i1+7) * w7 *2
                        f(j1+10) = f(j1+10) + ptot(i1+7) * w7 *2

                        f(i1 +1) = f(i1 +1) + ptot(j1+2) * w2 *2
                        f(i1 +3) = f(i1 +3) + ptot(j1+2) * w2 *2
                        f(i1 +6) = f(i1 +6) + ptot(j1+2) * w2 *2
                        f(i1+10) = f(i1+10) + ptot(j1+2) * w2 *2

                        f(i1 +1) = f(i1 +1) + ptot(j1+4) * w3 *2
                        f(i1 +3) = f(i1 +3) + ptot(j1+4) * w3 *2
                        f(i1 +6) = f(i1 +6) + ptot(j1+4) * w3 *2
                        f(i1+10) = f(i1+10) + ptot(j1+4) * w3 *2

                        f(i1 +1) = f(i1 +1) + ptot(j1+7) * w4 *2
                        f(i1 +3) = f(i1 +3) + ptot(j1+7) * w4 *2
                        f(i1 +6) = f(i1 +6) + ptot(j1+7) * w4 *2
                        f(i1+10) = f(i1+10) + ptot(j1+7) * w4 *2
                      end if

                      if ( .not. semidr) then
                        kr = kr + 6
                      end if
                    else
                      if (calci .or. calcj) then
                        if ( .not. semidr) then
                          w5 = wj(kr+1)
                          w6 = wj(kr+2)
                          w7 = wj(kr+3)
                        end if
                        f(i1+2) = f(i1+2) + qe(jj) * w5
                        f(i1+4) = f(i1+4) + qe(jj) * w6
                        f(i1+7) = f(i1+7) + qe(jj) * w7

                        f(j1 +1) = f(j1 +1) + ptot(i1+2) * w5 *2

                        f(j1 +1) = f(j1 +1) + ptot(i1+4) * w6 *2

                        f(j1 +1) = f(j1 +1) + ptot(i1+7) * w7 *2
                      end if
                      if ( .not. semidr) then
                        kr = kr + 3
                      end if
                    end if
                  else if (iorbs(jj) > 1) then
                    if (calci .or. calcj) then
                      if ( .not. semidr) then
                        w2 = wj(kr+1)
                        w3 = wj(kr+2)
                        w4 = wj(kr+3)
                      end if
                      f(j1+2) = f(j1+2) + qe(ii) * w2
                      f(j1+4) = f(j1+4) + qe(ii) * w3
                      f(j1+7) = f(j1+7) + qe(ii) * w4

                      f(i1 +1) = f(i1 +1) + ptot(j1+2) * w2 *2

                      f(i1 +1) = f(i1 +1) + ptot(j1+4) * w3 *2

                      f(i1 +1) = f(i1 +1) + ptot(j1+7) * w4 *2
                    end if
                    if ( .not. semidr) then
                      kr = kr + 3
                    end if
                  end if
                end if
              end if
            end if
          end do
          if (.not. direct .and. .not. resident_fock_done) then
            i = ijbo (ii, ii) + 1
            ilim = (iab*(iab+1)) / 2
            call mozyme_fock1_batch_record_or_run(f, ptot, wj, kr, i, kr+1, iab, ilim)
          end if
        end if
    end do
    call mozyme_fock2_4x1_batch_flush(f, ptot)
    if (.not. direct) then
      call mozyme_fock1_batch_flush(f, ptot, wj)
    end if



    if (direct .and. .not. resident_fock_done) then
      kr = 0
      call mozyme_fock1_batch_begin(numat)
      ired = 1
      do ii = 1, numat
        iab = iorbs(ii)
          if (iab /= 0) then
            i = ijbo (ii, ii) + 1
            ilim = (iab*(iab+1)) / 2
            call mozyme_fock1_batch_record_or_run(f, ptot, wj, kr, i, kr+1, iab, ilim)
          end if
      end do
      call mozyme_fock1_batch_flush(f, ptot, wj)
    end if

   !
    if (mode ==-1) then
      f(:) = -f(:)
    end if
    if (useps) then
  !    call addfck_for_MOZYME(f, ptot, iatsp, phinet, qscnet, qdenet, &
  !   & ipiden, gden, qscat)
    end if
    call mozyme_section_timer_end('fz2', profile_token)
    call mozyme_section_timer_report('fz2')
end subroutine fz2
!
subroutine fz2n (f, ptot, iorbs, nat, ifact, q, qe, wj, wk, ptot2, mode, &
     & kopt, ione, coord)
    use molkst_C, only: numat, norbs, mpack, numcal, l_feather
    use MOZYME_C, only : nijbo, &
       & direct, semidr
    use mozyme_fock1_batch, only: mozyme_fock1_batch_begin, &
      mozyme_fock1_batch_record_or_run, mozyme_fock1_batch_flush
    use mozyme_fock2_4x1_batch, only: mozyme_fock2_4x1_batch_begin, &
      mozyme_fock2_4x1_batch_record, mozyme_fock2_4x1_batch_flush
    use mozyme_resident_fock, only: mozyme_resident_fock_try, mozyme_resident_pair_supported, &
      mozyme_resident_point_supported, mozyme_point_charge_advance_kr, resident_strict_requested, &
      strict_resident_fock_abort
    use mozyme_section_timers, only: mozyme_section_timer_begin, &
      mozyme_section_timer_end, mozyme_section_timer_report
    use parameters_C, only: am, dd, ad, tore
    use cosmo_C, only: useps
    use funcon_C, only: ev, a0
   ! use permanent_arrays, only: iatsp, ipiden, phinet, &
   !      & qdenet, qscnet, gden, qscat

     use linear_cosmo, only: addfckz

    implicit none
   !***********************************************************************
   !
   ! FOCK2 FORMS THE TWO-ELECTRON TWO-CENTER REPULSION PART OF THE FOCK
   ! MATRIX
   ! ON INPUT  PTOT = TOTAL DENSITY MATRIX.
   !           W    = TWO-ELECTRON INTEGRAL MATRIX.
   !
   !  ON OUTPUT F   = PARTIAL FOCK MATRIX
   !***********************************************************************
    integer, intent (in) :: ione, mode
    integer, dimension (numat), intent (in) :: iorbs, kopt, nat
    integer, dimension (norbs), intent (in) :: ifact
    double precision, dimension (numat), intent (inout) :: q, qe
    double precision, dimension (mpack), intent (in) :: ptot
    double precision, dimension (mpack), intent (inout) :: f
    double precision, dimension (*), intent (in) :: wj, wk
    double precision, dimension (3, numat), intent (in) :: coord
    double precision, dimension (numat, 81), intent (inout) :: ptot2
   !
   !.. Local Scalars ..
    logical :: calci, calcj, batched_4x1, resident_fock_done
    integer, save :: icalcn = 0, krinc = 0
    integer :: i, i1, iab, ii, iim1, ij, ilim, ired, j, j1, jba, ji, jj, jk, &
   & jred, k, kj, kl, kr, l, li, lii, lij, lj, ljj, lk, m, ni, nj
    double precision :: sum, sumdia, sumoff, ade, aee, da, dx, dy, dz, r, r2, &
   & ri2, ri5, rm, rp, w1, w2, w3, w4, w5, w6, w7, enuc, rij, point, const
    double precision :: profile_token
    integer, dimension (256), save :: jindex
    double precision, dimension (16) :: pja, pjb, wk4x1
    double precision, dimension (10) :: wj4x1
    double precision, dimension (45) :: e1b, e2a
    double precision, dimension (171) :: fdummy = 0.d0
    double precision, dimension (2025) :: wjloc = 0.d0
   !
    call mozyme_section_timer_begin('fz2n', profile_token)
    dx = 0.d0
    dy = 0.d0
    dz = 0.d0
    r2 = 0.d0
    w2 = 0.d0
    w3 = 0.d0
    w4 = 0.d0
    w5 = 0.d0
    w6 = 0.d0
    w7 = 0.d0
    ni = 0
    nj = 0
   !
    if (icalcn /= numcal) then
      icalcn = numcal
      !
      !   SET UP GATHER-SCATTER TYPE ARRAYS FOR USE WITH TWO-ELECTRON
      !   INTEGRALS.  JINDEX ARE THE INDICES OF THE J-INTEGRALS FOR ATOM I
      !               JJNDEX ARE THE INDICES OF THE J-INTEGRALS FOR ATOM J
      !
      m = 0
      do i = 1, 4
        do j = 1, 4
          ij = Min (i, j)
          ji = i + j - ij
          do k = 1, 4
            do l = 1, 4
              m = m + 1
              kl = Min (k, l)
              lk = k + l - kl
              jindex(m) = (ifact(ji)+ij) * 10 + ifact(lk) + kl - 10
            end do
          end do
        end do
      end do
    end if
   !
   !  Put P(A,A) density into array PTOT2
   !
    call chrge_for_MOZYME (ptot, qe)
    resident_fock_done = mozyme_resident_fock_try(f, ptot, qe, iorbs, nat, ifact, &
      wj, wk, mode, kopt, ione, coord, .true.)
    if (.not. resident_fock_done .and. resident_strict_requested()) then
      call strict_resident_fock_abort('strict_resident_fock_cpu_fallback', &
        'MOZYME GPU strict resident Fock did not complete before CPU Fock')
      return
    end if
   !
   !   MODE=1:   ADD TO EXISTING FOCK MATRIX
   !   MODE=0:   CALCULATE FOCK MATRIX STARTING WITH H MATRIX
   !   MODE=-1:  REMOVE TERMS FROM FULL FOCK MATRIX.
   !
    if (mode ==-1) then
      f(1:mpack) = -f(1:mpack)
    end if
    l = 0
    do ii = 1, numat
      q(ii) = tore(nat(ii)) - qe(ii)
      i = nijbo (ii, ii)
      iab = iorbs(ii)
      m = 0
      do j = 1, iab
        do k = 1, iab
          m = m + 1
          jk = Min (j, k)
          kj = k + j - jk
          ptot2(ii, m) = ptot(i+ (kj*(kj-1))/2+jk)
        end do
      end do
    end do
    kr = 0
    call mozyme_fock1_batch_begin(numat)
    call mozyme_fock2_4x1_batch_begin(max(1, numat * min(numat, 64)))
    ired = 1
    do ii = 1, numat
      calci = (kopt(ired) == ii)
      if (calci .and. ired < numat) then
        ired = ired + 1
      end if
        if (mode == 0) then
          calci = .true.
        end if
        iab = iorbs(ii)
        if (iab /= 0) then
          jred = 1
          iim1 = ii - ione
          do jj = 1, iim1
  !          if (ii == 13 .and. jj == 7) then
  !            continue
  !          end if
            calcj = (kopt(jred) == jj)
            if (calcj .and. jred < numat) then
              jred = jred + 1
            end if
            jba = iorbs(jj)
            if (nijbo(ii, jj) >= 0) then
              if (resident_fock_done .and. mozyme_resident_pair_supported(iab, jba)) then
                if (.not. direct) then
                  if ((iab == 4 .and. jba == 1) .or. (iab == 1 .and. jba == 4)) then
                    kr = kr + 10
                  else
                    kr = kr + ((iab*(iab+1)) / 2) * ((jba*(jba+1)) / 2)
                  end if
                end if
                cycle
              end if
              if (direct .and. (calci .or. calcj)) then
                call rotate(nat(ii), nat(jj), coord(1, ii), coord(1, jj), wjloc, kr, e1b, e2a, enuc)
              end if
               !
              if (iab > 5 .or. jba > 5) then
                if ( .not. (calci .or. calcj)) then
                  if ( .not. direct) then
                    kr = kr + (iab*(iab+1)) / 2 * (jba*(jba+1)) / 2
                  end if
                else
                  !
                  !   Use "d"-orbital specific code
                  !
                  i = nijbo (ii, ii) + 1
                  j = nijbo (jj, jj) + 1
                  ij = nijbo (ii, jj) + 1
                     !
                  if (direct) then
                    call focd2z (iab, jba, f(i), f(j), f(ij), ptot(i), &
                   & ptot(j), ptot(ij), wjloc, wjloc, i == j, krinc)
                  else
                    call focd2z (iab, jba, f(i), f(j), f(ij), ptot(i), &
                   & ptot(j), ptot(ij), wj(kr+1), wk(kr+1), i == j, kr)
                  end if
                end if
              else if (iab >= 3 .and. jba >= 3) then
                !
                !                         HEAVY-ATOM  - HEAVY-ATOM
                !
                !   EXTRACT COULOMB TERMS
                !
                if (calci .or. calcj) then
                  do i = 1, 16
                    pja(i) = ptot2(ii, i)
                    pjb(i) = ptot2(jj, i)
                  end do
                  !
                  !  COULOMB TERMS
                  !
                  if (ii == jj) then
                    if (direct) then
                      call jab_for_MOZYME (1, 1, pja, pjb, wjloc, f(nijbo(ii, ii)+1), &
                     & fdummy)
                    else
                       call jab_for_MOZYME (1, 1, pja, pjb, wj(kr+1), f(nijbo(ii, ii)+1), &
                     & fdummy)
                    end if
                  else if (direct) then
                    call jab_for_MOZYME (1, 1, pja, pjb, wjloc, f(nijbo(ii, ii)+1), &
                   & f(nijbo(jj, jj)+1))
                  else
                     call jab_for_MOZYME (1, 1, pja, pjb, wj(kr+1), f(nijbo(ii, ii)+1), &
                   & f(nijbo(jj, jj)+1))
                  end if
                  !
                  !  EXCHANGE TERMS
                  !
                  !  EXTRACT INTERSECTION OF ATOMS II AND JJ IN THE SPIN
                  !  DENSITY MATRIX
                  !
                  l = nijbo (ii, jj) + 1
                  if (ii /= jj) then

                    if (direct) then
                      call kab_for_MOZYME (0, 0, ptot(l), wjloc, f(l))
                    else
                      call kab_for_MOZYME (0, 0, ptot(l), wk(kr+1), f(l))
                    end if
                  end if
                end if

                if ( .not. direct) then
                  kr = kr + 100
                end if

              else if (iab >= 3 .and. jba == 1) then
                if (calci .or. calcj) then
                  batched_4x1 = .false.
                  if (iab == 4 .and. jba == 1) then
                    if (direct) then
                      do m = 1, 10
                        wj4x1(m) = wjloc(m)
                      end do
                      do m = 1, 16
                        wk4x1(m) = wjloc(jindex(m))
                      end do
                    else
                      do m = 1, 10
                        wj4x1(m) = wj(kr + m)
                      end do
                      do m = 1, 16
                        wk4x1(m) = wk(kr + jindex(m))
                      end do
                    end if
                    batched_4x1 = mozyme_fock2_4x1_batch_record(f, ptot, &
                      nijbo(ii, ii) + 1, nijbo(jj, jj) + 1, nijbo(ii, jj) + 1, wj4x1, wk4x1)
                  end if
                  if (.not. batched_4x1) then
                  !
                  !                         LIGHT-ATOM  - HEAVY-ATOM
                  !
                  !   COULOMB TERMS
                  !
                  sumdia = 0.d0
                  sumoff = 0.d0
                  l = nijbo (ii, jj)
                  lj = nijbo (jj, jj) + 1
                  li = nijbo (ii, ii)
                  k = 0


                  if (direct) then


                    do i = 1, 4
                      do j = 1, i - 1
                        li = li + 1
                        k = k + 1
                        f(li) = f(li) + ptot(lj) * wjloc(k)
                        sumoff = sumoff + ptot(li) * wjloc(k)
                      end do
                      li = li + 1
                      k = k + 1
                      f(li) = f(li) + ptot(lj) * wjloc(k)
                      sumdia = sumdia + ptot(li) * wjloc(k)
                    end do
                    f(lj) = f(lj) + sumoff * 2.d0 + sumdia
                    !
                    !  EXCHANGE TERMS
                    !
                    !  EXTRACT INTERSECTION OF ATOMS II AND JJ IN THE SPIN
                    !  DENSITY MATRIX
                    !
                    l = nijbo (ii, jj)
                    k = 0
                    do i = 1, 4
                      sum = 0.d0
                      do j = 1, 4
                        k = k + 1
                        sum = sum + ptot(l+j) * wjloc(jindex(k))
                      end do
                      f(l+i) = f(l+i) - sum * 0.5d0
                    end do
                  else
                        !
                    do i = 1, 4
                      do j = 1, i - 1
                        li = li + 1
                        k = k + 1
                        f(li) = f(li) + ptot(lj) * wj(kr+k)
                        sumoff = sumoff + ptot(li) * wj(kr+k)
                      end do
                      li = li + 1
                      k = k + 1
                      f(li) = f(li) + ptot(lj) * wj(kr+k)
                      sumdia = sumdia + ptot(li) * wj(kr+k)
                    end do
                    f(lj) = f(lj) + sumoff * 2.d0 + sumdia
                    !
                    !  EXCHANGE TERMS
                    !
                    !  EXTRACT INTERSECTION OF ATOMS II AND JJ IN THE SPIN
                    !  DENSITY MATRIX
                    !
                    l = nijbo (ii, jj)
                    k = 0
                    do i = 1, 4
                      sum = 0.d0
                      do j = 1, 4
                        k = k + 1
                        sum = sum + ptot(l+j) * wk(kr+jindex(k))
                      end do
                      f(l+i) = f(l+i) - sum * 0.5d0
                    end do
                  end if
                  end if
                end if


                if ( .not. direct) then
                  kr = kr + 10
                end if


              else if (jba >= 3 .and. iab == 1) then
                if (calci .or. calcj) then
                  batched_4x1 = .false.
                  if (jba == 4 .and. iab == 1) then
                    if (direct) then
                      do m = 1, 10
                        wj4x1(m) = wjloc(m)
                      end do
                      do m = 1, 16
                        wk4x1(m) = wjloc(jindex(m))
                      end do
                    else
                      do m = 1, 10
                        wj4x1(m) = wj(kr + m)
                      end do
                      do m = 1, 16
                        wk4x1(m) = wk(kr + jindex(m))
                      end do
                    end if
                    batched_4x1 = mozyme_fock2_4x1_batch_record(f, ptot, &
                      nijbo(jj, jj) + 1, nijbo(ii, ii) + 1, nijbo(ii, jj) + 1, wj4x1, wk4x1)
                  end if
                  if (.not. batched_4x1) then
                  !
                  !                         HEAVY-ATOM - LIGHT-ATOM
                  !
                  !   COULOMB TERMS
                  !
                  sumdia = 0.d0
                  sumoff = 0.d0
                  l = nijbo (ii, jj)
                  lj = nijbo (jj, jj)
                  li = nijbo (ii, ii) + 1
                  k = 0

                  if (direct) then

                    do i = 1, 4
                      do j = 1, i - 1
                        k = k + 1
                        lj = lj + 1
                        f(lj) = f(lj) + ptot(li) * wjloc(k)
                        sumoff = sumoff + ptot(lj) * wjloc(k)
                      end do
                      lj = lj + 1
                      k = k + 1
                      f(lj) = f(lj) + ptot(li) * wjloc(k)
                      sumdia = sumdia + ptot(lj) * wjloc(k)
                    end do !
                    f(li) = f(li) + sumoff * 2.d0 + sumdia
                    !
                    !  EXCHANGE TERMS
                    !
                    !  EXTRACT INTERSECTION OF ATOMS II AND JJ IN THE SPIN
                    !  DENSITY MATRIX
                    !
                    l = nijbo (ii, jj)
                    k = 0
                    do i = 1, 4
                      sum = 0.d0
                      do j = 1, 4
                        k = k + 1
                        sum = sum + ptot(l+j) * wjloc(jindex(k))
                      end do
                      f(l+i) = f(l+i) - sum * 0.5d0
                    end do
                  else
                    do i = 1, 4
                      do j = 1, i - 1
                        k = k + 1
                        lj = lj + 1
                        f(lj) = f(lj) + ptot(li) * wj(kr+k)
                        sumoff = sumoff + ptot(lj) * wj(kr+k)
                      end do
                      lj = lj + 1
                      k = k + 1
                      f(lj) = f(lj) + ptot(li) * wj(kr+k)
                      sumdia = sumdia + ptot(lj) * wj(kr+k)
                    end do
                    f(li) = f(li) + sumoff * 2.d0 + sumdia
                    !
                    !  EXCHANGE TERMS
                    !
                    !  EXTRACT INTERSECTION OF ATOMS II AND JJ IN THE SPIN
                    !  DENSITY MATRIX
                    !
                    l = nijbo (ii, jj)
                    k = 0
                    do i = 1, 4
                      sum = 0.d0
                      do j = 1, 4
                        k = k + 1
                        sum = sum + ptot(l+j) * wk(kr+jindex(k))
                      end do
                      f(l+i) = f(l+i) - sum * 0.5d0
                    end do
                  end if
                  end if
                end if

                if ( .not. direct) then
                  kr = kr + 10
                end if

              else if (iab == 1 .and. jba == 1) then
                if (calci .or. calcj) then
                  !
                  !                         LIGHT-ATOM - LIGHT-ATOM
                  !
                  lii = nijbo (ii, ii) + 1
                  lij = nijbo (ii, jj) + 1
                  ljj = nijbo (jj, jj) + 1
                     !
                  if (direct) then
                        !
                    f (lii) = f (lii) + ptot (ljj) * wjloc (1)!
                    if (ii /= jj) then
                      f(ljj) = f(ljj) + ptot(lii) * wjloc(1)
                      f(lij) = f(lij) - ptot(lij) * wjloc(1) * 0.5d0
                    end if
                  else

                    f(lii) = f(lii) + ptot(ljj) * wj(kr+1)
                    if (ii /= jj) then
                      f(ljj) = f(ljj) + ptot(lii) * wj(kr+1)
                      f(lij) = f(lij) - ptot(lij) * wk(kr+1) * 0.5d0
                    end if
                  end if
                end if
                if ( .not. direct) then
                  kr = kr + 1
                end if
              end if
            else
              !
              !   Use point-charge approximation
              !
              if (resident_fock_done .and. &
                  mozyme_resident_point_supported(iab, jba, nijbo(ii, jj))) then
                call mozyme_point_charge_advance_kr(iab, jba, nijbo(ii, jj), kr)
                cycle
              end if
              i1 = nijbo (ii, ii)
              j1 = nijbo (jj, jj)

              if (iorbs(ii)*iorbs(jj) > 0) then

                if (calci .or. calcj) then

                  if (semidr) then
                    dx = coord(1, ii) - coord(1, jj)
                    dy = coord(2, ii) - coord(2, jj)
                    dz = coord(3, ii) - coord(3, jj)

                    r2 = dx * dx + dy * dy + dz * dz
                    ni = nat(ii)
                    nj = nat(jj)
                    aee = 0.5d0 / am(ni) + 0.5d0 / am(nj)
                    if (l_feather) then
                      rij = sqrt(r2)
                      call to_point(rij, point, const)
                      w1 = ev / Sqrt (r2/(a0**2)+aee**2)
                      w1 = w1*const + (1.d0 - const)*point
                    else
                      w1 = ev / Sqrt (r2/(a0**2)+aee**2)
                    end if
                  else
                    w1 = wj(kr+1)
                  end if
                  !
                  !   MONOPOLE
                  !
                  do i = 1, iorbs(ii)
                    i1 = i1 + i
                    f(i1) = f(i1) + qe(jj) * w1
                  end do
                  do j = 1, iorbs(jj)
                    j1 = j1 + j
                    f(j1) = f(j1) + qe(ii) * w1
                  end do
                  i1 = nijbo (ii, ii)
                  j1 = nijbo (jj, jj)
                end if

                if ( .not. semidr) then
                  kr = kr + 1
                end if
                  !
                if (nijbo(ii, jj) ==-2) then
                  !
                  !   DIPOLE, IF NEEDED
                  !
                  if ((calci .or. calcj) .and. semidr) then
                    r = Sqrt (r2)
                    dx = dx / r
                    dy = dy / r
                    dz = dz / r
                    if (Abs (dz) > 0.99999999d0) then
                      dz = Sign (1.d0, dz)
                    end if

                    if (iorbs(ii) > 1) then

                      da = dd(ni)
                      ade = 0.5d0 / ad(ni) + 0.5d0 / am(nj)
                      rp = Sqrt ((r/a0+da)**2+ade**2)
                      rm = Sqrt ((r/a0-da)**2+ade**2)
                      ri2 = ev * (0.5d0/rp-0.5d0/rm)
                      if (l_feather) then
                        call to_point(r, point, const)
                        ri2 = ri2*const
                      end if

                      w5 = ri2 * dx
                      w6 = ri2 * dy
                      w7 = ri2 * dz
                    end if

                    if (iorbs(jj) > 1) then

                      da = dd(nj)
                      ade = 0.5d0 / am(ni) + 0.5d0 / ad(nj)
                      rp = Sqrt ((r/a0+da)**2+ade**2)
                      rm = Sqrt ((r/a0-da)**2+ade**2)
                      ri5 = -ev * (0.5d0/rp-0.5d0/rm)
                      if (l_feather) then
                        call to_point(r, point, const)
                        ri5 = ri5*const
                      end if
                      w2 = ri5 * dx
                      w3 = ri5 * dy
                      w4 = ri5 * dz
                    end if
                  end if

                  if (iorbs(ii) > 1) then
                    if (iorbs(jj) > 1) then
                      if (calci .or. calcj) then
                        if ( .not. semidr) then
                          w2 = wj(kr+1)
                          w3 = wj(kr+2)
                          w4 = wj(kr+3)
                          w5 = wj(kr+4)
                          w6 = wj(kr+5)
                          w7 = wj(kr+6)
                        end if
                        f(j1+2) = f(j1+2) + qe(ii) * w2
                        f(j1+4) = f(j1+4) + qe(ii) * w3
                        f(j1+7) = f(j1+7) + qe(ii) * w4
                        f(i1+2) = f(i1+2) + qe(jj) * w5
                        f(i1+4) = f(i1+4) + qe(jj) * w6
                        f(i1+7) = f(i1+7) + qe(jj) * w7

                        f(j1 +1) = f(j1 +1) + ptot(i1+2) * w5 *2
                        f(j1 +3) = f(j1 +3) + ptot(i1+2) * w5 *2
                        f(j1 +6) = f(j1 +6) + ptot(i1+2) * w5 *2
                        f(j1+10) = f(j1+10) + ptot(i1+2) * w5 *2

                        f(j1 +1) = f(j1 +1) + ptot(i1+4) * w6 *2
                        f(j1 +3) = f(j1 +3) + ptot(i1+4) * w6 *2
                        f(j1 +6) = f(j1 +6) + ptot(i1+4) * w6 *2
                        f(j1+10) = f(j1+10) + ptot(i1+4) * w6 *2

                        f(j1 +1) = f(j1 +1) + ptot(i1+7) * w7 *2
                        f(j1 +3) = f(j1 +3) + ptot(i1+7) * w7 *2
                        f(j1 +6) = f(j1 +6) + ptot(i1+7) * w7 *2
                        f(j1+10) = f(j1+10) + ptot(i1+7) * w7 *2

                        f(i1 +1) = f(i1 +1) + ptot(j1+2) * w2 *2
                        f(i1 +3) = f(i1 +3) + ptot(j1+2) * w2 *2
                        f(i1 +6) = f(i1 +6) + ptot(j1+2) * w2 *2
                        f(i1+10) = f(i1+10) + ptot(j1+2) * w2 *2

                        f(i1 +1) = f(i1 +1) + ptot(j1+4) * w3 *2
                        f(i1 +3) = f(i1 +3) + ptot(j1+4) * w3 *2
                        f(i1 +6) = f(i1 +6) + ptot(j1+4) * w3 *2
                        f(i1+10) = f(i1+10) + ptot(j1+4) * w3 *2

                        f(i1 +1) = f(i1 +1) + ptot(j1+7) * w4 *2
                        f(i1 +3) = f(i1 +3) + ptot(j1+7) * w4 *2
                        f(i1 +6) = f(i1 +6) + ptot(j1+7) * w4 *2
                        f(i1+10) = f(i1+10) + ptot(j1+7) * w4 *2
                      end if

                      if ( .not. semidr) then
                        kr = kr + 6
                      end if
                    else
                      if (calci .or. calcj) then
                        if ( .not. semidr) then
                          w5 = wj(kr+1)
                          w6 = wj(kr+2)
                          w7 = wj(kr+3)
                        end if
                        f(i1+2) = f(i1+2) + qe(jj) * w5
                        f(i1+4) = f(i1+4) + qe(jj) * w6
                        f(i1+7) = f(i1+7) + qe(jj) * w7

                        f(j1 +1) = f(j1 +1) + ptot(i1+2) * w5 *2

                        f(j1 +1) = f(j1 +1) + ptot(i1+4) * w6 *2

                        f(j1 +1) = f(j1 +1) + ptot(i1+7) * w7 *2
                      end if
                      if ( .not. semidr) then
                        kr = kr + 3
                      end if
                    end if
                  else if (iorbs(jj) > 1) then
                    if (calci .or. calcj) then
                      if ( .not. semidr) then
                        w2 = wj(kr+1)
                        w3 = wj(kr+2)
                        w4 = wj(kr+3)
                      end if
                      f(j1+2) = f(j1+2) + qe(ii) * w2
                      f(j1+4) = f(j1+4) + qe(ii) * w3
                      f(j1+7) = f(j1+7) + qe(ii) * w4

                      f(i1 +1) = f(i1 +1) + ptot(j1+2) * w2 *2

                      f(i1 +1) = f(i1 +1) + ptot(j1+4) * w3 *2

                      f(i1 +1) = f(i1 +1) + ptot(j1+7) * w4 *2
                    end if
                    if ( .not. semidr) then
                      kr = kr + 3
                    end if
                  end if
                end if
              end if
            end if
          end do
          if (.not. direct .and. .not. resident_fock_done) then
            i = nijbo (ii, ii) + 1
            ilim = (iab*(iab+1)) / 2
            call mozyme_fock1_batch_record_or_run(f, ptot, wj, kr, i, kr+1, iab, ilim)
          end if
        end if
    end do
    call mozyme_fock2_4x1_batch_flush(f, ptot)
    if (.not. direct) then
      call mozyme_fock1_batch_flush(f, ptot, wj)
    end if
    if (direct .and. .not. resident_fock_done) then
      kr = 0
      call mozyme_fock1_batch_begin(numat)
      ired = 1
      do ii = 1, numat
          iab = iorbs(ii)
          if (iab /= 0) then
            i = nijbo (ii, ii) + 1
            ilim = (iab*(iab+1)) / 2
            call mozyme_fock1_batch_record_or_run(f, ptot, wj, kr, i, kr+1, iab, ilim)
          end if
      end do
      call mozyme_fock1_batch_flush(f, ptot, wj)
    end if
   !
    if (mode == -1) f = -f
   ! The following routine adds the dielectric correction to F.
    if (useps .and. mode == 0) call addfckz()
    call mozyme_section_timer_end('fz2n', profile_token)
    call mozyme_section_timer_report('fz2n')
end subroutine fz2n
subroutine focd2z (iab, jba, fii, fjj, fij, pii, pjj, pij, wj, wk, &
     & diagonal, kr)
#ifdef GPU
   use mod_vars_cuda, only: lgpu, mozyme_gpu, mozyme_fock_gpu, mozyme_f2_gpu
   use iso_c_binding, only: c_bool, c_int, c_double
   use chanel_C, only: iw
   use mozyme_resident_fock, only: resident_strict_requested, &
     strict_resident_fock_abort
#endif
   !
   !.. Implicit Declarations ..
    implicit none
#ifdef GPU
    interface
      function mopac_cuda_mozyme_fock2(iab, jba, diagonal, pii, pjj, pij, fii, fjj, fij, wj, wk) &
          bind(C,name='mopac_cuda_mozyme_fock2') result(code)
        use iso_c_binding, only: c_bool, c_int, c_double
        integer(c_int), value :: iab, jba
        logical(c_bool), value :: diagonal
        real(c_double) :: pii(*), pjj(*), pij(*), fii(*), fjj(*), fij(*), wj(*), wk(*)
        integer(c_int) :: code
      end function mopac_cuda_mozyme_fock2
    end interface
#endif
   !
   !.. Formal Arguments ..
    logical, intent (in) :: diagonal
    integer, intent (in) :: iab, jba
    integer, intent (inout) :: kr
    double precision, dimension ((iab*(iab+1))/2*(jba*(jba+1))/2), &
         & intent (in) :: wj, wk
    double precision, dimension ((iab*(iab+1))/2), intent (in) :: pii
    double precision, dimension ((iab*(iab+1))/2), intent (inout) :: fii
    double precision, dimension ((jba*(jba+1))/2), intent (in) :: pjj
    double precision, dimension ((jba*(jba+1))/2), intent (inout) :: fjj
    double precision, dimension (iab*jba), intent (in) :: pij
    double precision, dimension (iab*jba), intent (inout) :: fij
!.
    integer :: i, ij, ik, il, j, jk, jl, k, ka, kc, kl, l, loop
#ifdef GPU
    integer :: gpu_code
    integer :: env_len, env_status
    logical(c_bool) :: diag_flag
    logical :: trace_gpu
    logical, save :: printed_gpu_success = .false.
    logical, save :: printed_gpu_fallback = .false.
    character(len=16) :: gpu_profile_env, gpu_verbose_env
#endif
    double precision :: a, aa, bb
   !***********************************************************************
   !
   ! FOCKD2 FORMS THE TWO-ELECTRON TWO-CENTER REPULSION PART OF THE FOCK
   ! MATRIX
   ! ON INPUT  PTOT = TOTAL DENSITY MATRIX.
   !           P    = ALPHA OR BETA DENSITY MATRIX.
   !           W    = TWO-ELECTRON INTEGRAL MATRIX.
   !
   !  ON OUTPUT F   = PARTIAL FOCK MATRIX
   !***********************************************************************
    loop = 0
    if (iab <= 0 .or. jba <= 0) return
#ifdef GPU
    if (lgpu .and. mozyme_gpu .and. mozyme_fock_gpu .and. mozyme_f2_gpu) then
      diag_flag = diagonal
      if (.not. printed_gpu_success .and. .not. printed_gpu_fallback) then
        trace_gpu = .false.
        gpu_profile_env = ' '
        gpu_verbose_env = ' '
        call get_environment_variable('MOPAC_GPU_PROFILE', gpu_profile_env, length=env_len, status=env_status)
        if (env_status == 0 .and. env_len > 0 .and. trim(gpu_profile_env) /= '0') trace_gpu = .true.
        call get_environment_variable('MOPAC_GPU_VERBOSE', gpu_verbose_env, length=env_len, status=env_status)
        if (env_status == 0 .and. env_len > 0 .and. trim(gpu_verbose_env) /= '0') trace_gpu = .true.
        if (trace_gpu) then
          write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0)') &
            '[MOZYME GPU fock2]', 'attempt iab=', iab, 'jba=', jba
          call flush(iw)
        end if
      end if
      gpu_code = mopac_cuda_mozyme_fock2(iab, jba, diag_flag, pii, pjj, pij, fii, fjj, fij, wj, wk)
      if (gpu_code == 0) then
        if (.not. printed_gpu_success) then
          trace_gpu = .false.
          gpu_profile_env = ' '
          gpu_verbose_env = ' '
          call get_environment_variable('MOPAC_GPU_PROFILE', gpu_profile_env, length=env_len, status=env_status)
          if (env_status == 0 .and. env_len > 0 .and. trim(gpu_profile_env) /= '0') trace_gpu = .true.
          call get_environment_variable('MOPAC_GPU_VERBOSE', gpu_verbose_env, length=env_len, status=env_status)
          if (env_status == 0 .and. env_len > 0 .and. trim(gpu_verbose_env) /= '0') trace_gpu = .true.
          if (trace_gpu) then
            write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0)') &
              '[MOZYME GPU fock2]', 'success iab=', iab, 'jba=', jba
          end if
          printed_gpu_success = .true.
        end if
        kr = kr + (iab*(iab+1))/2 * (jba*(jba+1))/2
        return
      else
        if (resident_strict_requested()) then
          call strict_resident_fock_abort('strict_legacy_fock2_failed', &
            'MOZYME GPU strict legacy Fock2 failed')
          return
        end if
        if (.not. printed_gpu_fallback) then
          trace_gpu = .false.
          gpu_profile_env = ' '
          gpu_verbose_env = ' '
          call get_environment_variable('MOPAC_GPU_PROFILE', gpu_profile_env, length=env_len, status=env_status)
          if (env_status == 0 .and. env_len > 0 .and. trim(gpu_profile_env) /= '0') trace_gpu = .true.
          call get_environment_variable('MOPAC_GPU_VERBOSE', gpu_verbose_env, length=env_len, status=env_status)
          if (env_status == 0 .and. env_len > 0 .and. trim(gpu_verbose_env) /= '0') trace_gpu = .true.
          if (trace_gpu) then
            write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
              '[MOZYME GPU fock2]', 'fallback code=', gpu_code, 'iab=', iab, 'jba=', jba
          end if
          printed_gpu_fallback = .true.
        end if
      end if
    end if
#endif
    do i = 1, iab
      ka = (i*(i-1)) / 2
      aa = 2.0d00
      do j = 1, i
        if (i == j) then
          aa = 1.0d00
        end if
        ij = ka + j
        do k = 1, jba
          kc = (k*(k-1)) / 2
          bb = 2.0d00
          do l = 1, k
            if (k == l) then
              bb = 1.0d00
            end if
            kl = kc + l
            loop = loop + 1
            a = wj(loop)
            fii(ij) = fii(ij) + bb * a * pjj(kl)
            if ( .not. diagonal) then
              fjj(kl) = fjj(kl) + aa * a * pii(ij)
              a = wk(loop) * aa * bb * 0.125d0
              ik = (i-1) * jba + k
              il = (i-1) * jba + l
              jk = (j-1) * jba + k
              jl = (j-1) * jba + l
              fij(ik) = fij(ik) - a * pij(jl)
              fij(il) = fij(il) - a * pij(jk)
              fij(jk) = fij(jk) - a * pij(il)
              fij(jl) = fij(jl) - a * pij(ik)
            end if
          end do
        end do
      end do
    end do
    kr = kr + loop
end subroutine focd2z
