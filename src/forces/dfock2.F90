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

      subroutine dfock2(f, ptot, p, w, numat, nfirst, nlast, nati)
!-----------------------------------------------
!   M o d u l e s
!-----------------------------------------------
      use common_arrays_C, only : ifact, i1fact, ptot2
      use molkst_C, only : numcal, norbs, mpack

#ifdef GPU
      use mod_vars_cuda, only: lgpu, mozyme_gpu, mozyme_f2_gpu
      use gpu_fock_interfaces, only: mopac_cuda_mozyme_dfock2
#endif
!***********************************************************************
!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------
      implicit none
!-----------------------------------------------
!   G l o b a l   P a r a m e t e r s
!-----------------------------------------------
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------
      integer , intent(in) :: numat
      integer , intent(in) :: nati
      integer , intent(in) :: nfirst(numat)
      integer , intent(in) :: nlast(numat)
      double precision  :: f(mpack)
      double precision , intent(in) :: ptot(mpack)
      double precision , intent(in) :: p(mpack)
      double precision  :: w(*)
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------
      integer :: itype
      integer , dimension(256) :: jindex
      integer :: icalcn, i, m, j, ij, ji, k, ik, l, kl, lk, kk, ia, ib, jk, kj&
        , ii, jj, ja, jb, i1, j1, ll, kr, ka, ig, jg, idx_gpu, n_ij_gpu, &
        n_kl_gpu, n_cross_gpu, gpu_status
      double precision, dimension(81) :: pk
      double precision, dimension(16) :: pja, pjb
      double precision :: sumdia, sumoff, sum, elrep
#ifdef GPU
      double precision, allocatable :: pii_gpu(:), pjj_gpu(:), pij_gpu(:)
      double precision, allocatable :: dfii_gpu(:), dfjj_gpu(:), dfij_gpu(:)
      logical :: diag_gpu
#endif

      save itype, icalcn, jindex
!-----------------------------------------------
!***********************************************************************
!
!     DFOCK2 ADDS THE 2-ELECTRON 2-CENTER REPULSION CONTRIBUTION TO
!     THE FOCK MATRIX DERIVATIVE WITHIN THE NDDO FORMALISMS.
!  INPUT
!     F    : 1-ELECTRON CONTRIBUTIONS DERIVATIVES.
!     PTOT : TOTAL DENSITY MATRIX.
!     P    : ALPHA OR BETA DENSITY MATRIX. = 0.5 * PTOT
!     W    : NON VANISHING TWO-ELECTRON INTEGRAL DERIVATIVES
!            (ORDERED AS DEFINED IN DHCORE).
!     NATI : # OF THE ATOM SUPPORTING THE VARYING CARTESIAN COORDINATE.
!  OUTPUT
!     F    : FOCK MATRIX DERIVATIVE WITH RESPECT TO THE CART. COORD.
!
!***********************************************************************
      data itype/ 1/
      data icalcn/ 0/
      if (icalcn /= numcal) then
        if (allocated(ifact))  deallocate(ifact)
        if (allocated(i1fact)) deallocate(i1fact)
        if (allocated(ptot2))  deallocate(ptot2)
        allocate(ifact(norbs), i1fact(norbs), ptot2(numat,81))
        icalcn = numcal
        itype = 1
      end if
   10 continue
      select case (itype)
      case default
        do i = 1, norbs
          ifact(i) = (i*(i - 1))/2
          i1fact(i) = ifact(i) + i
        end do
!
!   SET UP GATHER-SCATTER TYPE ARRAYS FOR USE WITH TWO-ELECTRON
!   INTEGRALS.  JINDEX ARE THE INDICES OF THE J-INTEGRALS FOR ATOM I
!   INTEGRALS.  JJNDEX ARE THE INDICES OF THE J-INTEGRALS FOR ATOM J
!               KINDEX ARE THE INDICES OF THE K-INTEGRALS
!
        m = 0
        do i = 1, 4
          do j = 1, 4
            ij = min(i,j)
            ji = i + j - ij
            do k = 1, 4
              ik = min(i,k)
              do l = 1, 4
                m = m + 1
                kl = min(k,l)
                lk = k + l - kl
                jindex(m) = (ifact(ji)+ij)*10 + ifact(lk) + kl - 10
              end do
            end do
          end do
        end do
          itype = 3
        go to 10
      case (3)
        kk = 0
        l = 0
        do i = 1, numat
          ia = nfirst(i)
          ib = nlast(i)
          m = 0
          do j = ia, ib
            do k = ia, ib
              m = m + 1
              jk = min(j,k)
              kj = k + j - jk
              jk = jk + (kj*(kj - 1))/2
              ptot2(i,m) = ptot(jk)
            end do
          end do
        end do
        ii = nati
        ia = nfirst(ii)
        ib = nlast(ii)
        do jj = 1, numat
          if (ii == jj) cycle
          ja = nfirst(jj)
          jb = nlast(jj)
          if (ib - ia < 0 .or. jb - ja < 0) cycle ! One atom is a sparkle
#ifdef GPU
          if (lgpu .and. mozyme_gpu .and. mozyme_f2_gpu) then
            ig = ib - ia + 1
            jg = jb - ja + 1
            n_ij_gpu = ig * (ig + 1) / 2
            n_kl_gpu = jg * (jg + 1) / 2
            n_cross_gpu = ig * jg
            diag_gpu = ia == ja .and. ib == jb
            gpu_status = 0
            if (n_ij_gpu <= 0 .or. n_kl_gpu <= 0 .or. n_cross_gpu <= 0) then
              gpu_status = 1
            else
              if (.not. allocated(pii_gpu) .or. size(pii_gpu) < n_ij_gpu) then
                if (allocated(pii_gpu)) deallocate(pii_gpu)
                allocate(pii_gpu(n_ij_gpu), stat=gpu_status)
              end if
              if (gpu_status == 0) then
                if (.not. allocated(pjj_gpu) .or. size(pjj_gpu) < n_kl_gpu) then
                  if (allocated(pjj_gpu)) deallocate(pjj_gpu)
                  allocate(pjj_gpu(n_kl_gpu), stat=gpu_status)
                end if
              end if
              if (gpu_status == 0) then
                if (.not. allocated(pij_gpu) .or. size(pij_gpu) < n_cross_gpu) then
                  if (allocated(pij_gpu)) deallocate(pij_gpu)
                  allocate(pij_gpu(n_cross_gpu), stat=gpu_status)
                end if
              end if
              if (gpu_status == 0) then
                if (.not. allocated(dfii_gpu) .or. size(dfii_gpu) < n_ij_gpu) then
                  if (allocated(dfii_gpu)) deallocate(dfii_gpu)
                  allocate(dfii_gpu(n_ij_gpu), stat=gpu_status)
                end if
              end if
              if (gpu_status == 0) then
                if (.not. allocated(dfjj_gpu) .or. size(dfjj_gpu) < n_kl_gpu) then
                  if (allocated(dfjj_gpu)) deallocate(dfjj_gpu)
                  allocate(dfjj_gpu(n_kl_gpu), stat=gpu_status)
                end if
              end if
              if (gpu_status == 0) then
                if (.not. allocated(dfij_gpu) .or. size(dfij_gpu) < n_cross_gpu) then
                  if (allocated(dfij_gpu)) deallocate(dfij_gpu)
                  allocate(dfij_gpu(n_cross_gpu), stat=gpu_status)
                end if
              end if
              if (gpu_status == 0) then
                idx_gpu = 0
                do i1 = ia, ib
                  do j1 = ia, i1
                    idx_gpu = idx_gpu + 1
                    pii_gpu(idx_gpu) = ptot(ifact(i1) + j1)
                  end do
                end do
                idx_gpu = 0
                do i1 = ja, jb
                  do j1 = ja, i1
                    idx_gpu = idx_gpu + 1
                    pjj_gpu(idx_gpu) = ptot(ifact(i1) + j1)
                  end do
                end do
                idx_gpu = 0
                do i1 = ia, ib
                  do j1 = ja, jb
                    idx_gpu = idx_gpu + 1
                    if (i1 >= j1) then
                      pij_gpu(idx_gpu) = p(ifact(i1) + j1)
                    else
                      pij_gpu(idx_gpu) = p(ifact(j1) + i1)
                    end if
                  end do
                end do
                dfii_gpu(1:n_ij_gpu) = 0.0d0
                dfjj_gpu(1:n_kl_gpu) = 0.0d0
                dfij_gpu(1:n_cross_gpu) = 0.0d0
                gpu_status = mopac_cuda_mozyme_dfock2(ig, jg, diag_gpu, &
                     pii_gpu, pjj_gpu, pij_gpu, dfii_gpu, dfjj_gpu, dfij_gpu, &
                     w(kk+1), w(kk+1))
                if (gpu_status == 0) then
                  idx_gpu = 0
                  do i1 = ia, ib
                    do j1 = ia, i1
                      idx_gpu = idx_gpu + 1
                      f(ifact(i1) + j1) = f(ifact(i1) + j1) + dfii_gpu(idx_gpu)
                    end do
                  end do
                  idx_gpu = 0
                  do i1 = ja, jb
                    do j1 = ja, i1
                      idx_gpu = idx_gpu + 1
                      f(ifact(i1) + j1) = f(ifact(i1) + j1) + dfjj_gpu(idx_gpu)
                    end do
                  end do
                  if (.not. diag_gpu) then
                    idx_gpu = 0
                    do i1 = ia, ib
                      do j1 = ja, jb
                        idx_gpu = idx_gpu + 1
                        sum = dfij_gpu(idx_gpu)
                        if (i1 >= j1) then
                          f(ifact(i1) + j1) = f(ifact(i1) + j1) + sum
                        else
                          f(ifact(j1) + i1) = f(ifact(j1) + i1) + sum
                        end if
                      end do
                    end do
                  end if
                  kk = kk + n_ij_gpu * n_kl_gpu
                  cycle
                end if
              end if
            end if
          end if
#endif
          if (ib - ia>=6 .or. jb-ja>=6) then
            call fockdorbs(ia, ib, ja, jb, f, p, ptot, w, kk, ifact)
          else if (ib - ia>=3 .and. jb-ja>=3) then
!
!                         HEAVY-ATOM  - HEAVY-ATOM
!
!   EXTRACT COULOMB TERMS
!
            pja = ptot2(ii,:16)
            pjb = ptot2(jj,:16)
!
!  COULOMB TERMS
!
            call jab (ia, ja, pja, pjb, w(kk+1), f)
!
!  EXCHANGE TERMS
!
!
!  EXTRACT INTERSECTION OF ATOMS II AND JJ IN THE SPIN DENSITY MATRIX
!
            if (ia > ja) then
              l = 0
              do i = ia, ib
                if (jb - ja + 1 > 0) then
                  pk(l+1:jb-ja+1+l) = p(ifact(i)+ja:jb+ifact(i))
                  l = jb - ja + 1 + l
                end if
              end do
            else
              l = 0
              do i = ia, ib
                if (jb - ja + 1 > 0) then
                  pk(l+1:jb-ja+1+l) = p(ifact(ja:jb)+i)
                  l = jb - ja + 1 + l
                end if
              end do
            end if
            i1 = ia
            j1 = ja
            call kab (ia, ja, pk, w(kk+1), f)
            ia = i1
            ja = j1
            kk = kk + 100
          else if (ib - ia >= 3) then
!
!                         LIGHT-ATOM  - HEAVY-ATOM
!
!
!   COULOMB TERMS
!
            sumdia = 0.D0
            sumoff = 0.D0
            ll = i1fact(ja)
            k = 0
            do i = 0, 3
              j1 = ifact(ia+i) + ia - 1
              do j = 0, i - 1
                k = k + 1
                j1 = j1 + 1
                f(j1) = f(j1) + ptot(ll)*w(kk+k)
                sumoff = sumoff + ptot(j1)*w(kk+k)
              end do
              j1 = j1 + 1
              k = k + 1
              f(j1) = f(j1) + ptot(ll)*w(kk+k)
              sumdia = sumdia + ptot(j1)*w(kk+k)
            end do
            f(ll) = f(ll) + sumoff*2.D0 + sumdia
!
!  EXCHANGE TERMS
!
!
!  EXTRACT INTERSECTION OF ATOMS II AND JJ IN THE SPIN DENSITY MATRIX
!
            if (ia > ja) then
              k = 0
              do i = ia, ib
                i1 = ifact(i) + ja
                sum = 0.D0
                do j = ia, ib
                  k = k + 1
                  j1 = ifact(j) + ja
                  sum = sum + p(j1)*w(kk+jindex(k))
                end do
                f(i1) = f(i1) - sum
              end do
            else
              k = 0
              do i = ia, ib
                i1 = ifact(ja) + i
                sum = 0.D0
                do j = ia, ib
                  k = k + 1
                  j1 = ifact(ja) + j
                  sum = sum + p(j1)*w(kk+jindex(k))
                end do
                f(i1) = f(i1) - sum
              end do
            end if
            kk = kk + 10
          else if (jb - ja >= 3) then
!
!                         HEAVY-ATOM - LIGHT-ATOM
!
!
!   COULOMB TERMS
!
            sumdia = 0.D0
            sumoff = 0.D0
            ll = i1fact(ia)
            k = 0
            do i = 0, 3
              j1 = ifact(ja+i) + ja - 1
              do j = 0, i - 1
                k = k + 1
                j1 = j1 + 1
                f(j1) = f(j1) + ptot(ll)*w(kk+k)
                sumoff = sumoff + ptot(j1)*w(kk+k)
              end do
              j1 = j1 + 1
              k = k + 1
              f(j1) = f(j1) + ptot(ll)*w(kk+k)
              sumdia = sumdia + ptot(j1)*w(kk+k)
            end do
            f(ll) = f(ll) + sumoff*2.D0 + sumdia
!
!  EXCHANGE TERMS
!
!
!  EXTRACT INTERSECTION OF ATOMS II AND JJ IN THE SPIN DENSITY MATRIX
!
            if (ia > ja) then
              k = ifact(ia) + ja
              j = 0
              do i = k, k + 3
                sum = 0.D0
                do l = k, k + 3
                  j = j + 1
                  sum = sum + p(l)*w(kk+jindex(j))
                end do
                f(i) = f(i) - sum
              end do
            else
              j = 0
              do k = ja, ja + 3
                i = ifact(k) + ia
                sum = 0.D0
                do ll = ja, ja + 3
                  l = ifact(ll) + ia
                  j = j + 1
                  sum = sum + p(l)*w(kk+jindex(j))
                end do
                f(i) = f(i) - sum
              end do
            end if
            kk = kk + 10
          else
!
!                         LIGHT-ATOM - LIGHT-ATOM
!
            i1 = i1fact(ia)
            j1 = i1fact(ja)
            f(i1) = f(i1) + ptot(j1)*w(kk+1)
            f(j1) = f(j1) + ptot(i1)*w(kk+1)
            if (ia > ja) then
              ij = i1 + ja - ia
              f(ij) = f(ij) - p(ij)*w(kk+1)
            else
              ij = j1 + ia - ja
              f(ij) = f(ij) - p(ij)*w(kk+1)
            end if
            kk = kk + 1
          end if
        end do
        return
      case (2)
        kr = 0
        ii = nati
        ia = nfirst(ii)
        ib = nlast(ii)
        do jj = 1, numat
          if (jj == ii) cycle
          kr = kr + 1
          elrep = w(kr)
          ja = nfirst(jj)
          jb = nlast(jj)
          if (ja < ia) then
            do i = ia, ib
              ka = ifact(i)
              kk = ka + i
              do k = ja, jb
                ll = i1fact(k)
                ik = ka + k
                f(kk) = f(kk) + ptot(ll)*elrep
                f(ll) = f(ll) + ptot(kk)*elrep
                f(ik) = f(ik) - p(ik)*elrep
              end do
            end do
          else
            do i = ia, ib
              ka = ifact(i)
              kk = ka + i
              do k = ja, jb
                ll = i1fact(k)
                ik = ll + i - k
                f(kk) = f(kk) + ptot(ll)*elrep
                f(ll) = f(ll) + ptot(kk)*elrep
                f(ik) = f(ik) - p(ik)*elrep
              end do
            end do
          end if
        end do
        return
      end select
      end subroutine dfock2
