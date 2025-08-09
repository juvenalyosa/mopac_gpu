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

subroutine density_for_MOZYME (p, mode, nclose_loc, partp)
   !***********************************************************************
   !
   !   DENSIT COMPUTES THE DENSITY MATRIX GIVEN THE EIGENVECTOR MATRIX, AND
   !          INFORMATION ABOUT THE M.O. OCCUPANCY.
   !
   !  INPUT:  COCC  = COMPRESSED OCCUPIED EIGENVECTOR MATRIX
   !
   !   ON EXIT: P   = DENSITY MATRIX
   !
   !***********************************************************************
    use molkst_C, only: numat, mpack, keywrd
    use MOZYME_C, only : lijbo, nijbo, ncf, ncocc, &
      nncf, iorbs, cocc, icocc
    use chanel_C, only: iw
#ifdef GPU
    use mod_vars_cuda, only: lgpu, mozyme_gpu, mozyme_gpu_min_block, ngpus, mozyme_force_2gpu
    use mopac_cublas_interfaces
#endif
    implicit none
    integer, intent (in) :: mode, nclose_loc
    double precision, dimension (mpack), intent (in) :: partp
    double precision, dimension (mpack), intent (inout) :: p
    logical :: first = .true.
    logical, save :: prnt
    integer :: i, j, j1, ja, jj, k, k1, k2, ka, kk, l, loop, nj
    integer :: lbase, nb, nk
    double precision, allocatable :: xmat(:,:)
    double precision, allocatable :: xblk(:,:)
    double precision, allocatable :: avec(:,:), bvec(:,:)
    double precision :: spinfa, sum
    integer, external :: ijbo
    if (first) then
      first = .false.
      prnt = Index (keywrd, " DIAG ") /= 0
    end if
    if (mode == 0) then
      !
      !  INITIALIZE P:  FULL DENSITY CALCULATION
      !
      p(:) = 0.d0
    else if (mode ==-1) then
      !
      !   FULL DENSITY MATRIX IS IN PARTP, REMOVE DENSITY DUE TO
      !   OLD LMO's USED IN SCF
      !
      p(:) = -0.5d0 * partp(:)
    else
      !
      !   PARTIAL DENSITY MATRIX IS IN PARTP, BUILD THE REST OF P
      !
      p(:) = 0.5d0 * partp(:)
    end if

    do i = 1, nclose_loc
      loop = ncocc(i)
      ja = 0
      if (lijbo) then
        do jj = nncf(i) + 1, nncf(i) + ncf(i)
          j = icocc(jj)
          nj = iorbs(j)
          ka = loop
          do kk = nncf(i) + 1, nncf(i) + ncf(i)
            k = icocc(kk)
            if (j == k) then
#ifdef GPU
              if (mozyme_gpu .and. lgpu .and. (nj >= mozyme_gpu_min_block)) then
                ! Use SYRK on GPU to form outer product v*v^T for the diagonal block
                nb = nj
                lbase = nijbo(j, k)
                allocate(xmat(nb, nb))
                xmat = 0.0d0
                allocate(avec(nb,1))
                avec(:,1) = cocc(ja+1+loop:ja+nb+loop)
                if (ngpus > 1 .or. mozyme_force_2gpu) then
                  call syrk_cublas_2gpu('L','N', nb, 1, 1.0d0, avec, nb, 0.0d0, xmat, nb)
                else
                  call syrk_cublas('L','N', nb, 1, 1.0d0, avec, nb, 0.0d0, xmat, nb)
                end if
                deallocate(avec)
                l = lbase
                do j1 = 1, nb
                  do k1 = 1, j1
                    l = l + 1
                    p(l) = p(l) + xmat(j1, k1)
                  end do
                end do
                deallocate(xmat)
              else if (mozyme_gpu .and. (nj >= mozyme_gpu_min_block)) then
                nb = nj
                lbase = nijbo(j, k)
                allocate(xmat(nb, nb))
                xmat = 0.0d0
                call dsyrk('L','N', nb, 1, 1.0d0, cocc(ja+1+loop), nb, 0.0d0, xmat, nb)
                l = lbase
                do j1 = 1, nb
                  do k1 = 1, j1
                    l = l + 1
                    p(l) = p(l) + xmat(j1, k1)
                  end do
                end do
                deallocate(xmat)
              else
#endif
                l = nijbo (j, k)
                do j1 = 1, nj
                  sum = cocc(ja+j1+loop)
                  do k1 = 1, j1
                    k2 = ka + k1
                    l = l + 1
                    p(l) = p(l) + cocc(k2) * sum
                  end do
                end do
#ifdef GPU
              end if
#endif
            else if (j > k .and. nijbo (j, k) >= 0) then
              l = nijbo (j, k)
#ifdef GPU
              if (mozyme_gpu .and. lgpu .and. (nj >= mozyme_gpu_min_block) .and. (iorbs(k) >= mozyme_gpu_min_block)) then
                ! Off-diagonal block: form outer product a(nj) * b(nk)^T with GEMM (k=1)
                nk = iorbs(k)
                allocate(xblk(nj, nk))
                xblk = 0.0d0
                allocate(avec(nj,1), bvec(nk,1))
                avec(:,1) = cocc(ja+1+loop:ja+nj+loop)
                bvec(:,1) = cocc(ka+1:ka+nk)
                if (ngpus > 1 .or. mozyme_force_2gpu) then
                  call gemm_cublas_2gpu('N','T', nj, nk, 1, 1.0d0, avec, nj, bvec, nk, 0.0d0, xblk, nj)
                else
                  call gemm_cublas('N','T', nj, nk, 1, 1.0d0, avec, nj, bvec, nk, 0.0d0, xblk, nj)
                end if
                deallocate(avec, bvec)
                do j1 = 1, nj
                  do k1 = 1, nk
                    l = l + 1
                    p(l) = p(l) + xblk(j1, k1)
                  end do
                end do
                deallocate(xblk)
              else if (mozyme_gpu .and. (nj >= mozyme_gpu_min_block) .and. (iorbs(k) >= mozyme_gpu_min_block)) then
                nk = iorbs(k)
                allocate(xblk(nj, nk))
                xblk = 0.0d0
                call dgemm('N','T', nj, nk, 1, 1.0d0, cocc(ja+1+loop), nj, &
                           cocc(ka+1), nk, 0.0d0, xblk, nj)
                do j1 = 1, nj
                  do k1 = 1, nk
                    l = l + 1
                    p(l) = p(l) + xblk(j1, k1)
                  end do
                end do
                deallocate(xblk)
              else
#endif
                do j1 = 1, nj
                  sum = cocc(ja+j1+loop)
                  do k1 = 1, iorbs(k)
                    k2 = ka + k1
                    l = l + 1
                    p(l) = p(l) + cocc(k2) * sum
                  end do
                end do
#ifdef GPU
              end if
#endif
            end if
            ka = ka + iorbs(k)
          end do
          ja = ja + nj
        end do
      else
        do jj = nncf(i) + 1, nncf(i) + ncf(i)
          j = icocc(jj)
          nj = iorbs(j)
          ka = loop
          do kk = nncf(i) + 1, nncf(i) + ncf(i)
            k = icocc(kk)
            l = ijbo (j, k)
            if (j == k) then
#ifdef GPU
              if (mozyme_gpu .and. lgpu .and. (nj >= mozyme_gpu_min_block)) then
                nb = nj
                lbase = l
                allocate(xmat(nb, nb))
                xmat = 0.0d0
                allocate(avec(nb,1))
                avec(:,1) = cocc(ja+1+loop:ja+nb+loop)
                if (ngpus > 1 .or. mozyme_force_2gpu) then
                  call syrk_cublas_2gpu('L','N', nb, 1, 1.0d0, avec, nb, 0.0d0, xmat, nb)
                else
                  call syrk_cublas('L','N', nb, 1, 1.0d0, avec, nb, 0.0d0, xmat, nb)
                end if
                deallocate(avec)
                l = lbase
                do j1 = 1, nb
                  do k1 = 1, j1
                    l = l + 1
                    p(l) = p(l) + xmat(j1, k1)
                  end do
                end do
                deallocate(xmat)
              else if (mozyme_gpu .and. (nj >= mozyme_gpu_min_block)) then
                nb = nj
                lbase = l
                allocate(xmat(nb, nb))
                xmat = 0.0d0
                call dsyrk('L','N', nb, 1, 1.0d0, cocc(ja+1+loop), nb, 0.0d0, xmat, nb)
                l = lbase
                do j1 = 1, nb
                  do k1 = 1, j1
                    l = l + 1
                    p(l) = p(l) + xmat(j1, k1)
                  end do
                end do
                deallocate(xmat)
              else
#endif
                do j1 = 1, nj
                  sum = cocc(ja+j1+loop)
                  do k1 = 1, j1
                    k2 = ka + k1
                    l = l + 1
                    p(l) = p(l) + cocc(k2) * sum
                  end do
                end do
#ifdef GPU
              end if
#endif
            else if (j > k .and. l >= 0) then
#ifdef GPU
              if (mozyme_gpu .and. lgpu .and. (nj >= mozyme_gpu_min_block) .and. (iorbs(k) >= mozyme_gpu_min_block)) then
                nk = iorbs(k)
                allocate(xblk(nj, nk))
                xblk = 0.0d0
                allocate(avec(nj,1), bvec(nk,1))
                avec(:,1) = cocc(ja+1+loop:ja+nj+loop)
                bvec(:,1) = cocc(ka+1:ka+nk)
                if (ngpus > 1 .or. mozyme_force_2gpu) then
                  call gemm_cublas_2gpu('N','T', nj, nk, 1, 1.0d0, avec, nj, bvec, nk, 0.0d0, xblk, nj)
                else
                  call gemm_cublas('N','T', nj, nk, 1, 1.0d0, avec, nj, bvec, nk, 0.0d0, xblk, nj)
                end if
                deallocate(avec, bvec)
                do j1 = 1, nj
                  do k1 = 1, nk
                    l = l + 1
                    p(l) = p(l) + xblk(j1, k1)
                  end do
                end do
                deallocate(xblk)
              else if (mozyme_gpu .and. (nj >= mozyme_gpu_min_block) .and. (iorbs(k) >= mozyme_gpu_min_block)) then
                nk = iorbs(k)
                allocate(xblk(nj, nk))
                xblk = 0.0d0
                call dgemm('N','T', nj, nk, 1, 1.0d0, cocc(ja+1+loop), nj, &
                           cocc(ka+1), nk, 0.0d0, xblk, nj)
                do j1 = 1, nj
                  do k1 = 1, nk
                    l = l + 1
                    p(l) = p(l) + xblk(j1, k1)
                  end do
                end do
                deallocate(xblk)
              else
#endif
                do j1 = 1, nj
                  sum = cocc(ja+j1+loop)
                  do k1 = 1, iorbs(k)
                    k2 = ka + k1
                    l = l + 1
                    p(l) = p(l) + cocc(k2) * sum
                  end do
                end do
#ifdef GPU
              end if
#endif
            end if
            ka = ka + iorbs(k)
          end do
          ja = ja + nj
        end do
      end if
    end do
    if (mode == 0 .or. mode == 1) then
      !
      !    FULL DENSITY CALCULATION.  MULTIPLY BY 2 FOR SPIN
      !
      spinfa = 2.d0
    else if (mode ==-1) then
      !
      !   MAKING PARTIAL DENSITY MATRIX. REVERSE SIGN ONCE MORE
      !
      spinfa = -2.d0
    else
      spinfa = 1.d0
    end if
   !
    if (Abs(spinfa - 1.d0) > 1.d-10) then
      p(:) = spinfa * p(:)
    end if
    if (prnt) then
      sum = 0.d0
      if (lijbo) then
        do i = 1, numat
          j = nijbo(i, i) + 1
          if (iorbs(i) > 0) then
            sum = sum + p(j)
          end if
          if (iorbs(i) == 4) then
            sum = sum + p(j+2) + p(j+5) + p(j+9)
          end if
        end do
      else
        do i = 1, numat
          j = ijbo (i, i) + 1
          if (iorbs(i) > 0) then
            sum = sum + p(j)
          end if
          if (iorbs(i) == 4) then
            sum = sum + p(j+2) + p(j+5) + p(j+9)
          end if
        end do
      end if
      write (iw,*) " COMPUTED NUMBER OF ELECTRONS:", sum
    end if
end subroutine density_for_MOZYME
