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

! ---------------------------------------------------------------------------------------
      subroutine mult_symm_AB(a, b, alpha, ndim, mdim, c, beta, iopc)

!        Use mod_vars_cuda, only: ngpus
        Use iso_c_binding
#ifdef GPU
        Use call_gemm_cublas
        Use call_syrk_cublas
        Use mamult_cuda_i
        use common_arrays_C, only : ifact
#endif
        implicit none
        Integer :: iopc,ndim,mdim
        Integer :: i
#ifdef GPU
        integer :: igrid, iblock
        real :: tt
#endif
! Make arguments C_LOC-able for address comparison
#if CC12
        real(c_float), dimension(mdim), target :: a, b, c
#else
        real(c_double), dimension(mdim), target :: a, b, c
#endif
        double precision, allocatable, dimension(:,:) :: xa, xb, xc

        double precision :: alpha,beta
        logical :: same_ab
        type(c_ptr) :: pa, pb

        continue

! here, performs matrix multiplications

        Select case (iopc)

          case (1) ! mamult
            call mamult (a, b, c, ndim, beta)
#ifdef GPU
          case (2) ! mamult_gpu
            igrid = 512 ; iblock = 512
            tt = 0.0
            call mamult_gpu(a, b, c, ndim, mdim, ifact, beta, igrid, iblock, tt, 0)
#endif
          case (3) ! CPU path: prefer DSYRK when A==B
            allocate (xa(ndim,ndim), xb(ndim,ndim), xc(ndim,ndim),stat=i)

!            forall (i=1:ndim,j=1:ndim)
!              xa(i,j) = 0.d0
!              xb(i,j) = 0.d0
!            endforall

            call dtpttr( 'u', ndim, a, xa, ndim, i )
            if (i /= 0) stop 'error in dtpttr'

            call dtpttr( 'u', ndim, b, xb, ndim, i )
            if (i /= 0) stop 'error in dtpttr'

            do i = 1,ndim-1
               call dcopy(ndim-i,xa(i,i+1),ndim,xa(i+1,i),1)
               call dcopy(ndim-i,xb(i,i+1),ndim,xb(i+1,i),1)
            end do

            if (.not.(beta == 0.d0)) then
!              forall (i=1:ndim,j=1:ndim)
!                xc(i,j) = 0.d0
!              endforall

              call dtpttr( 'u', ndim, c, xc, ndim, i )
              if (i /= 0) stop 'error in dtpttr'
              do i = 1,ndim-1
                 call dcopy(ndim-i,xc(i,i+1),ndim,xc(i+1,i),1)
              end do
            end if

            pa = c_loc(a(1)); pb = c_loc(b(1))
            same_ab = (pa == pb)

            if (same_ab) then
              call dsyrk('U','N', ndim, ndim, alpha, xa, ndim, beta, xc, ndim)
            else
              call dgemm ("N", "N", ndim, ndim, ndim, alpha, xa, ndim, xb, ndim, beta, xc, &
                         & ndim)
            end if

            call dtrttp('u', ndim, xc, ndim, c, i )

            deallocate (xa,xb,xc,stat=i)

#ifdef GPU
          case (4) ! GPU path: prefer cuBLAS SYRK when A==B

            allocate (xa(ndim,ndim), xb(ndim,ndim), xc(ndim,ndim),stat=i)

!            forall (i=1:ndim,j=1:ndim)
!              xa(i,j) = 0.d0
!              xb(i,j) = 0.d0
!            endforall

            call dtpttr( 'u', ndim, a, xa, ndim, i )
            if (i /= 0) stop 'error in dtpttr'

            call dtpttr( 'u', ndim, b, xb, ndim, i )
            if (i /= 0) stop 'error in dtpttr'

            do i = 1,ndim-1
               call dcopy(ndim-i,xa(i,i+1),ndim,xa(i+1,i),1)
               call dcopy(ndim-i,xb(i,i+1),ndim,xb(i+1,i),1)
            end do

            if (beta /= 0.d0) then
!              forall (i=1:ndim,j=1:ndim)
!                xc(i,j) = 0.d0
!              endforall

              call dtpttr( 'u', ndim, c, xc, ndim, i )
              if (i /= 0) stop 'error in dtpttr'
              do i = 1,ndim-1
                 call dcopy(ndim-i,xc(i,i+1),ndim,xc(i+1,i),1)
              end do
            end if

            pa = c_loc(a(1)); pb = c_loc(b(1))
            same_ab = (pa == pb)

            if (same_ab) then
               call syrk_cublas('U', 'N', ndim, ndim, alpha, xa, ndim, beta, xc, ndim)
            else
               call gemm_cublas ("N", "N", ndim, ndim, ndim, alpha, xa, ndim, xb, ndim, beta, xc, &
                           & ndim)
            end if

            call dtrttp('u', ndim, xc, ndim, c, i )

            deallocate (xa,xb,xc,stat=i)
#endif
        end select

        continue

      return
      end subroutine mult_symm_AB

! ---------------------------------------------------------------------------------------
