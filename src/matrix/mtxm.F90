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

      subroutine mtxm(a, nar, b, nbr, c, ncc)
#ifdef GPU
      use mod_vars_cuda, only: lgpu, ngpus
      use mopac_cublas_interfaces
#endif
      implicit none
      integer  :: nar
      integer  :: nbr
      integer  :: ncc
      double precision  :: a(nbr,nar)
      double precision  :: b(nbr,ncc)
      double precision  :: c(nar,ncc)
!-----------------------------------------------
!     MATRIX PRODUCT C(NAR,NCC) = (A(NBR,NAR))' * B(NBR,NCC)
!     ALL MATRICES RECTANGULAR , PACKED.

#ifdef GPU
      if (lgpu) then
        if (ngpus > 1) then
          call gemm_cublas_multi('T','N', nar, ncc, nbr, 1.0d0, a, nbr, b, nbr, 0.0d0, c, nar)
        else
          call gemm_cublas('T','N', nar, ncc, nbr, 1.0d0, a, nbr, b, nbr, 0.0d0, c, nar)
        end if
      else
        call dgemm ('T', 'N', nar, ncc, nbr, 1.0D0, a, nbr, b, nbr, 0.0D0, c, nar)
      end if
#else
      call dgemm ('T', 'N', nar, ncc, nbr, 1.0D0, a, nbr, b, nbr, 0.0D0, c, nar)
#endif
      return
      end subroutine mtxm
