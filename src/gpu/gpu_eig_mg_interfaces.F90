! Molecular Orbital PACkage (MOPAC)
! Copyright 2025-2026 Juvenal Yosa, PhD (juvenal.yosa@gmail.com)
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

! Fortran interface for (future) cuSOLVERMg eigensolver integration
module gpu_eig_mg_interfaces
  use iso_c_binding
  implicit none
  interface
    subroutine mopac_cusolvermg_dsyevd(n, a, lda, w, info) bind(C, name='mopac_cusolvermg_dsyevd')
      import :: c_int, c_double
      integer(c_int), value :: n, lda
      real(c_double)        :: a(lda, n)
      real(c_double)        :: w(n)
      integer(c_int)        :: info
    end subroutine mopac_cusolvermg_dsyevd

    logical(c_bool) function mopac_cuda_has_cusolvermg() bind(C, name='mopac_cuda_has_cusolvermg')
      import :: c_bool
    end function mopac_cuda_has_cusolvermg
  end interface
end module gpu_eig_mg_interfaces
