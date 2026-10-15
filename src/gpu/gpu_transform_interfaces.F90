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

module gpu_transform_interfaces
  use iso_c_binding
  implicit none
  interface
    subroutine mopac_cuda_fmulC(n, fpacked, c, ldc, w, ldw) bind(C,name='mopac_cuda_fmulC')
      import :: c_int, c_double
      integer(c_int), value :: n, ldc, ldw
      real(c_double)        :: fpacked(n*(n+1)/2)
      real(c_double)        :: c(ldc, n)
      real(c_double)        :: w(ldw, n)
    end subroutine mopac_cuda_fmulC

    subroutine mopac_cuda_fmulC_from_dev(n, c, ldc, w, ldw) bind(C,name='mopac_cuda_fmulC_from_dev')
      import :: c_int, c_double
      integer(c_int), value :: n, ldc, ldw
      real(c_double)        :: c(ldc, n)
      real(c_double)        :: w(ldw, n)
    end subroutine mopac_cuda_fmulC_from_dev
  end interface
end module gpu_transform_interfaces
