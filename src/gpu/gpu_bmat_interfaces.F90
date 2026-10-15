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

module gpu_bmat_interfaces
  use iso_c_binding
  implicit none
  interface
    subroutine mopac_cuda_bcol_from_residuals(linear, nfock, fppf, lfock, out) bind(C,name='mopac_cuda_bcol_from_residuals')
      import :: c_int, c_double
      integer(c_int), value :: linear, nfock, lfock
      real(c_double)        :: fppf(linear*nfock)
      real(c_double)        :: out(nfock)
    end subroutine mopac_cuda_bcol_from_residuals

    subroutine mopac_cuda_bfull_from_host(linear, nfock, fppf, bout) bind(C,name='mopac_cuda_bfull_from_host')
      import :: c_int, c_double
      integer(c_int), value :: linear, nfock
      real(c_double)        :: fppf(linear*nfock)
      real(c_double)        :: bout(nfock, nfock)
    end subroutine mopac_cuda_bfull_from_host

    subroutine mopac_cuda_bfull_from_device(linear, nfock, bout) bind(C,name='mopac_cuda_bfull_from_device')
      import :: c_int, c_double
      integer(c_int), value :: linear, nfock
      real(c_double)        :: bout(nfock, nfock)
    end subroutine mopac_cuda_bfull_from_device
  end interface
end module gpu_bmat_interfaces
