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

module gpu_diis_interfaces
  use iso_c_binding
  implicit none
  interface
    subroutine mopac_cuda_diis_init(linear, maxfock) bind(C,name='mopac_cuda_diis_init')
      import :: c_int
      integer(c_int), value :: linear, maxfock
    end subroutine mopac_cuda_diis_init

    subroutine mopac_cuda_diis_store(linear, col, r) bind(C,name='mopac_cuda_diis_store')
      import :: c_int, c_double
      integer(c_int), value :: linear, col
      real(c_double)        :: r(linear)
    end subroutine mopac_cuda_diis_store

    function mopac_cuda_diis_residual_resident(n, linear, col, f_host, p_host, host_out, copy_back) &
      bind(C,name='mopac_cuda_diis_residual_resident') result(ok)
      import :: c_int, c_ptr, c_bool
      integer(c_int), value :: n, linear, col
      type(c_ptr), value :: f_host, p_host, host_out
      integer(c_int), value :: copy_back
      logical(c_bool) :: ok
    end function mopac_cuda_diis_residual_resident

    subroutine mopac_cuda_diis_bcol(linear, nfock, lfock, out) bind(C,name='mopac_cuda_diis_bcol')
      import :: c_int, c_double
      integer(c_int), value :: linear, nfock, lfock
      real(c_double)        :: out(nfock)
    end subroutine mopac_cuda_diis_bcol

    subroutine mopac_cuda_diis_release() bind(C,name='mopac_cuda_diis_release')
    end subroutine mopac_cuda_diis_release
  end interface
end module gpu_diis_interfaces
