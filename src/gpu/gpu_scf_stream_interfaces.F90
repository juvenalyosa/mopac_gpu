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

module gpu_scf_stream_interfaces
  use iso_c_binding
  implicit none
  interface
    logical(c_bool) function mopac_cuda_scf_stream_supported() bind(C)
      import :: c_bool
    end function mopac_cuda_scf_stream_supported

    subroutine mopac_cuda_scf_stream_register(cookie) bind(C)
      import :: c_ptr
      type(c_ptr), value :: cookie
    end subroutine mopac_cuda_scf_stream_register

    subroutine mopac_cuda_scf_stream_publish(cookie, ia, ib, ja, jb, len, wj, wk, status) bind(C)
      import :: c_ptr, c_int, c_double
      type(c_ptr), value :: cookie
      integer(c_int), value :: ia, ib, ja, jb, len
      real(c_double), intent(in) :: wj(len), wk(len)
      integer(c_int), intent(out) :: status
    end subroutine mopac_cuda_scf_stream_publish

    subroutine mopac_cuda_scf_stream_finalize(cookie, status) bind(C)
      import :: c_ptr, c_int
      type(c_ptr), value :: cookie
      integer(c_int), intent(out) :: status
    end subroutine mopac_cuda_scf_stream_finalize
  end interface
end module gpu_scf_stream_interfaces
