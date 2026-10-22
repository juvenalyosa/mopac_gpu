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

module gpu_diag_state
  implicit none
  logical :: have_device_eigvecs = .false.
  integer :: device_eigvecs_n = 0
  ! Fingerprint of the host copy of the kept eigenvectors (gpu_diag_fingerprint_set): the
  ! density is built from the device copy only while the host matrix is still that one
  ! (iter's swap, for example, exchanges columns after the diagonalization).
  double precision :: device_eigvecs_fp = 0.d0
  logical :: device_eigvecs_fp_set = .false.
contains
  double precision function eigvecs_fingerprint(c, n) result(fp)
    implicit none
    integer, intent(in) :: n
    double precision, intent(in) :: c(n, n)
    integer :: j, m
    ! three rows of every column, weighted by the column number: O(n), and a column
    ! exchange or another matrix changes it
    m = max(1, n/2)
    fp = 0.d0
    do j = 1, n
      fp = fp + dble(j)*(c(1, j) + 2.d0*c(m, j) + 3.d0*c(n, j))
    end do
  end function eigvecs_fingerprint

  subroutine gpu_diag_fingerprint_set(c, n)
    implicit none
    integer, intent(in) :: n
    double precision, intent(in) :: c(n, n)
    device_eigvecs_fp = eigvecs_fingerprint(c, n)
    device_eigvecs_fp_set = .true.
  end subroutine gpu_diag_fingerprint_set

  ! True when the device holds the eigenvectors of size n and the host matrix c is
  ! still the one they were fetched into.
  logical function gpu_diag_matches(c, n) result(ok)
#ifdef GPU
    use iso_c_binding, only: c_int
#endif
    implicit none
    integer, intent(in) :: n
    double precision, intent(in) :: c(n, n)
#ifdef GPU
    interface
      function mopac_cuda_device_eigvecs_valid(n_c) bind(C, name='mopac_cuda_device_eigvecs_valid') result(v)
        import :: c_int
        integer(c_int), value :: n_c
        integer(c_int) :: v
      end function mopac_cuda_device_eigvecs_valid
    end interface
    ok = have_device_eigvecs .and. device_eigvecs_n == n .and. device_eigvecs_fp_set
    if (.not. ok) return
    ok = mopac_cuda_device_eigvecs_valid(int(n, c_int)) == 1
    if (.not. ok) return
    ok = eigvecs_fingerprint(c, n) == device_eigvecs_fp
#else
    ok = .false.
    if (n < 0) ok = c(1, 1) /= c(1, 1)
#endif
  end function gpu_diag_matches

  subroutine gpu_diag_mark(n)
    implicit none
    integer, intent(in) :: n
    have_device_eigvecs = .true.
    device_eigvecs_n = n
  end subroutine gpu_diag_mark

  subroutine gpu_diag_clear()
    implicit none
    have_device_eigvecs = .false.
    device_eigvecs_n = 0
    device_eigvecs_fp_set = .false.
  end subroutine gpu_diag_clear
end module gpu_diag_state

