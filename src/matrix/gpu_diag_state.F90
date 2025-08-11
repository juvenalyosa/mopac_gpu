module gpu_diag_state
  implicit none
  logical :: have_device_eigvecs = .false.
  integer :: device_eigvecs_n = 0
contains
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
  end subroutine gpu_diag_clear
end module gpu_diag_state

