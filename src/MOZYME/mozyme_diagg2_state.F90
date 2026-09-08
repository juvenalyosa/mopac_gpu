module mozyme_diagg2_state
  implicit none
  private

  integer, public :: diagg2_nrejct(2) = 0

  public :: mozyme_diagg2_retry
  public :: mozyme_diagg2_set_rejections
  public :: mozyme_diagg2_set_state

contains

  logical function mozyme_diagg2_retry() result(retry)
    implicit none

    retry = (diagg2_nrejct(1) == diagg2_nrejct(2) .and. &
      diagg2_nrejct(1) /= 0 .and. diagg2_nrejct(1) < 20)
  end function mozyme_diagg2_retry

  subroutine mozyme_diagg2_set_rejections(nrej)
    implicit none
    integer, intent(in) :: nrej

    diagg2_nrejct(2) = diagg2_nrejct(1)
    diagg2_nrejct(1) = nrej
  end subroutine mozyme_diagg2_set_rejections

  subroutine mozyme_diagg2_set_state(nrejct_value)
    implicit none
    integer, intent(in) :: nrejct_value(2)

    diagg2_nrejct = nrejct_value
  end subroutine mozyme_diagg2_set_state

end module mozyme_diagg2_state
