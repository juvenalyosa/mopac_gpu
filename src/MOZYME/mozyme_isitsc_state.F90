module mozyme_isitsc_state
  use iso_c_binding, only: c_double
  implicit none
  private

  logical, public :: isitsc_scf1 = .false.
  real(c_double), public :: isitsc_escf0(10) = 0.0_c_double

  public :: mozyme_isitsc_set_state

contains

  subroutine mozyme_isitsc_set_state(scf1_value, escf0_value)
    implicit none
    logical, intent(in) :: scf1_value
    real(c_double), intent(in) :: escf0_value(10)

    isitsc_scf1 = scf1_value
    isitsc_escf0 = escf0_value
  end subroutine mozyme_isitsc_set_state

end module mozyme_isitsc_state
