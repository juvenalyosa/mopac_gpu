module mozyme_diagg1_state
  implicit none
  private

  integer, public :: diagg1_nf = 0
  integer, public :: diagg1_icalcn = 0
  integer, public :: diagg1_mydisp = 0
  integer, public :: diagg1_ij0 = 0
  double precision, public :: diagg1_fref = 10.0d0
  double precision, public :: diagg1_oldlim = 0.0d0
  double precision, public :: diagg1_safety = 1.0d0

  public :: mozyme_diagg1_set_state

contains

  subroutine mozyme_diagg1_set_state(numcal_value, nf_value, fref_value, &
      oldlim_value, safety_value)
    implicit none
    integer, intent(in) :: numcal_value, nf_value
    double precision, intent(in) :: fref_value, oldlim_value, safety_value

    diagg1_icalcn = numcal_value
    diagg1_nf = nf_value
    diagg1_fref = fref_value
    diagg1_oldlim = oldlim_value
    diagg1_safety = safety_value
  end subroutine mozyme_diagg1_set_state

end module mozyme_diagg1_state
