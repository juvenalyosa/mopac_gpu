module gpu_scf_types
  use iso_c_binding
  implicit none
  integer, parameter :: GPU_SCF_FLAG_USE_DIIS = 1
  integer, parameter :: GPU_SCF_FLAG_RHF      = 2
  integer, parameter :: GPU_SCF_FLAG_UHF      = 4
  integer, parameter :: GPU_SCF_FLAG_DEBUG    = 8
  type, bind(C) :: gpu_scf_context
     integer(c_int)       :: norbs         = 0
     integer(c_int)       :: nalpha        = 0
     integer(c_int)       :: nbeta         = 0
     integer(c_int)       :: mpack         = 0
     integer(c_int)       :: max_iter      = 0
     integer(c_int)       :: numat         = 0
     integer(c_int)       :: n2elec        = 0
     integer(c_int)       :: periodic      = 0
     integer(c_int)       :: iterations    = 0
     integer(c_int)       :: flags         = 0
     real(c_double)       :: energy_tol    = 0.0_c_double
     real(c_double)       :: density_tol   = 0.0_c_double
     real(c_double)       :: energy_total  = 0.0_c_double
     real(c_double)       :: energy_delta  = 0.0_c_double
     real(c_double)       :: density_rms   = 0.0_c_double
     type(c_ptr)          :: h_core        = c_null_ptr
     type(c_ptr)          :: overlap       = c_null_ptr
     type(c_ptr)          :: density_alpha = c_null_ptr
     type(c_ptr)          :: density_beta  = c_null_ptr
     type(c_ptr)          :: density_total = c_null_ptr
     type(c_ptr)          :: fock_alpha    = c_null_ptr
     type(c_ptr)          :: fock_beta     = c_null_ptr
     type(c_ptr)          :: coeff_alpha   = c_null_ptr
     type(c_ptr)          :: coeff_beta    = c_null_ptr
     type(c_ptr)          :: eigvals_alpha = c_null_ptr
     type(c_ptr)          :: eigvals_beta  = c_null_ptr
     type(c_ptr)          :: nfirst        = c_null_ptr
     type(c_ptr)          :: nlast         = c_null_ptr
     type(c_ptr)          :: two_e_w       = c_null_ptr
     type(c_ptr)          :: two_e_wj      = c_null_ptr
     type(c_ptr)          :: two_e_wk      = c_null_ptr
     type(c_ptr)          :: work          = c_null_ptr
     type(c_ptr)          :: log_buffer    = c_null_ptr
  end type gpu_scf_context
contains

  subroutine gpu_scf_context_clear(ctx)
    type(gpu_scf_context), intent(inout) :: ctx
    ctx%norbs         = 0
    ctx%nalpha        = 0
    ctx%nbeta         = 0
    ctx%mpack         = 0
    ctx%max_iter      = 0
    ctx%numat         = 0
    ctx%n2elec        = 0
    ctx%periodic      = 0
    ctx%iterations    = 0
    ctx%flags         = 0
    ctx%energy_tol    = 0.0_c_double
    ctx%density_tol   = 0.0_c_double
    ctx%energy_total  = 0.0_c_double
    ctx%energy_delta  = 0.0_c_double
    ctx%density_rms   = 0.0_c_double
    ctx%h_core        = c_null_ptr
    ctx%overlap       = c_null_ptr
    ctx%density_alpha = c_null_ptr
    ctx%density_beta  = c_null_ptr
    ctx%density_total = c_null_ptr
    ctx%fock_alpha    = c_null_ptr
    ctx%fock_beta     = c_null_ptr
    ctx%coeff_alpha   = c_null_ptr
    ctx%coeff_beta    = c_null_ptr
    ctx%eigvals_alpha = c_null_ptr
    ctx%eigvals_beta  = c_null_ptr
    ctx%nfirst        = c_null_ptr
    ctx%nlast         = c_null_ptr
    ctx%two_e_w       = c_null_ptr
    ctx%two_e_wj      = c_null_ptr
    ctx%two_e_wk      = c_null_ptr
    ctx%work          = c_null_ptr
    ctx%log_buffer    = c_null_ptr
  end subroutine gpu_scf_context_clear

end module gpu_scf_types
