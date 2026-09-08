module gpu_mozyme_scf_interfaces
  use iso_c_binding
  implicit none
  private

  integer(c_int), parameter, public :: GPU_MOZYME_SCF_ABI_VERSION = 33_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_SUCCESS      = 0_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_NOT_READY    = -1_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_BAD_ARGUMENT = -2_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_UNSUPPORTED  = -3_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_CPU_BOUNDARY = -4_c_int
  integer(c_int), parameter, public :: &
    GPU_MOZYME_SCF_RESIDENT_DECISION_COMPLETE = 1_c_int

  integer(c_int), parameter, public :: GPU_MOZYME_SCF_STAGE_UPLOAD  = 1_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_STAGE_EIMP    = 2_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_STAGE_DIAGG   = 4_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_STAGE_DENSITY = 8_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_STAGE_FOCK    = 16_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_STAGE_CNVGZ   = 32_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_STAGE_HELECZ  = 64_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_STAGE_ISITSC  = 128_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_STAGE_ADDHB   = 256_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_STAGE_CHECK   = 512_c_int
  integer(c_int), parameter, public :: GPU_MOZYME_SCF_STAGE_FULL = &
    GPU_MOZYME_SCF_STAGE_UPLOAD + GPU_MOZYME_SCF_STAGE_EIMP + &
    GPU_MOZYME_SCF_STAGE_DIAGG + GPU_MOZYME_SCF_STAGE_DENSITY + &
    GPU_MOZYME_SCF_STAGE_ADDHB + GPU_MOZYME_SCF_STAGE_FOCK + &
    GPU_MOZYME_SCF_STAGE_CNVGZ + GPU_MOZYME_SCF_STAGE_HELECZ + &
    GPU_MOZYME_SCF_STAGE_ISITSC + GPU_MOZYME_SCF_STAGE_CHECK

  type, bind(C), public :: gpu_mozyme_scf_config
     integer(c_int) :: version   = GPU_MOZYME_SCF_ABI_VERSION
     integer(c_int) :: natoms    = 0_c_int
     integer(c_int) :: norbs     = 0_c_int
     integer(c_int) :: mpack     = 0_c_int
     integer(c_int) :: noccupied = 0_c_int
     integer(c_int) :: nvirtual  = 0_c_int
     integer(c_int) :: total_occupied = 0_c_int
     integer(c_int) :: total_virtual  = 0_c_int
     integer(c_int) :: max_iter  = 0_c_int
     integer(c_int) :: current_iter = 0_c_int
     integer(c_int) :: use_three_point = 0_c_int
     integer(c_int) :: density_mode = 0_c_int
     integer(c_int) :: fock_mode = 0_c_int
     integer(c_int) :: resident_fock_plan_id = 0_c_int
     integer(c_int) :: resident_fock_plan_full_coverage = 0_c_int
     integer(c_int) :: resident_fock_plan_partial_coverage = 0_c_int
     integer(c_int) :: resident_fock_plan_required_mask = 0_c_int
     integer(c_int) :: resident_fock_plan_covered_mask = 0_c_int
     integer(c_int) :: flags     = 0_c_int
     integer(c_int) :: diagg_mode = 0_c_int
     integer(c_int) :: density_indi = 0_c_int
     integer(c_int) :: lstart = 0_c_int
     real(c_double) :: shift = 0.0_c_double
     real(c_double) :: thresh = 0.0_c_double
     real(c_double) :: selcon = 0.0_c_double
     real(c_double) :: diagg_rot_const = 1.0_c_double
     real(c_double) :: diagg_bigeps = 0.0_c_double
     real(c_double) :: diagg_fref = 10.0_c_double
     real(c_double) :: diagg_oldlim = 0.0_c_double
     real(c_double) :: diagg_safety = 1.0_c_double
     integer(c_int) :: diagg_retry = 0_c_int
     integer(c_int) :: diagg_nf = 0_c_int
     integer(c_int) :: nhb = 0_c_int
     integer(c_int) :: addhb_due = 0_c_int
     integer(c_int) :: diagg2_nrejct(2) = 0_c_int
     integer(c_int) :: isitsc_iemin = 0_c_int
     integer(c_int) :: isitsc_iemax = 0_c_int
     integer(c_int) :: isitsc_scf1 = 0_c_int
     real(c_double) :: energy_scale = 1.0_c_double
     real(c_double) :: energy_offset = 0.0_c_double
     real(c_double) :: previous_escf = 0.0_c_double
     real(c_double) :: emin = 0.0_c_double
     real(c_double) :: ovmax = 0.0_c_double
     real(c_double) :: isitsc_escf0(10) = 0.0_c_double
  end type gpu_mozyme_scf_config

  type, bind(C), public :: gpu_mozyme_scf_state
     integer(c_int) :: version   = GPU_MOZYME_SCF_ABI_VERSION
     integer(c_int) :: flags     = 0_c_int
     integer(c_int) :: use_nijbo = 0_c_int
     integer(c_int) :: icocc_dim = 0_c_int
     integer(c_int) :: cocc_dim  = 0_c_int
     integer(c_int) :: icvir_dim = 0_c_int
     integer(c_int) :: cvir_dim  = 0_c_int
     integer(c_int) :: fmo_dim   = 0_c_int
     integer(c_int) :: partp_dim = 0_c_int
     integer(c_int) :: partf_dim = 0_c_int
     integer(c_int) :: nocc_slots = 0_c_int
     integer(c_int) :: nvir_slots = 0_c_int
     integer(c_int) :: p_dim = 0_c_int
     integer(c_int) :: f_dim = 0_c_int
     integer(c_int) :: h_dim = 0_c_int
     integer(c_int) :: pold_dim = 0_c_int
     integer(c_int) :: p1_dim = 0_c_int
     integer(c_int) :: p2_dim = 0_c_int
     integer(c_int) :: p3_dim = 0_c_int
     integer(c_int) :: idiag_dim = 0_c_int
     integer(c_int) :: iorbs_dim = 0_c_int
     integer(c_int) :: kopt_dim = 0_c_int
     integer(c_int) :: ncf_dim = 0_c_int
     integer(c_int) :: nncf_dim = 0_c_int
     integer(c_int) :: ncocc_dim = 0_c_int
     integer(c_int) :: nce_dim = 0_c_int
     integer(c_int) :: nnce_dim = 0_c_int
     integer(c_int) :: ncvir_dim = 0_c_int
     integer(c_int) :: ifmo_rows = 0_c_int
     integer(c_int) :: ifmo_cols = 0_c_int
     integer(c_int) :: eigs_dim = 0_c_int
     integer(c_int) :: nfmo_dim = 0_c_int
     integer(c_int) :: nfirst_dim = 0_c_int
     integer(c_int) :: nlast_dim = 0_c_int
     integer(c_int) :: nijbo_rows = 0_c_int
     integer(c_int) :: nijbo_cols = 0_c_int
     integer(c_int) :: coord_rows = 0_c_int
     integer(c_int) :: coord_cols = 0_c_int
     integer(c_int) :: nat_dim = 0_c_int
     integer(c_int) :: cosmo_enabled = 0_c_int
     integer(c_int) :: cosmo_nps = 0_c_int
     integer(c_int) :: cosmo_lm61 = 0_c_int
     integer(c_int) :: cosmo_cosurf_rows = 0_c_int
     integer(c_int) :: cosmo_cosurf_cols = 0_c_int
     integer(c_int) :: cosmo_phinet_rows = 0_c_int
     integer(c_int) :: cosmo_phinet_cols = 0_c_int
     integer(c_int) :: cosmo_qscnet_rows = 0_c_int
     integer(c_int) :: cosmo_qscnet_cols = 0_c_int
     integer(c_int) :: cosmo_qdenet_rows = 0_c_int
     integer(c_int) :: cosmo_qdenet_cols = 0_c_int
     integer(c_int) :: cosmo_qscat_dim = 0_c_int
     integer(c_int) :: cosmo_srad_dim = 0_c_int
     integer(c_int) :: cosmo_npoints_dim = 0_c_int
     integer(c_int) :: cosmo_a_diag_dim = 0_c_int
     integer(c_int) :: cosmo_a_part_dim = 0_c_int
     integer(c_int) :: cosmo_m_vec_dim = 0_c_int
     integer(c_int) :: cosmo_iblock_pos_dim = 0_c_int
     integer(c_int) :: cosmo_new_surface = 0_c_int
     integer(c_int) :: param_dim = 0_c_int
     real(c_double) :: cosmo_fepsi = 0.0_c_double
     real(c_double) :: cosmo_disex2 = 0.0_c_double
     real(c_double) :: cosmo_solv_energy = 0.0_c_double
     real(c_double) :: cosmo_ediel = 0.0_c_double
     real(c_double) :: cosmo_a0 = 0.0_c_double
     real(c_double) :: cosmo_ev = 0.0_c_double
     type(c_ptr) :: p      = c_null_ptr
     type(c_ptr) :: f      = c_null_ptr
     type(c_ptr) :: h      = c_null_ptr
     type(c_ptr) :: partp  = c_null_ptr
     type(c_ptr) :: partf  = c_null_ptr
     type(c_ptr) :: pold   = c_null_ptr
     type(c_ptr) :: p1     = c_null_ptr
     type(c_ptr) :: p2     = c_null_ptr
     type(c_ptr) :: p3     = c_null_ptr
     type(c_ptr) :: idiag  = c_null_ptr
     type(c_ptr) :: iorbs  = c_null_ptr
     type(c_ptr) :: kopt   = c_null_ptr
     type(c_ptr) :: ncf    = c_null_ptr
     type(c_ptr) :: nncf   = c_null_ptr
     type(c_ptr) :: ncocc  = c_null_ptr
     type(c_ptr) :: icocc  = c_null_ptr
     type(c_ptr) :: cocc   = c_null_ptr
     type(c_ptr) :: nce    = c_null_ptr
     type(c_ptr) :: nnce   = c_null_ptr
     type(c_ptr) :: ncvir  = c_null_ptr
     type(c_ptr) :: icvir  = c_null_ptr
     type(c_ptr) :: cvir   = c_null_ptr
     type(c_ptr) :: fmo    = c_null_ptr
     type(c_ptr) :: ifmo   = c_null_ptr
     type(c_ptr) :: eigs   = c_null_ptr
     type(c_ptr) :: nfmo   = c_null_ptr
     type(c_ptr) :: nfirst = c_null_ptr
     type(c_ptr) :: nlast  = c_null_ptr
     type(c_ptr) :: nijbo  = c_null_ptr
     type(c_ptr) :: coord = c_null_ptr
     type(c_ptr) :: nat = c_null_ptr
     type(c_ptr) :: param_dd = c_null_ptr
     type(c_ptr) :: param_qq = c_null_ptr
     type(c_ptr) :: param_tore = c_null_ptr
     type(c_ptr) :: cosmo_iatsp = c_null_ptr
     type(c_ptr) :: cosmo_ipiden = c_null_ptr
     type(c_ptr) :: cosmo_gden = c_null_ptr
     type(c_ptr) :: cosmo_qscat = c_null_ptr
     type(c_ptr) :: cosmo_srad = c_null_ptr
     type(c_ptr) :: cosmo_cosurf = c_null_ptr
     type(c_ptr) :: cosmo_phinet = c_null_ptr
     type(c_ptr) :: cosmo_qscnet = c_null_ptr
     type(c_ptr) :: cosmo_qdenet = c_null_ptr
     type(c_ptr) :: cosmo_npoints = c_null_ptr
     type(c_ptr) :: cosmo_a_diag = c_null_ptr
     type(c_ptr) :: cosmo_a_part = c_null_ptr
     type(c_ptr) :: cosmo_a_part_i = c_null_ptr
     type(c_ptr) :: cosmo_a_part_j = c_null_ptr
     type(c_ptr) :: cosmo_m_vec = c_null_ptr
     type(c_ptr) :: cosmo_iblock_pos = c_null_ptr
     type(c_ptr) :: cosmo_solv_energy_ptr = c_null_ptr
     type(c_ptr) :: cosmo_ediel_ptr = c_null_ptr
  end type gpu_mozyme_scf_state

  type, bind(C), public :: gpu_mozyme_scf_status
     integer(c_int) :: version   = GPU_MOZYME_SCF_ABI_VERSION
     integer(c_int) :: code      = GPU_MOZYME_SCF_NOT_READY
     integer(c_int) :: ready     = 0_c_int
     integer(c_int) :: resident  = 0_c_int
     integer(c_int) :: device_id = -1_c_int
     integer(c_int) :: natoms    = 0_c_int
     integer(c_int) :: norbs     = 0_c_int
     integer(c_int) :: mpack     = 0_c_int
     integer(c_int) :: iterations = 0_c_int
     integer(c_int) :: stage_completed = 0_c_int
     integer(c_int) :: stage_required  = GPU_MOZYME_SCF_STAGE_FULL
     integer(c_int) :: stage_missing   = GPU_MOZYME_SCF_STAGE_FULL
     integer(c_int) :: resident_decision = 0_c_int
     integer(c_int) :: resident_fock_plan_id = 0_c_int
     integer(c_int) :: resident_fock_plan_full_coverage = 0_c_int
     integer(c_int) :: resident_fock_plan_partial_coverage = 0_c_int
     integer(c_int) :: resident_fock_plan_required_mask = 0_c_int
     integer(c_int) :: resident_fock_plan_covered_mask = 0_c_int
     integer(c_int) :: final_publication_done = 0_c_int
     integer(c_int) :: final_publication_arrays = 0_c_int
     integer(c_size_t) :: final_publication_bytes = 0_c_size_t
     integer(c_int) :: final_publication_cosmo = 0_c_int
     real(c_double) :: energy_total = 0.0_c_double
     real(c_double) :: energy_delta = 0.0_c_double
     real(c_double) :: density_max  = 0.0_c_double
     real(c_double) :: density_rms  = 0.0_c_double
     real(c_double) :: wall_ms      = 0.0_c_double
     integer(c_int) :: diagg_nij    = 0_c_int
     integer(c_int) :: diagg_nf     = 0_c_int
     integer(c_int) :: idiagg       = 0_c_int
     integer(c_int) :: nhb          = 0_c_int
     integer(c_int) :: addhb_due    = 0_c_int
     integer(c_int) :: addhb_applied = 0_c_int
     integer(c_int) :: addhb_nij    = 0_c_int
     integer(c_int) :: diagg2_nrejct(2) = 0_c_int
     real(c_double) :: diagg_tiny   = 0.0_c_double
     real(c_double) :: next_tiny    = 0.0_c_double
     real(c_double) :: diagg_fref   = 0.0_c_double
     real(c_double) :: diagg_oldlim = 0.0_c_double
     real(c_double) :: diagg_safety = 0.0_c_double
     real(c_double) :: diagg_sumt   = 0.0_c_double
     real(c_double) :: diagg_sumb   = 0.0_c_double
     real(c_double) :: energy_scf   = 0.0_c_double
     integer(c_int) :: isitsc_okscf = 0_c_int
     integer(c_int) :: isitsc_iscf  = 0_c_int
     integer(c_int) :: isitsc_iemin = 0_c_int
     integer(c_int) :: isitsc_iemax = 0_c_int
     integer(c_int) :: isitsc_scf1  = 0_c_int
     integer(c_int) :: use_three_point = 0_c_int
     integer(c_int) :: lstart = 0_c_int
     real(c_double) :: shift = 0.0_c_double
     real(c_double) :: isitsc_escf0(10) = 0.0_c_double
     integer(c_int) :: resident_stage_calls(10) = 0_c_int
     real(c_double) :: resident_stage_ms(10) = 0.0_c_double
     integer(c_int) :: final_reorth_applied = 0_c_int
     real(c_double) :: final_reorth_ms = 0.0_c_double
     real(c_double) :: final_reorth_sum = 0.0_c_double
     integer(c_int) :: pls_supervisor_calls = 0_c_int
     integer(c_int) :: pls_restart_required = 0_c_int
     integer(c_int) :: pls_history_count = 0_c_int
     real(c_double) :: pls_ovmax_delta = 0.0_c_double
     real(c_double) :: pls_energy_delta = 0.0_c_double
     integer(c_int) :: pls_restart_reset_device_calls = 0_c_int
     integer(c_int) :: pls_restart_done = 0_c_int
     integer(c_int) :: cosmo_enabled = 0_c_int
     integer(c_int) :: cosmo_fock_calls = 0_c_int
     integer(c_int) :: cosmo_matvec_calls = 0_c_int
     integer(c_int) :: cosmo_cg_iterations = 0_c_int
     integer(c_int) :: cosmo_nps = 0_c_int
     integer(c_int) :: cosmo_lm61 = 0_c_int
     integer(c_int) :: cosmo_pair_count = 0_c_int
     real(c_double) :: cosmo_solv_energy = 0.0_c_double
     real(c_double) :: cosmo_ediel = 0.0_c_double
     real(c_double) :: cosmo_last_residual = 0.0_c_double
     integer(c_int) :: cosmo_cg_control_resident = 0_c_int
     integer(c_int) :: cosmo_cg_converged = 0_c_int
     integer(c_int) :: cosmo_cg_breakdown = 0_c_int
     integer(c_int) :: cosmo_cg_host_syncs = 0_c_int
     real(c_double) :: cosmo_cg_target_tol = 0.0_c_double
     integer(c_int) :: cnvgz_active_calls = 0_c_int
     integer(c_int) :: cnvgz_noop_calls = 0_c_int
     integer(c_int) :: strict_resident_host_syncs = 0_c_int
     integer(c_int) :: strict_resident_control_polls = 0_c_int
  end type gpu_mozyme_scf_status

  public :: mopac_cuda_mozyme_scf_setup
  public :: mopac_cuda_mozyme_scf_register_state
  public :: mopac_cuda_mozyme_scf_run
  public :: mopac_cuda_mozyme_scf_destroy
  public :: mopac_cuda_mozyme_scf_status

  interface
    function mopac_cuda_mozyme_scf_setup(config, context) &
      bind(C, name='mopac_cuda_mozyme_scf_setup') result(code)
      import :: c_int, c_ptr, gpu_mozyme_scf_config
      type(gpu_mozyme_scf_config), intent(in) :: config
      type(c_ptr), intent(out) :: context
      integer(c_int) :: code
    end function mopac_cuda_mozyme_scf_setup

    function mopac_cuda_mozyme_scf_register_state(context, state) &
      bind(C, name='mopac_cuda_mozyme_scf_register_state') result(code)
      import :: c_int, c_ptr, gpu_mozyme_scf_state
      type(c_ptr), value :: context
      type(gpu_mozyme_scf_state), intent(in) :: state
      integer(c_int) :: code
    end function mopac_cuda_mozyme_scf_register_state

    function mopac_cuda_mozyme_scf_run(context, status) &
      bind(C, name='mopac_cuda_mozyme_scf_run') result(code)
      import :: c_int, c_ptr, gpu_mozyme_scf_status
      type(c_ptr), value :: context
      type(gpu_mozyme_scf_status), intent(out) :: status
      integer(c_int) :: code
    end function mopac_cuda_mozyme_scf_run

    function mopac_cuda_mozyme_scf_destroy(context) &
      bind(C, name='mopac_cuda_mozyme_scf_destroy') result(code)
      import :: c_int, c_ptr
      type(c_ptr), value :: context
      integer(c_int) :: code
    end function mopac_cuda_mozyme_scf_destroy

    function mopac_cuda_mozyme_scf_status(context, status) &
      bind(C, name='mopac_cuda_mozyme_scf_status') result(code)
      import :: c_int, c_ptr, gpu_mozyme_scf_status
      type(c_ptr), value :: context
      type(gpu_mozyme_scf_status), intent(out) :: status
      integer(c_int) :: code
    end function mopac_cuda_mozyme_scf_status
  end interface

end module gpu_mozyme_scf_interfaces
