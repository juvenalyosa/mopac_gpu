! Molecular Orbital PACkage (MOPAC)
! Copyright 2021 Virginia Polytechnic Institute and State University
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

submodule (mopac_api) mopac_api_operations
  use Common_arrays_C, only: xparam, grad, lopt
  use molkst_C, only: keywrd, escf, moperr, nvar, jobnam, run, verson
  implicit none

  interface

    ! initialize MOPAC modules before calculations
    module subroutine mopac_initialize(system)
      type(mopac_system), intent(in) :: system
    end subroutine mopac_initialize

    ! extract data from MOPAC modules after calculations
    module subroutine mopac_finalize(properties)
      type(mopac_properties), intent(out) :: properties
    end subroutine mopac_finalize

    ! save MOPAC density matrices
    module subroutine mopac_save(state)
      type(mopac_state), intent(inout) :: state
    end subroutine mopac_save

    ! load MOPAC density matrices, or construct initial guesses
    module subroutine mopac_load(state)
      type(mopac_state), intent(in) :: state
    end subroutine mopac_load

    ! save MOZYME density matrix
    module subroutine mozyme_save(state)
      type(mozyme_state), intent(inout) :: state
    end subroutine mozyme_save

    ! load MOZYME density matrix, or construct initial guess
    module subroutine mozyme_load(state)
      type(mozyme_state), intent(in) :: state
    end subroutine mozyme_load

  end interface

contains

  ! MOPAC electronic ground state calculation
  module subroutine mopac_scf(system, state, properties) bind(c)
#ifdef WIN32
!dec$ attributes dllexport :: mopac_scf
#endif
    use mopac_api_flags, only: keep_fock
    type(mopac_system), intent(in) :: system
    type(mopac_state), intent(inout) :: state
    type(mopac_properties), intent(out) :: properties

    keywrd = " 1SCF PULAY BONDS"
    if (system%natom_move + system%nlattice_move > 0) keywrd = trim(keywrd) // " GRADIENTS"
    call mopac_initialize(system)
    if (.not. moperr) call mopac_load(state)
    ! call computational routine for SCF calculations
    if (.not. moperr) call compfg (xparam, .true., escf, .true., grad, .true.)
    if (.not. moperr) call mopac_save(state)
    keep_fock = .true.
    call mopac_finalize(properties)
    keep_fock = .false.
  end subroutine mopac_scf

  ! MOPAC geometry relaxation
  module subroutine mopac_relax(system, state, properties) bind(c)
#ifdef WIN32
!dec$ attributes dllexport :: mopac_relax
#endif
    type(mopac_system), intent(in) :: system
    type(mopac_state), intent(inout) :: state
    type(mopac_properties), intent(out) :: properties

    keywrd = " LBFGS PULAY BONDS"
    call mopac_initialize(system)
    if (.not. moperr) call mopac_load(state)
    ! call computational routine for LBFGS geometry optimization
    if (.not. moperr) call lbfgs (xparam, escf)
    if (.not. moperr) call mopac_save(state)
    call mopac_finalize(properties)
  end subroutine mopac_relax

  ! MOPAC vibrational calculation
  module subroutine mopac_vibe(system, state, properties) bind(c)
#ifdef WIN32
!dec$ attributes dllexport :: mopac_vibe
#endif
    type(mopac_system), intent(in) :: system
    type(mopac_state), intent(inout) :: state
    type(mopac_properties), intent(out) :: properties
    integer :: i

    keywrd = " FORCETS PULAY BONDS"
    call mopac_initialize(system)
    if (.not. moperr) call mopac_load(state)
    ! call computational routine to evaluate vibrational matrix
    if (.not. moperr) call force()
    ! recompute nvar because it is zero'd after vibrational calculations
    if (.not. moperr) then
      nvar = 0
      do i=1, system%natom+system%nlattice
        if (lopt(1,i) == 1) nvar = nvar + 3
      end do
    end if
    ! call computational routine for standard properties calculations
    if (.not. moperr) call compfg (xparam, .true., escf, .true., grad, .true.)
    if (.not. moperr) call mopac_save(state)
    call mopac_finalize(properties)
  end subroutine mopac_vibe

  ! MOZYME electronic ground state calculation
  module subroutine mozyme_scf(system, state, properties) bind(c)
#ifdef WIN32
!dec$ attributes dllexport :: mozyme_scf
#endif
    type(mopac_system), intent(in) :: system
    type(mozyme_state), intent(inout) :: state
    type(mopac_properties), intent(out) :: properties

    keywrd = " MOZYME 1SCF ALLBONDS"
    if (system%natom_move + system%nlattice_move > 0) keywrd = trim(keywrd) // " GRADIENTS"
    call mopac_initialize(system)
    if (.not. moperr) call mozyme_load(state)
    ! call computational routine for SCF calculations
    if (.not. moperr) call compfg (xparam, .true., escf, .true., grad, .true.)
    if (.not. moperr) call mozyme_save(state)
    call mopac_finalize(properties)
  end subroutine mozyme_scf

  ! MOZYME geometry relaxation
  module subroutine mozyme_relax(system, state, properties) bind(c)
#ifdef WIN32
!dec$ attributes dllexport :: mozyme_relax
#endif
    type(mopac_system), intent(in) :: system
    type(mozyme_state), intent(inout) :: state
    type(mopac_properties), intent(out) :: properties

    keywrd = " MOZYME LBFGS ALLBONDS"
    call mopac_initialize(system)
    if (.not. moperr) call mozyme_load(state)
    ! call computational routine for LBFGS geometry optimization
    if (.not. moperr) call lbfgs (xparam, escf)
    if (.not. moperr) call mozyme_save(state)
    call mopac_finalize(properties)
  end subroutine mozyme_relax

  ! MOZYME vibrational calculation
  module subroutine mozyme_vibe(system, state, properties) bind(c)
#ifdef WIN32
!dec$ attributes dllexport :: mozyme_vibe
#endif
    type(mopac_system), intent(in) :: system
    type(mozyme_state), intent(inout) :: state
    type(mopac_properties), intent(out) :: properties
    integer :: i

    keywrd = " MOZYME FORCETS PULAY BONDS"
    call mopac_initialize(system)
    if (.not. moperr) call mozyme_load(state)
    ! call computational routine to evaluate vibrational matrix
    if (.not. moperr) call force()
    ! recompute nvar because it is zero'd after vibrational calculations
    if (.not. moperr) then
      nvar = 0
      do i=1, system%natom+system%nlattice
        if (lopt(1,i) == 1) nvar = nvar + 3
      end do
    end if
    ! call computational routine for standard properties calculations
    if (.not. moperr) call compfg (xparam, .true., escf, .true., grad, .true.)
    if (.not. moperr) call mozyme_save(state)
    call mopac_finalize(properties)
  end subroutine mozyme_vibe

  ! Run MOPAC conventionally from an input file
  module function run_mopac_from_input(path_to_file) bind(c)
#ifdef WIN32
!dec$ attributes dllexport :: run_mopac_from_input
#endif
    interface
      subroutine run_mopac() bind(c)
      end subroutine run_mopac
    end interface
    integer(c_int) :: run_mopac_from_input
    character(kind=c_char), dimension(*), intent(in) :: path_to_file
    integer :: i
    i = 1
    do
      if(path_to_file(i) == ' ' .or. path_to_file(i) == c_null_char) exit
      jobnam(i:i) = path_to_file(i)
      i = i + 1
    end do
    run = 2
    call run_mopac
    if (moperr) then
      run_mopac_from_input = 1
    else
      run_mopac_from_input = 0
    end if
    run = 1
    jobnam = ' '
  end function run_mopac_from_input

  ! MOZYME QM/MM session: set up the QM system once (keywords as mozyme_scf, plus HB-SMOOTH, and the MOZYME
  ! GPU defaults that run_mopac applies), keep the electronic state between steps
  module subroutine mozyme_qmmm_init(system, state, status) bind(c)
#ifdef WIN32
!dec$ attributes dllexport :: mozyme_qmmm_init
#endif
#ifdef GPU
    use mod_vars_cuda, only : lgpu, mozyme_gpu, mozyme_gpu_requested, mozyme_resident_fock_gpu, &
      mozyme_gpu_disable_reason, MOZYME_GPU_REASON_NONE
    use mozyme_gpu_scf_driver, only : mozyme_gpu_apply_default_environment
#endif
    use qmmm_C, only : n_mm
    type(mopac_system), intent(in) :: system
    type(mozyme_state), intent(in) :: state
    integer(c_int), intent(out) :: status
#ifdef GPU
    logical :: applied
    character(len=16) :: env
    integer :: env_len, env_stat
#endif
    keywrd = " MOZYME 1SCF ALLBONDS GRADIENTS HB-SMOOTH"
    ! a tight SCF (MD forces) needs smaller LMO cut-offs: without them the SCF stops at its noise floor
    if (system%tolerance < 0.1d0) keywrd = trim(keywrd) // " THRESH=1.D-15"
    n_mm = 0
#ifdef GPU
    ! api_setup_gpu (mopac_initialize) uses a GPU only when lgpu is set beforehand (or MOPAC_FORCEGPU):
    ! the QM/MM session wants it unless MOPAC_NOGPU is set (without this the whole SCF ran on the CPU)
    env = ' '
    call get_environment_variable('MOPAC_NOGPU', env, length=env_len, status=env_stat)
    lgpu = .not. (env_stat == 0 .and. env_len > 0 .and. index('0 F FALSE N NO OFF', trim(adjustl(env))) == 0)
    ! as run_mopac: small QM regions are faster on the CPU (Tyr side chain, 16 atoms: 0.014 s per step on
    ! the CPU, 0.026 s on a T4); MOPAC_FORCEGPU still forces the GPU
    if (system%natom <= 100) lgpu = .false.
#endif
    call mopac_initialize(system)
#ifdef GPU
    if (.not. moperr .and. lgpu) then
      call mozyme_gpu_apply_default_environment(7, applied)
      mozyme_gpu_requested = .true.
      mozyme_gpu = .true.
      mozyme_resident_fock_gpu = .true.
      mozyme_gpu_disable_reason = MOZYME_GPU_REASON_NONE
    end if
#endif
    if (.not. moperr) call mozyme_load(state)
    status = 0
    if (moperr) status = 1
  end subroutine mozyme_qmmm_init

  ! One QM/MM evaluation: QM coordinates (Angstroms), MM point charges (Angstroms, electrons), switching
  ! radii (mm_roff <= 0: none); returns the heat of formation (kcal/mol, including the QM-MM
  ! electrostatics) and its gradient on the QM atoms and on the MM charges (kcal/(mol Angstrom))
  module subroutine mozyme_qmmm_step(natom, coord, nmm, mm_coord, mm_charge, mm_ron, mm_roff, &
      heat, grad_out, mm_grad_out, mm_pot_out, status) bind(c)
#ifdef WIN32
!dec$ attributes dllexport :: mozyme_qmmm_step
#endif
    use qmmm_C, only : n_mm, mm_xyz, mm_q, mm_grad, mm_pot, ron => mm_ron, roff => mm_roff
    use molkst_C, only : iscf, emin
    integer(c_int), value, intent(in) :: natom, nmm
    real(c_double), intent(in) :: coord(*), mm_coord(*), mm_charge(*)
    real(c_double), value, intent(in) :: mm_ron, mm_roff
    real(c_double), intent(out) :: heat, grad_out(*), mm_grad_out(*), mm_pot_out(*)
    integer(c_int), intent(out) :: status
    integer :: n
    status = 1
    if (moperr .or. natom*3 /= nvar) return
    xparam(:nvar) = coord(:nvar)
    if (allocated(mm_xyz)) then
      if (size(mm_q) /= nmm) deallocate (mm_xyz, mm_q, mm_grad)
    end if
    if (.not. allocated(mm_xyz)) allocate (mm_xyz(3, max(nmm, 1)), mm_q(max(nmm, 1)), mm_grad(3, max(nmm, 1)))
    n = max(nmm, 0)
    n_mm = n
    if (n > 0) then
      mm_xyz(:, :n) = reshape(mm_coord(:3*n), [3, n])
      mm_q(:n) = mm_charge(:n)
    end if
    ron = mm_ron
    roff = mm_roff
!   Every step needs the SCF converged to SELCON: the rapid exit of the SCF when the heat is far from
!   the lowest heat so far (EMIN, meant for geometry optimization) gave gradients that were not the
!   derivatives of the energy, and the total energy of NVE dynamics drifted.
    emin = 0.d0
    call compfg (xparam, .true., escf, .true., grad, .true.)
    if (moperr) return
    heat = escf
    status = 0
    if (iscf == 2) status = 2          ! SCF not converged (results returned all the same)
    grad_out(:nvar) = grad(:nvar)
    if (n > 0) then
      mm_grad_out(:3*n) = reshape(mm_grad(:, :n), [3*n])
      mm_pot_out(:n) = mm_pot(:n)
    end if
  end subroutine mozyme_qmmm_step

  ! End of a QM/MM session: the electronic state and the properties of the last step
  module subroutine mozyme_qmmm_finalize(state, properties) bind(c)
#ifdef WIN32
!dec$ attributes dllexport :: mozyme_qmmm_finalize
#endif
    use qmmm_C, only : n_mm
    type(mozyme_state), intent(inout) :: state
    type(mopac_properties), intent(out) :: properties
    if (.not. moperr) call mozyme_save(state)
    call mopac_finalize(properties)
    n_mm = 0
  end subroutine mozyme_qmmm_finalize

  ! Get MOPAC version string
  module subroutine get_mopac_version(version) bind(c)
#ifdef WIN32
!dec$ attributes dllexport :: get_mopac_version
#endif
    character(kind=c_char), dimension(*), intent(out) :: version
    integer :: i
    i = 1
    do
      if(verson(i:i) == ' ') exit
      version(i) = verson(i:i)
      i = i + 1
    end do
    version(i) = c_null_char
  end subroutine get_mopac_version

end submodule mopac_api_operations
