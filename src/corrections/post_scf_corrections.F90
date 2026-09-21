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


! Cache of the last post-SCF correction evaluation.  compfg evaluates the
! corrections for the energy and deriv evaluates them again, with gradients,
! at the same geometry: the second evaluation (dispersion + H-bond pair search,
! ~40 ms per geometry step at 7000 atoms) only repeats the first.  When compfg
! announces that a gradient will follow (post_scf_gradient_expected) the energy
! call also accumulates the gradient contribution, and the gradient call at the
! same geometry (exact coordinate match) adds the stored contribution instead.
module post_scf_cache_C
  implicit none
  logical :: post_scf_gradient_expected = .false.
  logical, private :: cache_valid = .false.
  logical, private :: cache_has_grad = .false.
  integer, private :: cache_numat = -1, cache_numcal = -1, cache_id = -1
  integer, private :: cache_nbonds_sum = -1
  double precision, private :: cache_correction = 0.d0
  double precision, private :: cache_e_disp = 0.d0, cache_e_hb = 0.d0, cache_e_hh = 0.d0
  integer, private :: cache_p_hbonds = 0
  double precision, allocatable, private :: cache_coord(:,:), cache_dxyz(:), dxyz_before(:)
contains
  integer function post_scf_nbonds_sum()
    use common_arrays_C, only: nbonds
    use molkst_C, only: numat
    post_scf_nbonds_sum = -1
    if (allocated(nbonds)) then
      if (size(nbonds) >= numat) post_scf_nbonds_sum = sum(nbonds(1:numat))
    end if
  end function post_scf_nbonds_sum

  ! True when the cache holds this geometry (and a gradient, when one is needed).
  logical function post_scf_cache_hit(need_grad)
    use common_arrays_C, only: coord
    use molkst_C, only: numat, numcal, id
    logical, intent(in) :: need_grad
    post_scf_cache_hit = .false.
    if (.not. cache_valid) return
    if (need_grad .and. .not. cache_has_grad) return
    if (cache_numat /= numat .or. cache_numcal /= numcal .or. cache_id /= id) return
    if (cache_nbonds_sum /= post_scf_nbonds_sum()) return
    if (.not. allocated(cache_coord)) return
    if (size(cache_coord, 2) < numat) return
    post_scf_cache_hit = all(cache_coord(1:3, 1:numat) == coord(1:3, 1:numat))
  end function post_scf_cache_hit

  subroutine post_scf_cache_restore(correction, apply_grad)
    use common_arrays_C, only: dxyz
    use molkst_C, only: numat, E_disp, E_hb, E_hh, P_Hbonds
    double precision, intent(out) :: correction
    logical, intent(in) :: apply_grad
    correction = cache_correction
    E_disp = cache_e_disp
    E_hb = cache_e_hb
    E_hh = cache_e_hh
    P_Hbonds = cache_p_hbonds
    if (apply_grad) dxyz(1:3*numat) = dxyz(1:3*numat) + cache_dxyz(1:3*numat)
  end subroutine post_scf_cache_restore

  ! Remember dxyz before an evaluation that accumulates the gradient.
  subroutine post_scf_cache_begin_grad()
    use common_arrays_C, only: dxyz
    use molkst_C, only: numat
    if (allocated(dxyz_before)) then
      if (size(dxyz_before) < 3*numat) deallocate(dxyz_before)
    end if
    if (.not. allocated(dxyz_before)) allocate(dxyz_before(3*numat))
    dxyz_before(1:3*numat) = dxyz(1:3*numat)
  end subroutine post_scf_cache_begin_grad

  subroutine post_scf_cache_store(correction, has_grad, restore_dxyz)
    use common_arrays_C, only: coord, dxyz
    use molkst_C, only: numat, numcal, id, E_disp, E_hb, E_hh, P_Hbonds
    double precision, intent(in) :: correction
    logical, intent(in) :: has_grad, restore_dxyz
    if (allocated(cache_coord)) then
      if (size(cache_coord, 2) < numat) deallocate(cache_coord)
    end if
    if (.not. allocated(cache_coord)) allocate(cache_coord(3, numat))
    cache_coord(1:3, 1:numat) = coord(1:3, 1:numat)
    cache_numat = numat
    cache_numcal = numcal
    cache_id = id
    cache_nbonds_sum = post_scf_nbonds_sum()
    cache_correction = correction
    cache_e_disp = E_disp
    cache_e_hb = E_hb
    cache_e_hh = E_hh
    cache_p_hbonds = P_Hbonds
    cache_has_grad = has_grad
    if (has_grad) then
      if (allocated(cache_dxyz)) then
        if (size(cache_dxyz) < 3*numat) deallocate(cache_dxyz)
      end if
      if (.not. allocated(cache_dxyz)) allocate(cache_dxyz(3*numat))
      cache_dxyz(1:3*numat) = dxyz(1:3*numat) - dxyz_before(1:3*numat)
      if (restore_dxyz) dxyz(1:3*numat) = dxyz_before(1:3*numat)
    end if
    cache_valid = .true.
  end subroutine post_scf_cache_store
end module post_scf_cache_C

subroutine post_scf_corrections(correction, l_grad)
!
!    Add dispersion, hydrogen bonding, extra H-H repulsion, and other energies to
!    improve intermolecular interaction geometries and energies.
!
  use molkst_C, only : keywrd, E_disp, E_hb, E_hh, method_pm7, P_Hbonds, &
    method_pm6_dh_plus, method_pm6_dh2, method_pm6_d3h4, method_pm6_dh2x, method_pm6_d3h4x, &
    method_pm6_d3, method_pm6_d3_not_h4, method_pm7_hh, method_pm7_minus, method_pm6_org, method_PM8
  use common_arrays_C, only: dxyz
  use molkst_C, only : numat, moperr
  use mozyme_section_timers, only : mozyme_section_timer_begin, mozyme_section_timer_end
  use post_scf_cache_C, only : post_scf_gradient_expected, post_scf_cache_hit, &
    post_scf_cache_restore, post_scf_cache_begin_grad, post_scf_cache_store
  implicit none
  double precision, intent(out) ::  correction
  logical, intent (in) :: l_grad
!
! Local variables
!
  logical ::  prt
  logical :: lg, use_cache, capture_grad
  double precision :: psc_timer
  double precision, external :: & !                  Original references
                          !
  energy_corr_hh_rep,   & ! Rezac J., Hobza P., "Advanced Corrections of Hydrogen Bonding and
                          ! Dispersion for Semiempirical Quantum Mechanical Methods", J. Chem.
                          ! Theory and Comp 8, 141-151 (2012).
                          !
                          !
  PM6_DH_Dispersion,    & ! S. Grimme, "Accurate description of van der Waals complexes by
                          ! density functional theory including empirical corrections."
                          ! J Comput Chem. 2004 Sep;25(12):1463-73.
                          !
  dftd3,                & ! Grimme S., Antony J., Ehrlich S., Krieg H., "A consistent and accurate
                          ! ab initio parametrization of density functional dispersion correction
                          ! (DFT-D) for the 94 elements H-Pu", J. Chem. Phys. 132, 154104:154104
                          ! (2010).
                          !
  disp_DnX,             & ! (DH2X) Rezac and Hobza's correction: "A halogen-bonding correction
                          ! for the semiempirical PM6 method" Chem. Phys. Lett. 506 286-289 (2011)
                          !
                          ! (D3H4X) Rezac J., Hobza P., "Advanced Corrections of Hydrogen Bonding
                          ! and Dispersion for Semiempirical Quantum Mechanical Methods", J. Chem.
                          ! Theory and Comp 8, 141-151 (2012)
                          !
  Hydrogen_bond_corrections !(DH2) Korth M., Pitonak M., Rezac J., Hobza P., "A Transferable
                          ! H-bonding Correction for Semiempirical Quantum-Chemical Methods",
                          ! J. Chem. Theory and Computation 6, 344-352 (2010).
                          !
                          ! (DH+) Korth M., "Third-Generation Hydrogen-Bonding Corrections for
                          ! Semiempirical QM Methods and Force Fields", J. Chem. Theory Comput.
                          ! 6, 3808-3816 (2010).
!
! PM6-D3H4X: Brahmkshatriya, P. S.; Dobes, P.; Fanfrlik, J.; Rezac, J.; Paruch, K.; Bronowska,
! A.; Lepsik, M.; Hobza, P. "Quantum Mechanical Scoring: Structural and Energetic Insights into
! Cyclin-Dependent Kinase 2 Inhibition by Pyrazolo[1,5-a]pyrimidines" Curr. Comput.-Aid. Drug.
! 2013 , 9 (1), 118�129.
!
  prt = (index(keywrd," 0SCF ") + index(keywrd," PRT ") /= 0 .and. index(keywrd," DISP") /= 0)
  use_cache = .not. prt .and. numat > 0
  if (use_cache) then
    if (post_scf_cache_hit(l_grad)) then
      call post_scf_cache_restore(correction, l_grad)
      return
    end if
  end if
  ! Evaluate the gradient now when compfg says deriv will ask for it at this
  ! geometry (dxyz must already have its full size).
  capture_grad = use_cache .and. (l_grad .or. post_scf_gradient_expected)
  if (capture_grad .and. .not. l_grad) then
    capture_grad = allocated(dxyz)
    if (capture_grad) capture_grad = size(dxyz) >= 3*numat
  end if
  lg = l_grad .or. capture_grad
  if (capture_grad) call post_scf_cache_begin_grad()
  correction = 0.d0
  E_hb       = 0.d0
  E_hh       = 0.d0
  E_disp     = 0.d0
  P_Hbonds   = 0
  if (.not. allocated(dxyz)) allocate (dxyz(1))
!
! All the hydrogen-bond corrections are in Hydrogen_bond_corrections
!
  if (method_pm6_d3h4x) then
    correction = correction + dftd3(lg, dxyz)
    correction = correction + Hydrogen_bond_corrections(lg, prt)
    correction = correction + energy_corr_hh_rep(lg, dxyz)
    correction = correction + disp_DnX(lg)
  else if (method_pm6_d3h4) then
    correction = correction + dftd3(lg, dxyz)
    correction = correction + Hydrogen_bond_corrections(lg, prt)
    correction = correction + energy_corr_hh_rep(lg, dxyz)
  else if (method_pm6_d3_not_h4) then
    correction = correction + dftd3(lg, dxyz)
    correction = correction + energy_corr_hh_rep(lg, dxyz)
  else if (method_pm6_d3) then
    correction = correction + dftd3(lg, dxyz)
  else if (method_pm6_dh_plus) then
    correction = correction + PM6_DH_Dispersion(lg)
    correction = correction + Hydrogen_bond_corrections(lg, prt)
  else if (method_pm6_dh2) then
!
! Hydrogen_bond_corrections uses partial atomic charges if method_pm6_dh2 .or. method_pm6_dh2x
!
    correction = correction + PM6_DH_Dispersion(lg)
    correction = correction + Hydrogen_bond_corrections(lg, prt)
  else if (method_pm6_dh2x) then
    correction = correction + PM6_DH_Dispersion(lg)
    correction = correction + Hydrogen_bond_corrections(lg, prt)
    correction = correction + disp_DnX(lg)
  else if (method_pm7_hh) then
    correction = correction + energy_corr_hh_rep(lg, dxyz)
    correction = correction + PM6_DH_Dispersion(lg)
    correction = correction + Hydrogen_bond_corrections(lg, prt)
  else if (method_pm7_minus) then
    return
  else if (method_pm6_org) then
    correction = correction + dftd3(lg, dxyz)
    correction = correction + Hydrogen_bond_corrections(lg, prt)
    correction = correction + energy_corr_hh_rep(lg, dxyz)
  else if (method_pm8) then
    correction = correction + dftd3(lg, dxyz)
    correction = correction + Hydrogen_bond_corrections(lg, prt)
    correction = correction + energy_corr_hh_rep(lg, dxyz)
  else if (method_pm7) then
    call mozyme_section_timer_begin('post_scf_dispersion', psc_timer)
    correction = correction + PM6_DH_Dispersion(lg)
    call mozyme_section_timer_end('post_scf_dispersion', psc_timer)
    call mozyme_section_timer_begin('post_scf_hbonds', psc_timer)
    correction = correction + Hydrogen_bond_corrections(lg, prt)
    call mozyme_section_timer_end('post_scf_hbonds', psc_timer)
  end if
  if (use_cache .and. .not. moperr) call post_scf_cache_store(correction, capture_grad, .not. l_grad)
  if (index(keywrd, " SILENT") == 0) then
    if (prt .and. P_Hbonds > 0) call print_post_scf_corrections
  end if
  return
end subroutine post_scf_corrections
