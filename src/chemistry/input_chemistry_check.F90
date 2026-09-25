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

!
!  Chemical and electronic sanity check of a (PDB-derived) input system.
!
!  Crystal structures often contain groups that are not complete molecules:
!  a sulfate on a crystallographic special position is deposited as half a
!  molecule (e.g. S, O1, O3 with occupancy 0.5; the other half is a symmetry
!  mate that is not in the file).  MOZYME builds a Lewis structure for such a
!  fragment anyway (an "SO2(2-)" with dangling oxygens), the initial guess is
!  tens of thousands of kcal/mol too high and the SCF becomes ill-conditioned
!  (1G6X: 112 iterations and run-dependent energies instead of ~30).
!
!  getpdb records occupancies and REMARK 375 special positions; geochk calls
!  input_chemistry_check after the Lewis structure is known.  Errors stop the
!  job unless LET is present; warnings are only printed.
!
module input_chemistry_check_C
  implicit none
  private
  public :: pdb_check_reset, pdb_check_record_atom, pdb_check_record_remark, &
    input_chemistry_check

  integer, parameter :: key_len = 9       ! txtatm(18:26): "SO4 A  64"
  integer, parameter :: max_special = 2000
  integer, save :: n_special = 0
  character(len=key_len), save :: special_keys(max_special)
  integer, save :: n_partial = 0
  integer, save :: max_partial_listed = 2000
  character(len=key_len), save, allocatable :: partial_keys(:)
  double precision, save, allocatable :: partial_occ(:)
  integer, save :: structural_numcal = -1   ! job in which each part last ran
  integer, save :: electronic_numcal = -1

  ! Heavy-atom templates of common crystallisation additives and ions.
  integer, parameter :: n_het = 17
  character(len=3), parameter :: het_name(n_het) = [character(len=3) :: &
    "SO4", "PO4", "NO3", "EDO", "GOL", "ACT", "FMT", "DMS", "MPD", "TRS", &
    "CL ", "NA ", "K  ", "MG ", "CA ", "ZN ", "HOH"]
  integer, parameter :: het_heavy(n_het) = [5, 5, 4, 4, 6, 4, 3, 4, 8, 8, &
    1, 1, 1, 1, 1, 1, 1]
  ! Standard amino acids: heavy atoms of the residue inside a chain (no OXT).
  integer, parameter :: n_aa = 20
  character(len=3), parameter :: aa_name(n_aa) = [character(len=3) :: &
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", &
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
  integer, parameter :: aa_heavy(n_aa) = [5, 11, 8, 8, 6, 9, 9, 4, 10, 8, &
    8, 9, 8, 11, 7, 6, 7, 14, 12, 7]

contains

  subroutine pdb_check_reset()
    n_special = 0
    n_partial = 0
    structural_numcal = -1
    electronic_numcal = -1
  end subroutine pdb_check_reset

  ! One accepted ATOM/HETATM record (columns 18-26 = residue, 55-60 = occupancy).
  subroutine pdb_check_record_atom(line)
    character(len=*), intent(in) :: line
    double precision :: occ
    integer :: ios, i
    character(len=key_len) :: key
    if (len(line) < 60) return
    if (line(55:60) == " ") return
    read (line(55:60), *, iostat=ios) occ
    if (ios /= 0) return
    if (occ >= 0.999d0 .or. occ <= 0.d0) return
    key = line(18:26)
    do i = 1, n_partial
      if (partial_keys(i) == key) then
        partial_occ(i) = min(partial_occ(i), occ)
        return
      end if
    end do
    if (.not. allocated(partial_keys)) then
      allocate (partial_keys(max_partial_listed), partial_occ(max_partial_listed))
    end if
    if (n_partial >= max_partial_listed) return
    n_partial = n_partial + 1
    partial_keys(n_partial) = key
    partial_occ(n_partial) = occ
  end subroutine pdb_check_record_atom

  ! "REMARK 375 S    SO4 A  64  LIES ON A SPECIAL POSITION."
  subroutine pdb_check_record_remark(line)
    character(len=*), intent(in) :: line
    character(len=16) :: tok(5)
    character(len=key_len) :: key
    character(len=4) :: seq
    integer :: ios, i
    if (len(line) < 12) return
    if (line(1:10) /= "REMARK 375") return
    if (index(line, "SPECIAL POSITION") == 0) return
    tok = " "
    read (line(11:), *, iostat=ios) tok
    if (tok(2) == " ") return
    if (len_trim(tok(3)) > 1) then ! no chain letter: atom, residue, sequence
      seq = adjustr(tok(3)(1:4))
      key = tok(2)(1:3)//"  "//seq
    else                           ! atom, residue, chain, sequence
      seq = adjustr(tok(4)(1:4))
      key = tok(2)(1:3)//" "//tok(3)(1:1)//seq
    end if
    do i = 1, n_special
      if (special_keys(i) == key) return
    end do
    if (n_special >= max_special) return
    n_special = n_special + 1
    special_keys(n_special) = key
  end subroutine pdb_check_record_remark

  logical function on_special_position(key)
    character(len=*), intent(in) :: key
    integer :: i
    on_special_position = .false.
    do i = 1, n_special
      if (special_keys(i) == key) then
        on_special_position = .true.
        return
      end if
    end do
  end function on_special_position

  !
  !  ions(1:numat): formal charges of the Lewis structure (before geochk hides
  !  the sulfate/phosphate charges).  electronic = .false. while ADD-H/SITE is
  !  still changing the hydrogens: only the structural checks are meaningful.
  !
  subroutine input_chemistry_check(ions, electronic)
    use molkst_C, only: numat, keywrd, numcal
    use common_arrays_C, only: nat, txtatm, nbonds, ibonds
    use chanel_C, only: iw
    implicit none
    integer, intent(in) :: ions(*)
    logical, intent(in) :: electronic
    integer :: i, j, k, l, nres, nerr, nwarn, heavy, q, n_ionizable, &
      n_charged, n_acid_so4
    integer, allocatable :: res_of(:), res_heavy(:), res_q(:), table(:), &
      res_h_on_o(:)
    character(len=key_len), allocatable :: res_key(:)
    integer :: tsize, h, expected
    character(len=3) :: rname
    logical :: special, standard_aa, have_labels, do_structural, do_electronic

    ! Each part runs once per job: the structural part at the first call
    ! (also in ADD-H and non-MOZYME runs), the electronic part at the first
    ! call made with a Lewis structure.
    do_structural = (structural_numcal /= numcal)
    do_electronic = electronic .and. (electronic_numcal /= numcal)
    if (.not. (do_structural .or. do_electronic)) return
    if (do_structural) structural_numcal = numcal
    if (do_electronic) electronic_numcal = numcal
    if (numat <= 0) return
    have_labels = .false.
    do i = 1, numat
      if (txtatm(i)(18:26) /= " ") then
        have_labels = .true.
        exit
      end if
    end do
    if (.not. have_labels .and. n_special == 0 .and. n_partial == 0) return
    !
    !  Group atoms into residues by txtatm(18:26) (open-addressing hash).
    !
    tsize = 2*numat + 1
    allocate (res_of(numat), res_heavy(numat), res_q(numat), &
      res_h_on_o(numat), res_key(numat), table(tsize))
    table = 0
    nres = 0
    res_heavy = 0
    res_q = 0
    res_h_on_o = 0
    do i = 1, numat
      h = 0
      do j = 18, 26
        h = mod(h*31 + ichar(txtatm(i)(j:j)), tsize)
      end do
      do
        k = table(h + 1)
        if (k == 0) then
          nres = nres + 1
          res_key(nres) = txtatm(i)(18:26)
          table(h + 1) = nres
          k = nres
          exit
        end if
        if (res_key(k) == txtatm(i)(18:26)) exit
        h = mod(h + 1, tsize)
      end do
      res_of(i) = k
      if (nat(i) /= 1) res_heavy(k) = res_heavy(k) + 1
      res_q(k) = res_q(k) + ions(i)
      if (nat(i) == 1 .and. nbonds(i) > 0) then
        l = ibonds(1, i)
        if (nat(l) == 8) res_h_on_o(k) = res_h_on_o(k) + 1
      end if
    end do

    nerr = 0
    nwarn = 0
    write (iw, '(/10x,a)') "INPUT CHEMISTRY CHECK"
    !
    !  Structural checks: fragments and special positions.
    !
    if (do_structural) then
    do k = 1, nres
      if (res_key(k) == " ") cycle
      rname = res_key(k)(1:3)
      heavy = res_heavy(k)
      special = on_special_position(res_key(k))
      expected = 0
      do j = 1, n_het
        if (rname == het_name(j)) expected = het_heavy(j)
      end do
      standard_aa = .false.
      do j = 1, n_aa
        if (rname == aa_name(j)) then
          standard_aa = .true.
          expected = aa_heavy(j)
        end if
      end do
      if (special .and. heavy > 1) then
        nerr = nerr + 1
        write (iw, '(10x,a,i0,a)') "ERROR   "//res_key(k)//": lies on a crystallographic special position (REMARK 375); ", &
          heavy, " heavy atoms are in the file,"
        write (iw, '(10x,a)') "        the rest of the group is a symmetry mate that is not.  Remove the group or complete it."
      else if (special) then
        nwarn = nwarn + 1
        write (iw, '(10x,a)') "WARNING "//res_key(k)//": single atom on a crystallographic special position (REMARK 375)."
      end if
      if (expected > 0 .and. heavy < expected .and. .not. special) then
        if (standard_aa) then
          nwarn = nwarn + 1
          write (iw, '(10x,a,i0,a,i0,a)') "WARNING "//res_key(k)//": ", heavy, " heavy atoms, a complete residue has ", &
            expected, " (missing side-chain atoms were capped with hydrogen)."
        else
          nerr = nerr + 1
          write (iw, '(10x,a,i0,a,i0,a)') "ERROR   "//res_key(k)//": incomplete group, ", heavy, &
            " heavy atoms instead of ", expected, ".  Remove the fragment or complete it."
        end if
      end if
    end do
    do k = 1, n_partial
      if (on_special_position(partial_keys(k))) cycle
      nwarn = nwarn + 1
      if (nwarn <= 50) write (iw, '(10x,a,f5.2,a)') "WARNING "//partial_keys(k)//": partial occupancy (", &
        partial_occ(k), ") in the PDB file; the model may hold only part of a disordered group."
    end do
    end if
    !
    !  Electronic checks on the final structure (Lewis formal charges).
    !
    if (do_electronic) then
      do i = 1, numat
        ! H and B..F only: hypervalent S/P legitimately carry +2/+1 in the
        ! MOZYME Lewis structure of sulfate/phosphate (single S-O(-) bonds).
        if (nat(i) /= 1 .and. (nat(i) < 5 .or. nat(i) > 9)) cycle
        if (abs(ions(i)) < 2) cycle
        nerr = nerr + 1
        write (iw, '(10x,a,i0,a,sp,i0,a)') "ERROR   atom ", i, " ("//trim(txtatm(i))//") has a formal charge of ", &
          ions(i), " in the Lewis structure: impossible for this element."
      end do
      n_ionizable = 0
      n_charged = 0
      n_acid_so4 = 0
      do k = 1, nres
        rname = res_key(k)(1:3)
        q = res_q(k)
        select case (rname)
        case ("ARG", "LYS", "ASP", "GLU", "HIS")
          n_ionizable = n_ionizable + 1
          if (q /= 0) n_charged = n_charged + 1
        case ("SO4", "PO4")
          if (res_h_on_o(k) > 0 .and. res_heavy(k) >= 5) n_acid_so4 = n_acid_so4 + 1
        end select
        if (abs(q) > 2) then
          nwarn = nwarn + 1
          write (iw, '(10x,a,sp,i0,a)') "WARNING "//res_key(k)//": net formal charge ", q, &
            " in the Lewis structure."
        end if
      end do
      if (n_ionizable >= 3 .and. n_charged == 0) then
        nwarn = nwarn + 1
        write (iw, '(10x,a,i0,a)') "WARNING all ", n_ionizable, &
          " ARG/LYS/ASP/GLU/HIS residues are neutral (gas-phase protonation);"
        write (iw, '(10x,a)') "        for a system at pH ~7 hydrogenate with ADD-H SITE=(IONIZE)."
      end if
      if (n_acid_so4 > 0) then
        nwarn = nwarn + 1
        write (iw, '(10x,a,i0,a)') "WARNING ", n_acid_so4, &
          " sulfate/phosphate groups carry O-H hydrogens (acid form); SITE=(SO4,PO4) ionizes them."
      end if
    end if
    if (nerr == 0 .and. nwarn == 0) then
      write (iw, '(10x,a)') "No problems found."
    else
      write (iw, '(10x,a,i0,a,i0,a)') "Errors: ", nerr, "   Warnings: ", nwarn, &
        "   (see HTTP://OpenMOPAC.net/Manual/setpi.html for Lewis structures)"
    end if
    deallocate (res_of, res_heavy, res_q, res_h_on_o, res_key, table)
    if (nerr > 0) then
      if (index(keywrd, " LET") /= 0) then
        write (iw, '(10x,a)') "Keyword LET present: continuing despite the errors above."
      else
        call mopend("INPUT CHEMISTRY CHECK FOUND ERRORS.  FIX THE INPUT, OR ADD KEYWORD ""LET"" TO CONTINUE ANYWAY.")
      end if
    end if
  end subroutine input_chemistry_check
end module input_chemistry_check_C
