! Simple GA-style initial guess harness to improve SCF convergence
module ga_harness
  implicit none
  private
  public :: ga_initial_guess, maybe_ga_refine_initial_guess
contains

  subroutine ga_initial_guess(pa, pb, p, w, h, f, fb, &
                              norbs, mpack, numat, nfirst, nlast, &
                              uhf, na1el, nb1el, nclose, fract, id, best_e)
    use chanel_C, only : iw
    implicit none
    integer, intent(in) :: norbs, mpack, numat, id
    integer, intent(in) :: nfirst(numat), nlast(numat)
    integer, intent(in) :: na1el, nb1el, nclose
    logical, intent(in) :: uhf
    double precision, intent(in) :: fract
    double precision, intent(inout) :: pa(mpack), pb(mpack), p(mpack)
    double precision, intent(inout) :: w(*), h(mpack), f(mpack), fb(mpack)
    double precision, intent(out) :: best_e

    integer :: pop, i, j, cand, status
    double precision :: base_pa_norm, base_pb_norm
    double precision, allocatable :: cand_pa(:), cand_pb(:)
    double precision :: e_cand, eea, eeb, w1, w2, rnd
    character(len=32) :: env

    best_e = 1.0d300
    pop = 6
    call get_environment_variable('MOPAC_GA_POP', env, status=status)
    if (status == 0) then
      read(env,*,err=10,end=10) pop
    end if
10  continue
    pop = max(2, min(16, pop))

    allocate(cand_pa(mpack), cand_pb(mpack))

    do cand = 1, pop
      ! Start from current diagonal guess encoded in pa/pb
      cand_pa(:) = pa(:)
      if (uhf) then
        cand_pb(:) = pb(:)
      else
        cand_pb(:) = 0.0d0
      end if

      ! Randomize spin weights slightly and add tiny diagonal jitter
      w1 = na1el/(na1el + 1.d-6 + nb1el)
      w2 = 1.d0 - w1
      if (.not. uhf) then
        w1 = 1.0d0
        w2 = 0.0d0
      end if
      call random_seed()
      do i = 1, norbs
        j = (i*(i+1))/2
        call random_number(rnd)
        rnd = (rnd - 0.5d0) * 0.02d0  ! +/- 1% jitter
        cand_pa(j) = cand_pa(j) * max(0.d0, min(2.d0, w1 * (1.d0 + rnd)))
        if (uhf) then
          call random_number(rnd)
          rnd = (rnd - 0.5d0) * 0.02d0
          cand_pb(j) = cand_pb(j) * max(0.d0, min(2.d0, w2 * (1.d0 + rnd)))
        end if
      end do

      ! Build Fock matrices for this candidate (minimal single pass)
      call dcopy(mpack, h, 1, f, 1)
      if (id /= 0) then
        call fock2 (f, cand_pa, cand_pa, w, w, w, numat, nfirst, nlast, 2)
      else
        call fock2 (f, cand_pa, cand_pa, w, w, w, numat, nfirst, nlast, 2)
      end if
      eea = helect_quick(norbs, cand_pa, h, f)
      if (uhf) then
        call dcopy(mpack, h, 1, fb, 1)
        if (id /= 0) then
          call fock2 (fb, cand_pa, cand_pb, w, w, w, numat, nfirst, nlast, 2)
        else
          call fock2 (fb, cand_pa, cand_pb, w, w, w, numat, nfirst, nlast, 2)
        end if
        eeb = helect_quick(norbs, cand_pb, h, fb)
      else
        eeb = 0.d0
      end if
      e_cand = eea + eeb

      if (e_cand < best_e) then
        best_e = e_cand
        pa(:) = cand_pa(:)
        if (uhf) pb(:) = cand_pb(:)
      end if
    end do

    ! Refresh total density p from pa/pb
    if (uhf) then
      p(:) = pa(:) + pb(:)
    else
      p(:) = pa(:) * 2.d0
    end if

    deallocate(cand_pa, cand_pb)
  end subroutine ga_initial_guess

  subroutine maybe_ga_refine_initial_guess(pa, pb, p, w, h, f, fb, norbs, mpack, numat, nfirst, nlast, &
                                          uhf, na1el, nb1el, nclose, fract, id)
    implicit none
    integer, intent(in) :: norbs, mpack, numat, id
    integer, intent(in) :: nfirst(numat), nlast(numat)
    integer, intent(in) :: na1el, nb1el, nclose
    logical, intent(in) :: uhf
    double precision, intent(in) :: fract
    double precision, intent(inout) :: pa(mpack), pb(mpack), p(mpack)
    double precision, intent(inout) :: w(*), h(mpack), f(mpack), fb(mpack)
    integer :: status
    character(len=8) :: env
    logical :: enabled
    double precision :: tmp_best
    enabled = .false.
    call get_environment_variable('MOPAC_GA_INIT', env, status=status)
    if (status == 0) then
      if (trim(adjustl(env)) /= '' .and. trim(adjustl(env)) /= '0' .and. trim(adjustl(env)) /= 'off') then
        enabled = .true.
      end if
    end if
    if (.not. enabled) return
    call ga_initial_guess(pa, pb, p, w, h, f, fb, norbs, mpack, numat, nfirst, nlast, &
                          uhf, na1el, nb1el, nclose, fract, id, tmp_best)
  end subroutine maybe_ga_refine_initial_guess

  double precision function helect_quick(norbs, p, h, f)
    implicit none
    integer, intent(in) :: norbs
    double precision, intent(in) :: p(*), h(*), f(*)
    integer :: i, ij
    helect_quick = 0.d0
    ij = 0
    do i = 1, norbs
      ij = ij + i
      helect_quick = helect_quick + p(ij)*(h(ij) + f(ij))
    end do
  end function helect_quick

end module ga_harness
