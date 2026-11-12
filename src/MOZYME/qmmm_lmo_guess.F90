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

subroutine qmmm_lmo_guess
!
!  Starting LMOs for the next step of a QM/MM session.  The SCF of a step started from the converged LMOs
!  of the last step, C(n), lags behind the moving geometry; this extrapolates the occupied space with a
!  polynomial of degree lmo_guess (at most) in the projectors P(n-k) onto the occupied spaces of the last
!  steps (the time steps are assumed equal), applied to the LMOs of the last step:
!
!     occupied   C' = a0 C(n) + Sum_k ak P(n-k) C(n)        (a = 2, -1;  3, -3, 1;  4, -6, 4, -1)
!     virtual    V' = V(n) - Sum_k ak P(n-k) V(n)
!
!  P(n) C(n) = C(n) and P(n) V(n) = 0, so a space that did not move gives C' = C(n), V' = V(n).  The
!  projectors do not depend on how the LMOs of a step are rotated among themselves, which the SCF never
!  changes: extrapolating the coefficients themselves amplified such rotations from step to step (3 points:
!  the SCF failed after 90 steps).  P(n-k) X is kept on the atoms of each LMO of X.  The result is
!  normalized and re-orthogonalized, and the next SCF builds the density and the Fock matrix from it
!  before its first rotation (rebuild_from_lmos).
!
    use qmmm_C, only : lmo_guess, lmo_hist, lmo_set
    use MOZYME_C, only : noccupied, nvirtual, ncf, nncf, ncocc, icocc, cocc, nce, nnce, ncvir, icvir, cvir, &
      iorbs, rebuild_from_lmos
    use molkst_C, only : norbs, numat
    use mozyme_gpu_scf_driver, only : mozyme_gpu_scf_note_host_lmos_modified
    use mozyme_section_timers, only : mozyme_section_timer_begin, mozyme_section_timer_end
    implicit none
    double precision, parameter :: coef(0:3, 3) = reshape([2.d0, -1.d0, 0.d0, 0.d0, &
                                                           3.d0, -3.d0, 1.d0, 0.d0, &
                                                           4.d0, -6.d0, 4.d0, -1.d0], [4, 3])
    type(lmo_set) :: last
    double precision, dimension(:), allocatable :: ws, new_occ, new_vir
    integer, dimension(:), allocatable :: at, mark, cand, a_start, a_lmo
    integer :: k, n_hist, order
    double precision :: timer
    call mozyme_section_timer_begin('qmmm_lmo_guess', timer)
    ! the converged occupied LMOs of the last step
    last%nocc = noccupied
    last%ncf = ncf(:noccupied)
    last%nncf = nncf(:noccupied)
    last%ncocc = ncocc(:noccupied)
    last%icocc = icocc
    last%cocc = cocc
    n_hist = 0
    do k = 1, size(lmo_hist)
      if (lmo_hist(k)%nocc /= noccupied) exit
      n_hist = k
    end do
    order = min(lmo_guess, n_hist)
    if (order > 0) then
      allocate (at(numat), a_start(numat + 1))
      at = -1
      new_occ = coef(0, order)*cocc
      new_vir = cvir
      do k = 1, order
        call incidence(lmo_hist(k))
        allocate (mark(lmo_hist(k)%nocc), cand(lmo_hist(k)%nocc))
        mark = 0
        call add_projected(noccupied, ncf, nncf, ncocc, icocc, cocc, new_occ, coef(k, order), lmo_hist(k))
        call add_projected(nvirtual, nce, nnce, ncvir, icvir, cvir, new_vir, -coef(k, order), lmo_hist(k))
        deallocate (mark, cand, a_lmo)
      end do
      cocc(:) = new_occ      ! in place: the GPU context may hold the addresses of these arrays
      cvir(:) = new_vir
    end if
    do k = size(lmo_hist), 2, -1
      call move_set(lmo_hist(k - 1), lmo_hist(k))
    end do
    call move_set(last, lmo_hist(1))
    if (order > 0) then
      allocate (ws(norbs))
      call normalize(noccupied, ncf, nncf, ncocc, icocc, cocc)
      call normalize(nvirtual, nce, nnce, ncvir, icvir, cvir)
      call reorth(ws)
      call normalize(noccupied, ncf, nncf, ncocc, icocc, cocc)
      call normalize(nvirtual, nce, nnce, ncvir, icvir, cvir)
      rebuild_from_lmos = .true.
      call mozyme_gpu_scf_note_host_lmos_modified()
    end if
    call mozyme_section_timer_end('qmmm_lmo_guess', timer)
  contains

    subroutine incidence(h)
      ! a_lmo(a_start(ia)+1 : a_start(ia+1)): the LMOs of h that contain atom ia
      type(lmo_set), intent(in) :: h
      integer :: j, kk, ia
      integer, allocatable :: fill(:)
      a_start = 0
      do j = 1, h%nocc
        do kk = h%nncf(j) + 1, h%nncf(j) + h%ncf(j)
          a_start(h%icocc(kk) + 1) = a_start(h%icocc(kk) + 1) + 1
        end do
      end do
      do ia = 1, numat
        a_start(ia + 1) = a_start(ia + 1) + a_start(ia)
      end do
      allocate (a_lmo(a_start(numat + 1)), fill(numat))
      fill = a_start(:numat)
      do j = 1, h%nocc
        do kk = h%nncf(j) + 1, h%nncf(j) + h%ncf(j)
          ia = h%icocc(kk)
          fill(ia) = fill(ia) + 1
          a_lmo(fill(ia)) = j
        end do
      end do
    end subroutine incidence

    subroutine add_projected(n, nc, nnc, ncv, ic, x, y, a, h)
      ! y = y + a P(h) x for every LMO of x (coefficients x, offsets ncv, atoms ic), kept on its own atoms;
      ! P(h) = Sum_j |h_j><h_j| over the occupied LMOs h_j of an earlier step
      integer, intent(in) :: n, nc(:), nnc(:), ncv(:), ic(:)
      double precision, intent(in) :: x(:), a
      double precision, intent(inout) :: y(:)
      type(lmo_set), intent(in) :: h
      integer :: i, kk, l, ia, j, jj, m, ncand, nl, lh
      double precision :: s
      do i = 1, n
        ! offsets of the coefficients of each atom of LMO i, and the LMOs of h that share an atom with it
        ncand = 0
        l = ncv(i)
        do kk = nnc(i) + 1, nnc(i) + nc(i)
          ia = ic(kk)
          at(ia) = l
          l = l + iorbs(ia)
          do m = a_start(ia) + 1, a_start(ia + 1)
            j = a_lmo(m)
            if (mark(j) /= i) then
              mark(j) = i
              ncand = ncand + 1
              cand(ncand) = j
            end if
          end do
        end do
        do m = 1, ncand
          j = cand(m)
          s = 0.d0
          lh = h%ncocc(j)
          do jj = h%nncf(j) + 1, h%nncf(j) + h%ncf(j)
            ia = h%icocc(jj)
            nl = iorbs(ia)
            if (at(ia) >= 0) s = s + dot_product(x(at(ia) + 1:at(ia) + nl), h%cocc(lh + 1:lh + nl))
            lh = lh + nl
          end do
          s = a*s
          lh = h%ncocc(j)
          do jj = h%nncf(j) + 1, h%nncf(j) + h%ncf(j)
            ia = h%icocc(jj)
            nl = iorbs(ia)
            if (at(ia) >= 0) y(at(ia) + 1:at(ia) + nl) = y(at(ia) + 1:at(ia) + nl) + s*h%cocc(lh + 1:lh + nl)
            lh = lh + nl
          end do
        end do
        do kk = nnc(i) + 1, nnc(i) + nc(i)
          at(ic(kk)) = -1
        end do
      end do
    end subroutine add_projected

    subroutine move_set(from, to)
      type(lmo_set), intent(inout) :: from, to
      to%nocc = from%nocc
      call move_alloc(from%ncf, to%ncf)
      call move_alloc(from%nncf, to%nncf)
      call move_alloc(from%ncocc, to%ncocc)
      call move_alloc(from%icocc, to%icocc)
      call move_alloc(from%cocc, to%cocc)
      from%nocc = -1
    end subroutine move_set

    subroutine normalize(n, nc, nnc, ncv, ic, c)
      integer, intent(in) :: n, nc(:), nnc(:), ncv(:), ic(:)
      double precision, intent(inout) :: c(:)
      integer :: i, kk, m
      double precision :: s
      do i = 1, n
        m = 0
        do kk = nnc(i) + 1, nnc(i) + nc(i)
          m = m + iorbs(ic(kk))
        end do
        s = sum(c(ncv(i) + 1:ncv(i) + m)**2)
        if (s > 0.d0) c(ncv(i) + 1:ncv(i) + m) = c(ncv(i) + 1:ncv(i) + m)/sqrt(s)
      end do
    end subroutine normalize
end subroutine qmmm_lmo_guess
