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

module hbond_neighbours_C
!
!  Short-range neighbour lists for the hydrogen-bond corrections (PM6-DH+, PM7, ...).
!
!  The original H-bond machinery scans all "numat" atoms for every candidate
!  pair (setup_DH_Plus) and for every acceptor (find_XH_bonds), which is
!  O(nrpairs*numat): ~1.4 s per call for a 6700-atom protein.  This module
!  builds, once per call, a cell-grid list of every atom within "hb_nbr_rcut"
!  of each atom.  Lists are sorted by ascending atom number so that callers
!  iterating over them visit atoms in exactly the same order as the original
!  1..numat loops, which keeps the results bit-identical.
!
!  Only used for isolated molecules (id == 0); periodic systems keep the
!  original loops (distance() handles the translations there).
!
  implicit none
  private
  public :: hb_nbr_build, hb_nbr_clear, hb_nbr_ready, hb_nbr_rcut, hb_nbr_start, hb_nbr_list
  logical :: hb_nbr_ready = .false.
  double precision :: hb_nbr_rcut = 0.d0
  integer :: hb_nbr_numat = 0
  integer, allocatable :: hb_nbr_start(:), hb_nbr_list(:)
contains

  subroutine hb_nbr_clear()
    implicit none
    hb_nbr_ready = .false.
    hb_nbr_numat = 0
    hb_nbr_rcut = 0.d0
    if (allocated(hb_nbr_start)) deallocate(hb_nbr_start)
    if (allocated(hb_nbr_list)) deallocate(hb_nbr_list)
  end subroutine hb_nbr_clear

  subroutine hb_nbr_build(numat, coord, rcut)
!
!  Build the neighbour lists: for every atom i, hb_nbr_list(hb_nbr_start(i):hb_nbr_start(i+1)-1)
!  holds all j /= i with |r_i - r_j| < rcut, in ascending order of j.
!  On exit hb_nbr_ready is .false. if the lists could not be built (degenerate box);
!  callers must then fall back to the full scans.
!
    implicit none
    integer, intent(in) :: numat
    double precision, intent(in) :: coord(3, *), rcut
    integer :: i, j, k, m, n, ix, iy, iz, jx, jy, jz, nx, ny, nz, ncell, ic, jc, total, cap, ntmp, stat
    integer, allocatable :: cell_of(:), cell_count(:), cell_start(:), cell_atoms(:), counts(:), tmp(:)
    double precision :: lo(3), hi(3), inv, r2, d(3), rcut2
    integer, parameter :: max_cells_per_atom = 16
    call hb_nbr_clear()
    if (numat <= 0 .or. rcut <= 0.d0) return
    lo = coord(1:3, 1)
    hi = coord(1:3, 1)
    do i = 2, numat
      lo = min(lo, coord(1:3, i))
      hi = max(hi, coord(1:3, i))
    end do
    inv = 1.d0/rcut
    nx = int((hi(1) - lo(1))*inv) + 1
    ny = int((hi(2) - lo(2))*inv) + 1
    nz = int((hi(3) - lo(3))*inv) + 1
    if (dble(nx)*dble(ny)*dble(nz) > dble(max_cells_per_atom)*dble(numat) + 1000.d0) return
    ncell = nx*ny*nz
    allocate (cell_of(numat), cell_count(ncell), cell_start(ncell + 1), cell_atoms(numat), &
      counts(numat), hb_nbr_start(numat + 1), stat=stat)
    if (stat /= 0) then
      call hb_nbr_clear()
      return
    end if
    cell_count = 0
    do i = 1, numat
      ix = min(nx - 1, int((coord(1, i) - lo(1))*inv))
      iy = min(ny - 1, int((coord(2, i) - lo(2))*inv))
      iz = min(nz - 1, int((coord(3, i) - lo(3))*inv))
      ic = 1 + ix + nx*(iy + ny*iz)
      cell_of(i) = ic
      cell_count(ic) = cell_count(ic) + 1
    end do
    cell_start(1) = 1
    do ic = 1, ncell
      cell_start(ic + 1) = cell_start(ic) + cell_count(ic)
    end do
    cell_count = 0
    do i = 1, numat  ! ascending i => each cell's member list is ascending
      ic = cell_of(i)
      cell_atoms(cell_start(ic) + cell_count(ic)) = i
      cell_count(ic) = cell_count(ic) + 1
    end do
    rcut2 = rcut*rcut
!
!  Pass 1: count neighbours per atom
!
    do i = 1, numat
      n = 0
      ic = cell_of(i) - 1
      ix = mod(ic, nx)
      iy = mod(ic/nx, ny)
      iz = ic/(nx*ny)
      do jz = max(0, iz - 1), min(nz - 1, iz + 1)
        do jy = max(0, iy - 1), min(ny - 1, iy + 1)
          do jx = max(0, ix - 1), min(nx - 1, ix + 1)
            jc = 1 + jx + nx*(jy + ny*jz)
            do m = cell_start(jc), cell_start(jc + 1) - 1
              j = cell_atoms(m)
              if (j == i) cycle
              d = coord(1:3, i) - coord(1:3, j)
              r2 = d(1)**2 + d(2)**2 + d(3)**2
              if (r2 < rcut2) n = n + 1
            end do
          end do
        end do
      end do
      counts(i) = n
    end do
    hb_nbr_start(1) = 1
    do i = 1, numat
      hb_nbr_start(i + 1) = hb_nbr_start(i) + counts(i)
    end do
    total = hb_nbr_start(numat + 1) - 1
    allocate (hb_nbr_list(max(1, total)), stat=stat)
    if (stat /= 0) then
      call hb_nbr_clear()
      return
    end if
!
!  Pass 2: fill, then sort each atom's list by ascending atom number
!
    cap = 0
    do i = 1, numat
      cap = max(cap, counts(i))
    end do
    allocate (tmp(max(1, cap)))
    do i = 1, numat
      ntmp = 0
      ic = cell_of(i) - 1
      ix = mod(ic, nx)
      iy = mod(ic/nx, ny)
      iz = ic/(nx*ny)
      do jz = max(0, iz - 1), min(nz - 1, iz + 1)
        do jy = max(0, iy - 1), min(ny - 1, iy + 1)
          do jx = max(0, ix - 1), min(nx - 1, ix + 1)
            jc = 1 + jx + nx*(jy + ny*jz)
            do m = cell_start(jc), cell_start(jc + 1) - 1
              j = cell_atoms(m)
              if (j == i) cycle
              d = coord(1:3, i) - coord(1:3, j)
              r2 = d(1)**2 + d(2)**2 + d(3)**2
              if (r2 < rcut2) then
                ntmp = ntmp + 1
                tmp(ntmp) = j
              end if
            end do
          end do
        end do
      end do
      ! insertion sort (lists are short)
      do k = 2, ntmp
        j = tmp(k)
        m = k - 1
        do while (m >= 1)
          if (tmp(m) <= j) exit
          tmp(m + 1) = tmp(m)
          m = m - 1
        end do
        tmp(m + 1) = j
      end do
      do k = 1, ntmp
        hb_nbr_list(hb_nbr_start(i) + k - 1) = tmp(k)
      end do
    end do
    deallocate (tmp, cell_of, cell_count, cell_start, cell_atoms, counts)
    hb_nbr_numat = numat
    hb_nbr_rcut = rcut
    hb_nbr_ready = .true.
  end subroutine hb_nbr_build
end module hbond_neighbours_C
