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

  subroutine hb_slot_orders(slots, ih, nbonds, norder, perm, w)
!
!  HB-SMOOTH (EH_plus_blend): the orders of the neighbours in slots(1:3) of an H-bond list entry
!  (farthest from the hydrogen ih first) and their weights.  Two distinct neighbours (nbonds >= 2):
!  both orders; three (nbonds >= 3, slots(3) a third atom): all six; otherwise the order as it is.
!
    implicit none
    integer, intent (in) :: slots(3), ih, nbonds
    integer, intent (out) :: norder, perm(3,6)
    double precision, intent (out) :: w(6)
    integer :: k, l, m, p(3,6)
    double precision :: d(3), sf(3,3), total
    double precision, external :: distance
    double precision, parameter :: width = 0.01d0
    data p /1,2,3, 1,3,2, 2,1,3, 2,3,1, 3,1,2, 3,2,1/
    norder = 1
    perm(:, 1) = slots
    w = 0.d0
    w(1) = 1.d0
    if (nbonds < 2 .or. slots(1) <= 0 .or. slots(2) <= 0 .or. slots(1) == slots(2)) return
    d(1) = distance(slots(1), ih)
    d(2) = distance(slots(2), ih)
    if (nbonds >= 3 .and. slots(3) > 0 .and. slots(3) /= slots(1) .and. slots(3) /= slots(2) &
      .and. slots(3) /= ih) then
      d(3) = distance(slots(3), ih)
      do l = 1, 3          !  sf(l,m): weight of "l farther from the H than m"
        do m = 1, 3
          sf(l, m) = 0.5d0*(1.d0 + tanh((d(l) - d(m))/width))
        end do
      end do
      norder = 6
      total = 0.d0
      do k = 1, 6
        perm(:, k) = slots(p(:, k))
        w(k) = sf(p(1, k), p(2, k))*sf(p(2, k), p(3, k))*sf(p(1, k), p(3, k))
        total = total + w(k)
      end do
      w = w/total
    else
      norder = 2
      perm(:, 2) = (/slots(2), slots(1), slots(3)/)
      w(1) = 0.5d0*(1.d0 + tanh((d(1) - d(2))/width))
      w(2) = 1.d0 - w(1)
    end if
  end subroutine hb_slot_orders
