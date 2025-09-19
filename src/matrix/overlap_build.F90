module overlap_build
  use parameters_C, only : betas, betap, betad, natorb
  use molkst_C,    only : norbs, numat
  use common_arrays_C, only : nat, nfirst, h
  implicit none
contains

  subroutine build_overlap_packed(overlap)
    implicit none
    double precision, intent (out) :: overlap((norbs*(norbs + 1))/2)
    integer :: i, im1, j, ii, jj, ij, if, jf, ni, nj, norbi, norbj
    integer :: ifact(norbs + 1)
    double precision :: bi(9), bj(9)
    do i = 1, norbs
      ifact(i) = (i*(i - 1))/2
    end do
    ifact(norbs+1) = (norbs*(norbs + 1))/2
    overlap = 0.d0
    do i = 1, numat
      if = nfirst(i)
      im1 = i - 1
      ni = nat(i)
      bi(1) = betas(ni)*0.5D0
      bi(2) = betap(ni)*0.5D0
      bi(3) = bi(2)
      bi(4) = bi(2)
      bi(5) = betad(ni)*0.5D0
      bi(6) = bi(5)
      bi(7) = bi(5)
      bi(8) = bi(5)
      bi(9) = bi(5)
      norbi = natorb(ni)
      do j = 1, im1
        nj = nat(j)
        bj(1) = betas(nj)*0.5D0
        bj(2) = betap(nj)*0.5D0
        bj(3) = bj(2)
        bj(4) = bj(2)
        bj(5) = betad(nj)*0.5D0
        bj(6) = bj(5)
        bj(7) = bj(5)
        bj(8) = bj(5)
        bj(9) = bj(5)
        norbj = natorb(nj)
        jf = nfirst(j)
        do ii = 1, norbi
          do jj = 1, norbj
            ij = ((if + ii - 1)*(if + ii - 2))/2 + jf + jj - 1
            overlap(ij) = h(ij)/(bi(ii) + bj(jj))
          end do
        end do
      end do
    end do
    overlap(ifact(2:norbs+1)) = 1.D0
  end subroutine build_overlap_packed

  subroutine unpack_upper_to_full(n, up_packed, S)
    implicit none
    integer, intent(in) :: n
    double precision, intent(in)  :: up_packed((n*(n+1))/2)
    double precision, intent(out) :: S(n,n)
    integer :: i,j,ij,ifact
    S = 0.d0
    ij = 0
    do j = 1, n
      do i = 1, j
        ij = ij + 1
        S(i,j) = up_packed(ij)
        S(j,i) = up_packed(ij)
      end do
    end do
  end subroutine unpack_upper_to_full

end module overlap_build

