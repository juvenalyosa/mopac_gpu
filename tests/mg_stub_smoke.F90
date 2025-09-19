program mg_stub_smoke
#ifdef GPU
  use gpu_eig_mg_interfaces
#endif
  implicit none
  integer, parameter :: n=16
  real*8 :: A(n,n), W(n)
  integer :: i, j, info
  do j=1,n
    do i=1,n
      if (i==j) then
        A(i,j) = dble(i)
      else
        A(i,j) = 0.d0
      end if
    end do
  end do
#ifdef GPU
  call mopac_cusolvermg_dsyevd(n, A, n, W, info)
  if (info == 0) then
    print *, 'mg_stub_smoke: MG unexpectedly succeeded (ok)'
  else
    print *, 'mg_stub_smoke: stub path exercised (info=', info, ')'
  end if
#else
  print *, 'mg_stub_smoke: GPU not enabled'
#endif
end program mg_stub_smoke

