module mamult_cuda_i
  implicit none
contains
  subroutine mamult_gpu(a, b, c, ndim, mdim, ifact, beta, igrid, iblock, tt, flag)
    implicit none
    integer, intent(in) :: ndim, mdim, igrid, iblock, flag
    integer, intent(in) :: ifact(*)
    double precision, intent(in) :: a(mdim), b(mdim), beta
    double precision, intent(inout) :: c(mdim)
    double precision, intent(inout) :: tt
    ! This is a stub; should never be called in current code paths.
    stop 'mamult_gpu stub called unexpectedly'
  end subroutine mamult_gpu
end module mamult_cuda_i

