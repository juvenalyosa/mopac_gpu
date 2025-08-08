! CPU fallback implementation for rot_cuda interfaces used by GPU path
module call_rot_cuda
  use iso_c_binding, only: c_int, c_double, c_char
  implicit none
  private
  public :: rot_cuda, rot_cuda_2gpu
#ifdef GPU
  interface
    subroutine call_rot_cuda_gpu(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny) &
      bind(C, name="call_rot_cuda_gpu")
      implicit none
      integer(c_int), value :: nocc, lumo, n
      real(c_double), intent(in)    :: fmo(*)
      real(c_double), intent(in)    :: eig(n)
      real(c_double), intent(inout) :: vector(n,n)
      real(c_double), intent(in)    :: ci0(n,nocc)
      real(c_double), intent(in)    :: ca0(n,*)
      real(c_double), value         :: bigeps, tiny
    end subroutine call_rot_cuda_gpu
    subroutine call_rot_cuda_2gpu_gpu(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny) &
      bind(C, name="call_rot_cuda_2gpu_gpu")
      implicit none
      integer(c_int), value :: nocc, lumo, n
      real(c_double), intent(in)    :: fmo(*)
      real(c_double), intent(in)    :: eig(n)
      real(c_double), intent(inout) :: vector(n,n)
      real(c_double), intent(in)    :: ci0(n,nocc)
      real(c_double), intent(in)    :: ca0(n,*)
      real(c_double), value         :: bigeps, tiny
    end subroutine call_rot_cuda_2gpu_gpu
  end interface
#endif
contains

  subroutine rot_cuda(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny)
    implicit none
    integer, intent(in) :: nocc, lumo, n
    double precision, intent(inout) :: vector(n,n)
    double precision, intent(in)    :: eig(n)
    double precision, intent(in)    :: fmo(:)
    double precision, intent(in)    :: ci0(n,nocc), ca0(n,*)
    double precision, intent(in)    :: bigeps, tiny
#ifdef GPU
    call call_rot_cuda_gpu(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny)
#else
    ! CPU fallback
    integer :: i, j, ij
    double precision :: x, a, b, d, e, alpha, beta
    ij = 0
    do i = 1, nocc
      do j = lumo, n
        ij = ij + 1
        x = fmo(ij)
        if (dabs(x) < tiny) cycle
        a = eig(i)
        b = eig(j)
        d = a - b
        if (dabs(x/d) < bigeps) cycle
        e = sign(dsqrt(4.0d0*x*x + d*d), d)
        alpha = dsqrt(0.5d0*(1.0d0 + d/e))
        beta  = -sign(dsqrt(1.0d0 - alpha*alpha), x)
        call drot(n, vector(1:n,i), 1, vector(1:n,j), 1, alpha, beta)
      end do
    end do
#endif
  end subroutine rot_cuda

  subroutine rot_cuda_2gpu(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny)
    implicit none
    integer, intent(in) :: nocc, lumo, n
    double precision, intent(inout) :: vector(n,n)
    double precision, intent(in)    :: eig(n)
    double precision, intent(in)    :: fmo(:)
    double precision, intent(in)    :: ci0(n,nocc), ca0(n,*)
    double precision, intent(in)    :: bigeps, tiny
#ifdef GPU
    call call_rot_cuda_2gpu_gpu(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny)
#else
    call rot_cuda(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny)
#endif
  end subroutine rot_cuda_2gpu

end module call_rot_cuda
