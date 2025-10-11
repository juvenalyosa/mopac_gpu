! Developed by Dr. Juvenal Yosa Reyes, UMCG Groningen, Universidad Simon Bolivar - Barranquilla - Colombia
module gpu_small_solve_interfaces
  use iso_c_binding
  implicit none
  interface
    subroutine mopac_cuda_solve_linear(n, a, lda, b, info) bind(C,name='mopac_cuda_solve_linear')
      import :: c_int, c_double
      integer(c_int), value :: n, lda
      real(c_double)        :: a(lda, n)
      real(c_double)        :: b(n)
      integer(c_int)        :: info
    end subroutine mopac_cuda_solve_linear
  end interface
end module gpu_small_solve_interfaces
