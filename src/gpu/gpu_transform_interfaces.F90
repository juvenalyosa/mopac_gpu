! Developed by Dr. Juvenal Yosa Reyes, UMCG Groningen, Universidad Simon Bolivar - Barranquilla - Colombia
module gpu_transform_interfaces
  use iso_c_binding
  implicit none
  interface
    subroutine mopac_cuda_fmulC(n, fpacked, c, ldc, w, ldw) bind(C,name='mopac_cuda_fmulC')
      import :: c_int, c_double
      integer(c_int), value :: n, ldc, ldw
      real(c_double)        :: fpacked(n*(n+1)/2)
      real(c_double)        :: c(ldc, n)
      real(c_double)        :: w(ldw, n)
    end subroutine mopac_cuda_fmulC

    subroutine mopac_cuda_fmulC_from_dev(n, c, ldc, w, ldw) bind(C,name='mopac_cuda_fmulC_from_dev')
      import :: c_int, c_double
      integer(c_int), value :: n, ldc, ldw
      real(c_double)        :: c(ldc, n)
      real(c_double)        :: w(ldw, n)
    end subroutine mopac_cuda_fmulC_from_dev
  end interface
end module gpu_transform_interfaces
