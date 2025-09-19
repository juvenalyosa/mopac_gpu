module gpu_fock_interfaces
  use iso_c_binding
  implicit none
  interface
    function mopac_cuda_fock2(norbs, mpack, numat, nfirst, nlast, ptot, p, f) bind(C,name='mopac_cuda_fock2') result(ok)
      import :: c_int, c_bool, c_double
      integer(c_int), value :: norbs, mpack, numat
      integer(c_int)        :: nfirst(numat), nlast(numat)
      real(c_double)        :: ptot(mpack), p(mpack)
      real(c_double)        :: f(mpack)
      logical(c_bool)       :: ok
    end function mopac_cuda_fock2
  end interface
end module gpu_fock_interfaces

