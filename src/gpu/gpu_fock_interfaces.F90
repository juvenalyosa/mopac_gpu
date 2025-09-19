module gpu_fock_interfaces
  use iso_c_binding
  implicit none
  interface
    function mopac_cuda_fock2(norbs, mpack, numat, nfirst, nlast, ptot, p, w, nati, f) bind(C,name='mopac_cuda_fock2') result(ok)
      import :: c_int, c_bool, c_double
      integer(c_int), value :: norbs, mpack, numat
      integer(c_int)        :: nfirst(numat), nlast(numat)
      real(c_double)        :: ptot(mpack), p(mpack)
      real(c_double)        :: w(*)
      integer(c_int), value :: nati
      real(c_double)        :: f(mpack)
      logical(c_bool)       :: ok
    end function mopac_cuda_fock2

    function mopac_cuda_fock2_keep(norbs, mpack, numat, nfirst, nlast, ptot, p, w, nati) bind(C,name='mopac_cuda_fock2_keep') result(ok)
      import :: c_int, c_bool, c_double
      integer(c_int), value :: norbs, mpack, numat
      integer(c_int)        :: nfirst(numat), nlast(numat)
      real(c_double)        :: ptot(mpack), p(mpack)
      real(c_double)        :: w(*)
      integer(c_int), value :: nati
      logical(c_bool)       :: ok
    end function mopac_cuda_fock2_keep

    function mopac_cuda_fock2_scf(norbs, mpack, numat, nfirst, nlast, ptot, p, w, fout) bind(C,name='mopac_cuda_fock2_scf') result(ok)
      import :: c_int, c_bool, c_double
      integer(c_int), value :: norbs, mpack, numat
      integer(c_int)        :: nfirst(numat), nlast(numat)
      real(c_double)        :: ptot(mpack), p(mpack)
      real(c_double)        :: w(*)
      real(c_double)        :: fout(mpack)
      logical(c_bool)       :: ok
    end function mopac_cuda_fock2_scf
  end interface
end module gpu_fock_interfaces
