module gpu_fock_interfaces
  use iso_c_binding
  implicit none
  interface
    function mopac_cuda_fock2(norbs, mpack, numat, nfirst, nlast, ptot, p, w, nati, f) &
      bind(C,name='mopac_cuda_fock2') result(ok)
      use iso_c_binding
      integer(c_int), value :: norbs, mpack, numat
      integer(c_int)        :: nfirst(numat), nlast(numat)
      real(c_double)        :: ptot(mpack), p(mpack)
      real(c_double)        :: w(*)
      integer(c_int), value :: nati
      real(c_double)        :: f(mpack)
      logical(c_bool)       :: ok
    end function mopac_cuda_fock2

    function mopac_cuda_fock2_keep(norbs, mpack, numat, nfirst, nlast, ptot, p, w, nati) &
      bind(C,name='mopac_cuda_fock2_keep') result(ok)
      use iso_c_binding
      integer(c_int), value :: norbs, mpack, numat
      integer(c_int)        :: nfirst(numat), nlast(numat)
      real(c_double)        :: ptot(mpack), p(mpack)
      real(c_double)        :: w(*)
      integer(c_int), value :: nati
      logical(c_bool)       :: ok
    end function mopac_cuda_fock2_keep

    function mopac_cuda_fock2_scf(norbs, mpack, numat, nfirst, nlast, ptot, p, w, fout) &
      bind(C,name='mopac_cuda_fock2_scf') result(ok)
      use iso_c_binding
      integer(c_int), value :: norbs, mpack, numat
      integer(c_int)        :: nfirst(numat), nlast(numat)
      real(c_double)        :: ptot(mpack), p(mpack)
      real(c_double)        :: w(*)
      real(c_double)        :: fout(mpack)
      logical(c_bool)       :: ok
    end function mopac_cuda_fock2_scf

    function mopac_cuda_mozyme_fock1(iab, ilim, ptot, f, w) &
      bind(C,name='mopac_cuda_mozyme_fock1') result(code)
      use iso_c_binding
      integer(c_int), value :: iab, ilim
      real(c_double)        :: ptot(*), f(*), w(*)
      integer(c_int)        :: code
    end function mopac_cuda_mozyme_fock1

    function mopac_cuda_mozyme_fock2(iab, jba, diagonal, pii, pjj, pij, fii, fjj, fij, wj, wk) &
      bind(C,name='mopac_cuda_mozyme_fock2') result(code)
      use iso_c_binding
      integer(c_int), value :: iab, jba
      logical(c_bool), value :: diagonal
      real(c_double)        :: pii(*), pjj(*), pij(*), fii(*), fjj(*), fij(*), wj(*), wk(*)
      integer(c_int)        :: code
    end function mopac_cuda_mozyme_fock2

    function mopac_cuda_mozyme_dfock2(iab, jba, diagonal, pii, pjj, pij, dfii, dfjj, dfij, wj, wk) &
      bind(C,name='mopac_cuda_mozyme_dfock2') result(code)
      use iso_c_binding
      integer(c_int), value :: iab, jba
      logical(c_bool), value :: diagonal
      real(c_double)        :: pii(*), pjj(*), pij(*), dfii(*), dfjj(*), dfij(*), wj(*), wk(*)
      integer(c_int)        :: code
    end function mopac_cuda_mozyme_dfock2
  end interface
end module gpu_fock_interfaces
