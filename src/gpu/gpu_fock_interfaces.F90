! Developed by Dr. Juvenal Yosa Reyes, UMCG Groningen, Universidad Simon Bolivar - Barranquilla - Colombia
module gpu_fock_interfaces
  use iso_c_binding
  implicit none
  public :: mopac_cuda_fock2, mopac_cuda_fock2_keep, mopac_cuda_fock2_scf
  public :: mopac_cuda_mozyme_fock1, mopac_cuda_mozyme_fock1_batch
  public :: mopac_cuda_mozyme_fock2_4x1_batch
  public :: mopac_cuda_mozyme_sparse_fock_setup, mopac_cuda_mozyme_sparse_fock_run
  public :: mopac_cuda_mozyme_fock2, mopac_cuda_mozyme_dfock2
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

    function mopac_cuda_fock2_scf(norbs, mpack, numat, nfirst, nlast, ptot, p, w, wj, wk, periodic, n2elec, fout) &
      bind(C,name='mopac_cuda_fock2_scf') result(ok)
      use iso_c_binding
      integer(c_int), value :: norbs, mpack, numat
      integer(c_int)        :: nfirst(numat), nlast(numat)
      real(c_double)        :: ptot(mpack), p(mpack)
      real(c_double)        :: w(*)
      real(c_double)        :: wj(*)
      real(c_double)        :: wk(*)
      integer(c_int), value :: periodic
      integer(c_int), value :: n2elec
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

    function mopac_cuda_mozyme_fock1_batch(ntasks, f_offsets, w_offsets, iabs, ilims, ptot, f, w) &
      bind(C,name='mopac_cuda_mozyme_fock1_batch') result(code)
      use iso_c_binding
      integer(c_int), value :: ntasks
      integer(c_int)        :: f_offsets(*), w_offsets(*), iabs(*), ilims(*)
      real(c_double)        :: ptot(*), f(*), w(*)
      integer(c_int)        :: code
    end function mopac_cuda_mozyme_fock1_batch

    function mopac_cuda_mozyme_fock2_4x1_batch(ntasks, heavy_offsets, light_offsets, cross_offsets, &
      wj_values, wk_values, ptot, f) bind(C,name='mopac_cuda_mozyme_fock2_4x1_batch') result(code)
      use iso_c_binding
      integer(c_int), value :: ntasks
      integer(c_int)        :: heavy_offsets(*), light_offsets(*), cross_offsets(*)
      real(c_double)        :: wj_values(*), wk_values(*), ptot(*), f(*)
      integer(c_int)        :: code
    end function mopac_cuda_mozyme_fock2_4x1_batch

    function mopac_cuda_mozyme_sparse_fock_setup(mpack, natoms, one_count, one_f_offsets, one_w_offsets, one_iabs, one_ilims, &
      one_w_values_count, one_w_values, pair_count, pair_iabs, pair_jbas, pair_i_offsets, pair_j_offsets, &
      pair_cross_offsets, pair_diag_flags, pair_w_offsets, pair_w_values_count, pair_wj_values, pair_wk_values, &
      pair4x1_count, pair4x1_heavy_offsets, pair4x1_light_offsets, pair4x1_cross_offsets, pair4x1_wj_values, &
      pair4x1_wk_values, point_count, point_iabs, point_jbas, point_i_atoms, point_j_atoms, point_i_offsets, &
      point_j_offsets, point_addr_flags, point_w_values) bind(C,name='mopac_cuda_mozyme_sparse_fock_setup') result(code)
      use iso_c_binding
      integer(c_int), value :: mpack, natoms, one_count, one_w_values_count, pair_count, pair_w_values_count, pair4x1_count
      integer(c_int), value :: point_count
      integer(c_int)        :: one_f_offsets(*), one_w_offsets(*), one_iabs(*), one_ilims(*)
      integer(c_int)        :: pair_iabs(*), pair_jbas(*), pair_i_offsets(*), pair_j_offsets(*)
      integer(c_int)        :: pair_cross_offsets(*), pair_diag_flags(*), pair_w_offsets(*)
      integer(c_int)        :: pair4x1_heavy_offsets(*), pair4x1_light_offsets(*), pair4x1_cross_offsets(*)
      integer(c_int)        :: point_iabs(*), point_jbas(*), point_i_atoms(*), point_j_atoms(*)
      integer(c_int)        :: point_i_offsets(*), point_j_offsets(*), point_addr_flags(*)
      real(c_double)        :: one_w_values(*), pair_wj_values(*), pair_wk_values(*)
      real(c_double)        :: pair4x1_wj_values(*), pair4x1_wk_values(*)
      real(c_double)        :: point_w_values(*)
      integer(c_int)        :: code
    end function mopac_cuda_mozyme_sparse_fock_setup

    function mopac_cuda_mozyme_sparse_fock_run(mpack, ptot, qe, f) &
      bind(C,name='mopac_cuda_mozyme_sparse_fock_run') result(code)
      use iso_c_binding
      integer(c_int), value :: mpack
      real(c_double)        :: ptot(*), qe(*), f(*)
      integer(c_int)        :: code
    end function mopac_cuda_mozyme_sparse_fock_run

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
