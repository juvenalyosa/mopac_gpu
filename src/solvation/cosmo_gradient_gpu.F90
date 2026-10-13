! Molecular Orbital PACkage (MOPAC)
! Copyright 2021 Virginia Polytechnic Institute and State University
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!    http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

subroutine diegrd_gpu (dxyz, done)
!
!  The O(nps**2) part of the COSMO gradient (diegrd: surface charge - surface charge, the SWIG
!  switching term and surface charge - atomic multipoles) on the GPU, src/gpu/cosmo_gradient.cu.
!  done = .false. (dxyz unchanged) when the device cannot do it; diegrd then uses the CPU.
!
    use molkst_C, only : numat
    use cosmo_C, only : nps, fepsi, cosurf, iatsp, qscnet, qdenet, srad, swig, swig_zeta, &
      swig_f
    use parameters_C, only : dd, qq
    use common_arrays_C, only : coord, nfirst, nlast, nat
    use funcon_C, only : ev, a0, fpc_9
    use mozyme_section_timers, only : mozyme_section_timer_begin, mozyme_section_timer_end
#ifdef GPU
    use iso_c_binding, only : c_int, c_double
#endif
    implicit none
    double precision, dimension (3, numat), intent (inout) :: dxyz
    logical, intent (out) :: done
#ifdef GPU
    interface
      function mopac_cuda_cosmo_gradient(numat_c, nps_c, swig_c, nden_c, coord_c, srad_c, &
          idel_c, iden_c, ddi_c, qqi2_c, pxyz_c, q_c, zeta_c, fsw_c, iatsp_c, qden_c, &
          fact_c, fepsi_c, dxyz_c, ms_c) bind(C, name='mopac_cuda_cosmo_gradient') result(rc)
        use iso_c_binding, only : c_int, c_double
        integer(c_int), value :: numat_c, nps_c, swig_c, nden_c
        real(c_double) :: coord_c(*), srad_c(*), ddi_c(*), qqi2_c(*), pxyz_c(*), q_c(*), &
          zeta_c(*), fsw_c(*), qden_c(*), dxyz_c(*)
        integer(c_int) :: idel_c(*), iden_c(*), iatsp_c(*)
        real(c_double), value :: fact_c, fepsi_c
        real(c_double) :: ms_c
        integer(c_int) :: rc
      end function mopac_cuda_cosmo_gradient
    end interface
    integer(c_int), allocatable :: idel(:), iden(:), iat(:)
    double precision, allocatable :: ddi(:), qqi2(:), pxyz(:, :), zeta(:), fsw(:)
    integer :: i, k, n
    integer(c_int) :: rc
    real(c_double) :: ms
    double precision :: timer
    allocate (idel(numat), iden(numat), iat(nps), ddi(numat), qqi2(numat), pxyz(nps, 3))
    done = .false.
    n = 0
    do i = 1, numat
      idel(i) = nlast(i) + 1 - nfirst(i)
      iden(i) = n
      n = n + (idel(i)*(idel(i) + 1))/2
      ddi(i) = dd(nat(i)) * a0
      qqi2(i) = (a0*qq(nat(i)))**2
    end do
    if (n > size (qdenet, 1)) return
    do k = 1, nps
      iat(k) = iatsp(k) - 1
      pxyz(k, 1:3) = cosurf(1:3, k)
    end do
    if (swig) then
      allocate (zeta(nps), fsw(nps))
      zeta = swig_zeta(1:nps)
      fsw = swig_f(1:nps)
    else
      allocate (zeta(1), fsw(1))
      zeta = 0.d0
      fsw = 1.d0
    end if
    call mozyme_section_timer_begin('diegrd_gpu', timer)
    rc = mopac_cuda_cosmo_gradient(int(numat, c_int), int(nps, c_int), &
      merge(1_c_int, 0_c_int, swig), int(size (qdenet, 1), c_int), coord, srad, idel, iden, ddi, &
      qqi2, pxyz, qscnet(1:nps, 3), zeta, fsw, iat, qdenet(:, 3), -ev*a0*fpc_9, fepsi, dxyz, ms)
    call mozyme_section_timer_end('diegrd_gpu', timer)
    done = (rc == 0)
#else
    done = .false.
#endif
end subroutine diegrd_gpu
