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

    Subroutine eigenvectors_LAPACK(eigenvecs, xmat, eigvals, ndim)
      USE chanel_C, only : iw
#ifdef GPU
      Use mod_vars_cuda, only: lgpu, ngpus, prec
      use eigenvectors_cuda_mod, only: eigenvectors_CUDA, eigenvectors_CUDA_keep, eigenvectors_CUDA_fetch
      use gpu_ortho_interfaces
      use overlap_build, only: build_overlap_packed, unpack_upper_to_full
      use partial_eig_gpu
      use molkst_C, only: uhf, nclose, nalpha, nbeta
      use eig_call_context, only: current_spin
#endif
#if (MAGMA)
      Use magma
      use initMagma
#endif

      implicit none
      Integer :: ndim
      double precision :: eigenvecs(ndim,ndim), &
                    & eigvals(ndim),xmat((ndim*(ndim+1))/2)
      integer :: i, j
      Integer :: lwork, liwork, info
      double precision,dimension(1:10) :: work_tmp
      Integer, dimension(1:10) :: iwork_tmp
      double precision, allocatable :: work(:)
      Integer, allocatable :: iwork(:)
#ifdef GPU
      ! Local controls for GPU diag selection
      integer :: thr_min, stat_env
      character(len=32) :: env, fast, fetch, ortho
      logical :: fastgpu, fetch_eigs, ortho_gpu
      double precision, allocatable :: Sfull(:,:), Spack(:)
      ! Partial eigensolve controls
      integer :: nocc, itmax
      double precision :: etol
#endif
!==============================================================================
! Code to find all eigenvectors and all eigenvalues for a symmetric General matrix
! using LAPACK and MAGMA
! Gerd Bruno Rocha and Julio Carvalho Maia 11/17/2013.
!==============================================================================
      continue
      eigvals = 0.d0
      eigenvecs = 0.d0

!
! Perturb secular determinant matrix to split exact degeneracies
! (This is to get around a bug in the diagonalizer that causes eigenvectors to not be orthonormal)
!
!  The following unusual construction works.  Do NOT change it unless degeneracy tests have been done,
!  and the results are reproducible.
!
      forall (i=1:ndim)
          xmat((i*(i + 1))/2) = xmat((i*(i + 1))/2) + i*1.d-10   ! Do NOT go much higher than 1.d-10, otherwise the geometry
      endforall                                              ! optimization might go into an endless loop.
      call dtpttr( 'u', ndim, xmat, eigenvecs, ndim, i )
      
#ifdef MKL
#ifdef GPU
if (lgpu .and. (ngpus > 1 .and. ndim > 100)) then
      call mkl_dimatcopy('C', 'T' , ndim, ndim, 1.0d0, eigenvecs, ndim, ndim)
end if
#endif
#endif
      if (i /= 0) stop 'error in dtpttr'

#ifdef GPU
      ! Use GPU eigensolver only when problem size is large enough.
      ! Threshold can be overridden via environment variable MOPAC_GPU_EIGEN_MIN.
      if (lgpu) then
        thr_min = 400
        env = '' ; fast = '' ; fetch = '' ; ortho = ''
        fastgpu = .false. ; fetch_eigs = .false. ; ortho_gpu = .false.
        stat_env = 1
        call get_environment_variable('MOPAC_GPU_EIGEN_MIN', env, status=stat_env)
        if (stat_env == 0) then
          ! Ignore parse errors silently; keep default if read fails
          read(env, *, end=10, err=10) thr_min
        end if
10      continue
        call get_environment_variable('MOPAC_FASTGPU', fast, status=stat_env)
        if (stat_env == 0) fastgpu = (trim(adjustl(fast)) /= '')
        call get_environment_variable('MOPAC_EIG2HOST', fetch, status=stat_env)
        if (stat_env == 0) fetch_eigs = (trim(adjustl(fetch)) /= '')
        call get_environment_variable('MOPAC_ORTHO_GPU', ortho, status=stat_env)
        if (stat_env == 0) ortho_gpu = (trim(adjustl(ortho)) /= '')

        ! Optional: apply GPU orthogonalization transform F' = X^T F X using overlap S
        if (ortho_gpu) then
          allocate(Spack((ndim*(ndim+1))/2), Sfull(ndim,ndim), stat=i)
          if (i == 0) then
            call build_overlap_packed(Spack)
            call unpack_upper_to_full(ndim, Spack, Sfull)
            i = 0
            call mopac_cuda_transform_fock_with_s(ndim, Sfull, ndim, eigenvecs, ndim, i)
            ! eigenvecs now contains transformed F' on host
          end if
          if (allocated(Sfull)) deallocate(Sfull)
          if (allocated(Spack)) deallocate(Spack)
          ! Force keep-on-GPU path since we now supply full F' in eigenvecs
          fastgpu = .true.
        end if
        ! Optional: experimental partial eigensolve for RHF (smallest nclose eigenpairs)
        env = ''
        call get_environment_variable('MOPAC_PARTIAL_EIG', env, status=stat_env)
        if (stat_env == 0 .and. trim(adjustl(env)) /= '') then
          if (uhf) then
            if (current_spin == 1) then
              nocc = max(1, min(nclose + nalpha, ndim))
            else if (current_spin == 2) then
              nocc = max(1, min(nclose + nbeta, ndim))
            else
              nocc = max(1, min(nclose, ndim))
            end if
          else
            nocc = max(1, min(nclose, ndim))
          end if
          etol = 1.d-8
          itmax = 50
          call get_environment_variable('MOPAC_PARTIAL_TOL', env, status=stat_env)
          if (stat_env == 0) then
            read(env, *, end=20, err=20) etol
          end if
20        continue
          call get_environment_variable('MOPAC_PARTIAL_MAXIT', env, status=stat_env)
          if (stat_env == 0) then
            read(env, *, end=30, err=30) itmax
          end if
30        continue
          ! Ensure we are in orthonormal basis when possible
          if (ortho_gpu) then
            ! eigenvecs currently holds transformed F'
            call block_subspace_smallest(eigenvecs, ndim, nocc, eigvals, eigenvecs, itmax, etol)
            return
          else
            ! Fall back to CPU GEMM path using current full matrix unpacked earlier
            call block_subspace_smallest(eigenvecs, ndim, nocc, eigvals, eigenvecs, itmax, etol)
            return
          end if
        end if

        if (ndim >= thr_min) then
          if (fastgpu) then
            ! Keep eigenvectors on device; optionally fetch to host
            call eigenvectors_CUDA_keep(eigenvecs, eigvals, ndim)
            if (fetch_eigs) then
              call eigenvectors_CUDA_fetch(eigenvecs, ndim)
            end if
          else
            call eigenvectors_CUDA(eigenvecs, xmat, eigvals, ndim)
          end if
          return
        end if
      end if
#endif

      j = i ! Dummy - to make FORCHECK not complain about "j"
      i = j ! Dummy
      if (i == -999) return
      lwork = -1
      liwork = -1

! GBR_new_addition

#if (MAGMA)
      if (lgpu .and. ndim > 100) then
         if (ngpus > 1) then
             call magma_dsyevd_Driver1(ngpus,'v','l',ndim,eigenvecs,ndim,eigvals,&
                    & work_tmp,lwork,iwork_tmp,liwork,info)
          else
             call magma_dsyevd_Driver1(ngpus,'v','u',ndim,eigenvecs,ndim,eigvals,&
                    & work_tmp,lwork,iwork_tmp,liwork,info)
          end if
      else
         call dsyevd('v','u',ndim,eigenvecs,ndim,eigvals,work_tmp,&
                    & lwork,iwork_tmp,liwork,info)
      end if
#else
      call dsyevd('v','u',ndim,eigenvecs,ndim,eigvals,work_tmp, &
                    & lwork,iwork_tmp,liwork,info)
#endif
      lwork = int(work_tmp(1))
      liwork = iwork_tmp(1)
      allocate (work(lwork), iwork(liwork), stat = i)
!      forall (j=1:lwork) work(j) = 0.d0
!      forall (j=1:liwork) iwork(j) = 0
#if (MAGMA)
      if (lgpu .and. ndim > 100) then
         if (ngpus > 1) then
             call magma_dsyevd_Driver2(ngpus,'v','l',ndim,eigenvecs,ndim,eigvals,&
                    & work,lwork,iwork,liwork,info)
          else
             call magma_dsyevd_Driver2(ngpus,'v','u',ndim,eigenvecs,ndim,eigvals,&
                    & work,lwork,iwork,liwork,info)
          end if
      else
         call dsyevd('v','u',ndim,eigenvecs,ndim,eigvals,work,&
                                & lwork,iwork,liwork,info)
      end if
#else

      call dsyevd('v','u',ndim,eigenvecs,ndim,eigvals,work, &
                                & lwork,iwork,liwork,info)
#endif

      deallocate (iwork,work,stat = j)
      if (info /= 0)  write(iw,*) ' dsyevd Diagonalization error., CODE =',info
      continue
      return
    End subroutine eigenvectors_LAPACK
