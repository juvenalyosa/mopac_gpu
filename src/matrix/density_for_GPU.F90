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

subroutine density_for_GPU (c, fract, ndubl, nsingl, occ, mpack, norbs, mode, pp, iopc)
#ifdef GPU
      Use mod_vars_cuda, only: real_cuda, prec, nthreads_gpu, nblocks_gpu, ngpus, resident_scf
      Use iso_c_binding
      Use density_cuda_i
      Use mopac_cublas_interfaces
      use gpu_density_interfaces
      use gpu_diag_state, only: have_device_eigvecs, device_eigvecs_n, gpu_diag_clear
      use gpu_runtime_interfaces, only: mopac_cuda_set_resident_mode
#endif
      implicit none
      Integer :: ndubl, nsingl, mode, mpack, norbs, nl1, nl2, nu1, nu2, i, j, l, &
	           & nl21, nl11, iopc
      double precision,allocatable :: xmat(:,:)
      double precision :: c(norbs,norbs), pp(mpack)
      double precision :: cst, sign, fract, frac, occ, sum1, sum2
#ifdef GPU
      double precision, allocatable :: pdens(:)
      integer :: iopc_eff, istat_env
      character(len=32) :: env_cpu
      logical :: use_resident, gpu_density_used, need_host_density
      character(len=32) :: env_resident
#endif
      if (ndubl /= 0 .and. nsingl > (norbs/2) .and. mode /= 2) then
        !
        !    TAKE POSITRON EQUIVALENT
        !
        sign = -1.d0
        frac = occ - fract
        cst = occ
        nl2 = nsingl + 1
        nu2 = norbs
        nl1 = ndubl + 1
        nu1 = nsingl
      else
        !
        !    TAKE ELECTRON EQUIVALENT
        !
        sign = 1.d0
        frac = fract
        cst = 0.d0
        nl2 = 1
        nu2 = ndubl
        nl1 = ndubl + 1
        nu1 = nsingl
      end if
      ! Allow runtime override to force CPU density even when lgpu is true
#ifdef GPU
      iopc_eff = iopc
      env_cpu = '' ; istat_env = 1
      call get_environment_variable('MOPAC_CPU_DENSITY', env_cpu, status=istat_env)
      if (istat_env == 0) then
        if (trim(adjustl(env_cpu)) /= '') then
          if (iopc == 4) iopc_eff = 5   ! GPU DSYRK -> CPU DSYRK
          if (iopc == 2) iopc_eff = 3   ! GPU DGEMM -> CPU DGEMM
        end if
      end if
      use_resident = .true.
      env_resident = '' ; istat_env = 1
      call get_environment_variable('MOPAC_RESIDENT_SCF', env_resident, status=istat_env)
      if (istat_env == 0) then
        env_resident = adjustl(env_resident)
        if (len_trim(env_resident) > 0) then
          select case (env_resident(1:1))
          case ('0','n','N','f','F','o','O')
            use_resident = .false.
          end select
          if (len_trim(env_resident) >= 3) then
            if (env_resident(1:3) == 'off' .or. env_resident(1:3) == 'OFF') use_resident = .false.
          end if
        end if
      end if
      gpu_density_used = .false.
      resident_scf = use_resident
      need_host_density = .not. use_resident
      call mopac_cuda_set_resident_mode(merge(1,0,use_resident))
      Select case (iopc_eff)
#else
      Select case (iopc)
#endif
        case(2)   ! Option to use dgemm from CUBLAS
#ifdef GPU
          if (have_device_eigvecs .and. device_eigvecs_n == norbs) then
            gpu_density_used = .true.
            allocate(xmat(norbs,norbs),stat = i)
            call mopac_cuda_density_from_dev_gemm(norbs, nl2, nu2, nl1, nu1, sign, frac, xmat, norbs)
            if (need_host_density) then
              forall (i=1:norbs)
                 xmat(i,i) = xmat(i,i) + cst
              endforall
            end if
#ifdef GPU
            if (use_resident) then
              call mopac_cuda_density_add_diag(norbs, cst)
            else
              call mopac_cuda_clear_density_cache()
            end if
#endif
            call dtrttp('u', norbs, xmat, norbs, pp, i )
#ifdef GPU
            if (use_resident) then
              call mopac_cuda_register_packed_density(mpack, pp)
            else
              call mopac_cuda_clear_density_cache()
            end if
#endif
            deallocate (xmat,stat=i)
            ! Default: keep device eigvecs to reduce transfers; clear only when explicitly requested
            env_cpu = '' ; istat_env = 1
            call get_environment_variable('MOPAC_EIG2HOST', env_cpu, status=istat_env)
            if (istat_env == 0 .and. trim(adjustl(env_cpu)) /= '') then
              call gpu_diag_clear()
            else
              env_cpu = '' ; istat_env = 1
              call get_environment_variable('MOPAC_CLEAR_DEVICE', env_cpu, status=istat_env)
              if (istat_env == 0 .and. trim(adjustl(env_cpu)) /= '') then
                call gpu_diag_clear()
              end if
            end if
          else
            nl21 = Min (norbs, nl2)
            nl11 = Min (norbs, nl1)
            allocate(xmat(norbs,norbs),stat = i)
            if (ngpus > 1) then
              gpu_density_used = .true.
              call gemm_cublas_multi ('N', 'T', norbs, norbs, nu2-nl2+1, 2.0_prec*sign, c(1:norbs,nl21:norbs),&
                          &   norbs, c(1:norbs,nl21:norbs), norbs, 0.0_prec, xmat, norbs)
              call gemm_cublas_multi ('N', 'T', norbs, norbs, nu1-nl1+1, frac*sign, c(1:norbs,nl11:norbs), &
                          &   norbs, c(1:norbs,nl11:norbs), norbs, 1.0_prec, xmat, norbs)
            else
              gpu_density_used = .true.
              call gemm_cublas ('N', 'T', norbs, norbs, nu2-nl2+1, 2.0_prec*sign, c(1:norbs,nl21:norbs),&
                          &   norbs, c(1:norbs,nl21:norbs), norbs, 0.0_prec, xmat, norbs)
              call gemm_cublas ('N', 'T', norbs, norbs, nu1-nl1+1, frac*sign, c(1:norbs,nl11:norbs), &
                          &   norbs, c(1:norbs,nl11:norbs), norbs, 1.0_prec, xmat, norbs)
            end if
            if (need_host_density) then
              forall (i=1:norbs)
                 xmat(i,i) = xmat(i,i) + cst
              endforall
            end if
#ifdef GPU
            if (use_resident) then
              call mopac_cuda_density_add_diag(norbs, cst)
            else
              call mopac_cuda_clear_density_cache()
            end if
#endif
            call dtrttp('u', norbs, xmat, norbs, pp, i )
#ifdef GPU
            if (use_resident) then
              call mopac_cuda_register_packed_density(mpack, pp)
            else
              call mopac_cuda_clear_density_cache()
            end if
#endif
            deallocate (xmat,stat=i)
          end if
#endif
        case(3)   ! Option to use dgemm from BLAS

          nl21 = Min (norbs, nl2)
          nl11 = Min (norbs, nl1)

          allocate(xmat(norbs,norbs),stat = i)
          if (norbs < 0) forall (j=1:norbs,i=1:norbs) xmat(i,j) = 0.d0  ! Dummy statement to 'fool' FORTRAN checks

          call dgemm ('N', 'T', norbs, norbs, nu2-nl2+1, 2.0d0*sign, c(1:norbs,nl21:norbs),&
                        &   norbs, c(1:norbs,nl21:norbs), norbs, 0.0d0, xmat, norbs)
          call dgemm ('N', 'T', norbs, norbs, nu1-nl1+1, frac*sign, c(1:norbs,nl11:norbs), &
                        &   norbs, c(1:norbs,nl11:norbs), norbs, 1.0d0, xmat, norbs)

          forall (i=1:norbs)
             xmat(i,i) = xmat(i,i) + cst
          endforall

          call dtrttp('u', norbs, xmat, norbs, pp, i )

          deallocate (xmat,stat=i)
#ifdef GPU
          call mopac_cuda_clear_density_cache()
#endif
        case(4)   ! Option to use dsyrk from CUBLAS
#ifdef GPU
          if (have_device_eigvecs .and. device_eigvecs_n == norbs .and. fract < 1.d-2) then
            gpu_density_used = .true.
            allocate(xmat(norbs,norbs),stat = i)
            call mopac_cuda_density_from_dev_syrk(norbs, ndubl, occ, xmat, norbs)
            call dtrttp('u', norbs, xmat, norbs, pp, i )
#ifdef GPU
            if (use_resident) then
              call mopac_cuda_register_packed_density(mpack, pp)
            else
              call mopac_cuda_clear_density_cache()
            end if
#endif
            deallocate(xmat,stat=i)
            env_cpu = '' ; istat_env = 1
            call get_environment_variable('MOPAC_EIG2HOST', env_cpu, status=istat_env)
            if (istat_env == 0 .and. trim(adjustl(env_cpu)) /= '') then
              call gpu_diag_clear()
            else
              env_cpu = '' ; istat_env = 1
              call get_environment_variable('MOPAC_CLEAR_DEVICE', env_cpu, status=istat_env)
              if (istat_env == 0 .and. trim(adjustl(env_cpu)) /= '') then
                call gpu_diag_clear()
              end if
            end if
          else
            allocate(xmat(norbs,norbs),stat = i)
            forall (j = 1:norbs, i=1:norbs) xmat(i, j) = 0.d0
            if (ngpus > 1) then
              gpu_density_used = .true.
              call syrk_cublas_multi ('U','N',norbs,ndubl, &
                   & occ,c(1:norbs,1:ndubl),norbs, &
                   & 0.d0,xmat,norbs)
            else
              gpu_density_used = .true.
              call syrk_cublas ('U','N',norbs,ndubl, &
                   & occ,c(1:norbs,1:ndubl),norbs, &
                   & 0.d0,xmat,norbs)
            end if
            if (need_host_density) then
              call dtrttp('u', norbs, xmat, norbs, pp, i )
            else
              pp(:) = 0.d0
            end if
#ifdef GPU
            if (use_resident) then
              call mopac_cuda_register_packed_density(mpack, pp)
            else
              call mopac_cuda_clear_density_cache()
            end if
#endif
            deallocate(xmat,stat=i)
          end if
#endif
        case(5)   ! Option to use dsyrk from BLAS
          if (fract < 1.d-2) then
            allocate(xmat(norbs,norbs),stat = i)
            forall (j = 1:norbs, i=1:norbs) xmat(i, j) = 0.d0
            call dsyrk ('u', 'n', norbs, ndubl, occ, c(1:norbs,1:ndubl), norbs, 0.d0, xmat, norbs) ! For RHF	      	      	
            call dtrttp('u', norbs, xmat, norbs, pp, i )
            deallocate (xmat,stat=i)
          else
!
! The following block should be re-cast in a modern style "someday"
! It's used only infrequently, so updating it is not urgent.
!
            l = 0
            do i = 1, norbs
              do j = 1, i
                l = l + 1
                sum1 = 0.D0
                sum2 = sum(c(i,nl2:nu2)*c(j,nl2:nu2))
                sum2 = sum2*occ
                sum1 = sum(c(i,nl1:nu1)*c(j,nl1:nu1))
                pp(l) = (sum2 + sum1*frac)*sign
              end do
              pp(l) = cst + pp(l)
            end do
          end if
#ifdef GPU
          call mopac_cuda_clear_density_cache()
#endif
    End select
    continue
    return
End subroutine density_for_GPU
