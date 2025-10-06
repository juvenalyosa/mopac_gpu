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
      integer :: iopc_eff, istat_env, istat_loc
      character(len=32) :: env_cpu
      logical :: use_resident, gpu_density_used, need_host_density
      logical :: allow_gpu
      logical :: legacy_env
      character(len=32) :: env_resident
      logical :: debug_density
      logical :: verify_density
      character(len=32) :: env_debug
      double precision, allocatable :: pp_ref(:)
      double precision, allocatable :: pp_dev(:)
      double precision, allocatable :: xmat_ref(:,:)
      double precision :: diff, max_diff, rms_acc, max_dev, rms_dev
      logical :: verification_failed
      integer :: idx, info_ref
      logical(c_bool) :: ok_dev
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
      allow_gpu = .true.
      legacy_env = .false.
      env_cpu = '' ; istat_env = 1
      call get_environment_variable('MOPAC_CPU_DENSITY', env_cpu, status=istat_env)
      if (istat_env == 0) then
        if (trim(adjustl(env_cpu)) /= '') then
          if (iopc == 4) iopc_eff = 5   ! GPU DSYRK -> CPU DSYRK
          if (iopc == 2) iopc_eff = 3   ! GPU DGEMM -> CPU DGEMM
          allow_gpu = .false.
        end if
      end if
      call get_environment_variable('MOPAC_GPU_EXACT_SC', env_cpu, status=istat_env)
      if (istat_env == 0) then
        env_cpu = adjustl(env_cpu)
        if (len_trim(env_cpu) /= 0) then
          select case (env_cpu(1:1))
          case ('0','n','N','f','F','o','O')
            allow_gpu = .false.
          case default
            allow_gpu = .true.
          end select
        end if
      end if

      env_cpu = '' ; istat_env = 1
      call get_environment_variable('MOPAC_FOCK_GPU', env_cpu, status=istat_env)
      if (istat_env == 0) then
        if (trim(adjustl(env_cpu)) /= '') legacy_env = .true.
      end if

      if (.not. allow_gpu .and. .not. legacy_env) then
        write(6,'(1x,a)') '[GPU DENSITY] GPU path disabled via MOPAC_GPU_EXACT_SC'
        call flush(6)
      else if (legacy_env .and. .not. allow_gpu) then
        write(6,'(1x,a)') '[GPU DENSITY] legacy MOPAC_FOCK_GPU request ignored – enable the GPU path explicitly'
        call flush(6)
      end if

      use_resident = allow_gpu
      env_resident = '' ; istat_env = 1
      call get_environment_variable('MOPAC_RESIDENT_SCF', env_resident, status=istat_env)
      if (istat_env == 0) then
        env_resident = adjustl(env_resident)
        if (len_trim(env_resident) > 0) then
          select case (env_resident(1:1))
          case ('0','n','N','f','F','o','O')
            use_resident = .false.
          case default
            use_resident = allow_gpu
          end select
          if (len_trim(env_resident) >= 3) then
            if (env_resident(1:3) == 'off' .or. env_resident(1:3) == 'OFF') use_resident = .false.
          end if
        end if
      end if
      if (.not. allow_gpu) use_resident = .false.
      gpu_density_used = .false.
      resident_scf = use_resident
      need_host_density = .true.
      call mopac_cuda_set_resident_mode(merge(1,0,use_resident))
      debug_density = .false.
      verify_density = .false.
      env_debug = '' ; istat_env = 1
      call get_environment_variable('MOPAC_GPU_VERIFY_DENSITY', env_debug, status=istat_env)
      if (istat_env == 0) then
        if (trim(adjustl(env_debug)) /= '') then
          verify_density = .true.
          debug_density = .true.
        end if
      end if
      env_debug = '' ; istat_env = 1
      call get_environment_variable('MOPAC_GPU_DENSITY_DEBUG', env_debug, status=istat_env)
      if (istat_env == 0) then
        if (trim(adjustl(env_debug)) /= '') debug_density = .true.
      end if
      if (verify_density) need_host_density = .true.
      if (debug_density) then
        write(6,'(1x,"[GPU density debug] enabled: iopc=",i0)') iopc_eff
        call flush(6)
      end if
      verification_failed = .false.
      Select case (iopc_eff)
#else
      Select case (iopc)
#endif
        case(2)   ! Option to use dgemm from CUBLAS
#ifdef GPU
          if (allow_gpu .and. have_device_eigvecs .and. device_eigvecs_n == norbs) then
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
            if (debug_density) then
              allocate(xmat_ref(norbs,norbs), stat=istat_loc)
              xmat_ref = 0.d0
              nl21 = Min (norbs, nl2)
              nl11 = Min (norbs, nl1)
              if (nu2 >= nl2) then
                call dgemm('N','T', norbs, norbs, nu2-nl2+1, 2.0d0*sign, &
                     c(1:norbs,nl21:norbs), norbs, c(1:norbs,nl21:norbs), norbs, 0.0d0, xmat_ref, norbs)
              end if
              if (nu1 >= nl1) then
                call dgemm('N','T', norbs, norbs, nu1-nl1+1, frac*sign, &
                     c(1:norbs,nl11:norbs), norbs, c(1:norbs,nl11:norbs), norbs, 1.0d0, xmat_ref, norbs)
              end if
              do idx = 1, norbs
                xmat_ref(idx,idx) = xmat_ref(idx,idx) + cst
              end do
              allocate(pp_ref(mpack), stat=istat_loc)
              info_ref = 0
              call dtrttp('u', norbs, xmat_ref, norbs, pp_ref, info_ref)
              max_diff = 0.d0
              rms_acc = 0.d0
              do idx = 1, mpack
                diff = pp(idx) - pp_ref(idx)
                if (abs(diff) > max_diff) max_diff = abs(diff)
                rms_acc = rms_acc + diff*diff
              end do
              if (mpack > 0) rms_acc = sqrt(rms_acc / mpack)
              if (verify_density .and. max_diff > 1.d-9) then
                write(6,'(1x,"[GPU density verify] case=2 max diff=",1pe12.5)') max_diff
                call flush(6)
                verification_failed = .true.
                pp(:) = pp_ref(:)
                need_host_density = .true.
                gpu_density_used = .false.
                use_resident = .false.
                resident_scf = .false.
                call mopac_cuda_set_resident_mode(0)
                call mopac_cuda_clear_density_cache()
              else
                write(*,'(1x,"[GPU density debug] case=2 max=",1pe12.5," rms=",1pe12.5)') max_diff, rms_acc
                if (max_diff > 1.d-6 .and. mpack >= 5) then
                  write(*,'(1x,"[GPU density debug] case=2 sample",5(1x,1pe12.5))') &
                       pp(1),pp_ref(1),pp(2),pp_ref(2),pp(3)
                end if
                call flush(6)
              end if
            end if
#ifdef GPU
            if (.not. verification_failed .and. use_resident) then
              call mopac_cuda_register_packed_density(mpack, pp)
              if (debug_density .and. .not. verification_failed) then
                allocate(pp_dev(mpack), stat=istat_loc)
                ok_dev = mopac_cuda_fetch_packed_density(pp_dev, int(mpack, kind=c_size_t))
                if (ok_dev .eqv. .true._c_bool) then
                  max_dev = 0.d0
                  rms_dev = 0.d0
                  do idx = 1, mpack
                    diff = pp_dev(idx) - pp_ref(idx)
                    if (abs(diff) > max_dev) max_dev = abs(diff)
                    rms_dev = rms_dev + diff*diff
                  end do
                  if (mpack > 0) rms_dev = sqrt(rms_dev / mpack)
                  write(*,'(1x,"[GPU density debug] case=2 device max=",1pe12.5," rms=",1pe12.5)') max_dev, rms_dev
                else
                  write(*,'(1x,"[GPU density debug] case=2 device fetch FAILED")')
                end if
                call flush(6)
                deallocate(pp_dev, stat=istat_loc)
              end if
            else
              call mopac_cuda_clear_density_cache()
            end if
#endif
            if (debug_density) then
              deallocate(pp_ref, xmat_ref, stat=istat_loc)
            end if
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
            if (debug_density) then
              nl21 = Min (norbs, nl2)
              nl11 = Min (norbs, nl1)
              allocate(xmat_ref(norbs,norbs), stat=istat_loc)
              xmat_ref = 0.d0
              if (nu2 >= nl2) then
                call dgemm('N','T', norbs, norbs, nu2-nl2+1, 2.0d0*sign, &
                     c(1:norbs,nl21:norbs), norbs, c(1:norbs,nl21:norbs), norbs, 0.0d0, xmat_ref, norbs)
              end if
              if (nu1 >= nl1) then
                call dgemm('N','T', norbs, norbs, nu1-nl1+1, frac*sign, &
                     c(1:norbs,nl11:norbs), norbs, c(1:norbs,nl11:norbs), norbs, 1.0d0, xmat_ref, norbs)
              end if
              do idx = 1, norbs
                xmat_ref(idx,idx) = xmat_ref(idx,idx) + cst
              end do
              allocate(pp_ref(mpack), stat=istat_loc)
              info_ref = 0
              call dtrttp('u', norbs, xmat_ref, norbs, pp_ref, info_ref)
              max_diff = 0.d0
              rms_acc = 0.d0
              do idx = 1, mpack
                diff = pp(idx) - pp_ref(idx)
                if (abs(diff) > max_diff) max_diff = abs(diff)
                rms_acc = rms_acc + diff*diff
              end do
              if (mpack > 0) rms_acc = sqrt(rms_acc / mpack)
              if (verify_density .and. max_diff > 1.d-9) then
                write(6,'(1x,"[GPU density verify] case=2b max diff=",1pe12.5)') max_diff
                call flush(6)
                verification_failed = .true.
                pp(:) = pp_ref(:)
                need_host_density = .true.
                gpu_density_used = .false.
                use_resident = .false.
                resident_scf = .false.
                call mopac_cuda_set_resident_mode(0)
                call mopac_cuda_clear_density_cache()
              else
                write(*,'(1x,"[GPU density debug] case=2b max=",1pe12.5," rms=",1pe12.5)') max_diff, rms_acc
                if (max_diff > 1.d-6 .and. mpack >= 5) then
                  write(*,'(1x,"[GPU density debug] case=2b sample",5(1x,1pe12.5))') &
                       pp(1),pp_ref(1),pp(2),pp_ref(2),pp(3)
                end if
                call flush(6)
              end if
              deallocate(pp_ref, xmat_ref, stat=istat_loc)
            end if
#ifdef GPU
            if (.not. verification_failed .and. use_resident) then
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
          if (use_resident) then
            call mopac_cuda_register_packed_density(mpack, pp)
          else
            call mopac_cuda_clear_density_cache()
          end if
#endif
        case(4)   ! Option to use dsyrk from CUBLAS
#ifdef GPU
          if (allow_gpu .and. have_device_eigvecs .and. device_eigvecs_n == norbs .and. fract < 1.d-2) then
            gpu_density_used = .true.
            allocate(xmat(norbs,norbs),stat = i)
            call mopac_cuda_density_from_dev_syrk(norbs, ndubl, occ, xmat, norbs)
            call dtrttp('u', norbs, xmat, norbs, pp, i )
            if (debug_density) then
              allocate(xmat_ref(norbs,norbs), stat=istat_loc)
              xmat_ref = 0.d0
              if (ndubl > 0) then
                call dgemm('N','T', norbs, norbs, ndubl, 2.0d0*sign, &
                     c(1:norbs,1:ndubl), norbs, c(1:norbs,1:ndubl), norbs, 0.0d0, xmat_ref, norbs)
              end if
              if (nsingl > ndubl) then
                call dgemm('N','T', norbs, norbs, nsingl-ndubl, frac*sign, &
                     c(1:norbs,ndubl+1:nsingl), norbs, c(1:norbs,ndubl+1:nsingl), norbs, 1.0d0, xmat_ref, norbs)
              end if
              do idx = 1, norbs
                xmat_ref(idx,idx) = xmat_ref(idx,idx) + cst
              end do
              allocate(pp_ref(mpack), stat=istat_loc)
              call dtrttp('u', norbs, xmat_ref, norbs, pp_ref, info_ref)
              max_diff = 0.d0
              rms_acc = 0.d0
              do idx = 1, mpack
                diff = pp(idx) - pp_ref(idx)
                if (abs(diff) > max_diff) max_diff = abs(diff)
                rms_acc = rms_acc + diff*diff
              end do
              if (mpack > 0) rms_acc = sqrt(rms_acc / mpack)
              write(*,'(1x,"[GPU density debug] case=4 max=",1pe12.5," rms=",1pe12.5)') max_diff, rms_acc
              if (max_diff > 1.d-6 .and. mpack >= 5) then
                write(*,'(1x,"[GPU density debug] case=4 sample ",5(1x,1pe12.5))') &
                     pp(1), pp_ref(1), pp(2), pp_ref(2), pp(3)
              end if
              call flush(6)
            end if
#ifdef GPU
            if (use_resident) then
              call mopac_cuda_register_packed_density(mpack, pp)
              if (debug_density) then
                allocate(pp_dev(mpack), stat=istat_loc)
                ok_dev = mopac_cuda_fetch_packed_density(pp_dev, int(mpack, kind=c_size_t))
                if (ok_dev .eqv. .true._c_bool) then
                  max_dev = 0.d0
                  rms_dev = 0.d0
                  do idx = 1, mpack
                    diff = pp_dev(idx) - pp_ref(idx)
                    if (abs(diff) > max_dev) max_dev = abs(diff)
                    rms_dev = rms_dev + diff*diff
                  end do
                  if (mpack > 0) rms_dev = sqrt(rms_dev / mpack)
                  write(*,'(1x,"[GPU density debug] case=4 device max=",1pe12.5," rms=",1pe12.5)') max_dev, rms_dev
                else
                  write(*,'(1x,"[GPU density debug] case=4 device fetch FAILED")')
                end if
                call flush(6)
                deallocate(pp_dev, stat=istat_loc)
              end if
            else
              call mopac_cuda_clear_density_cache()
            end if
#endif
            if (debug_density) then
              deallocate(pp_ref, xmat_ref, stat=istat_loc)
            end if
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
            gpu_density_used = .false.
            call dgemm('N','T', norbs, norbs, ndubl, 2.0d0*sign, &
                 c(1:norbs,1:ndubl), norbs, c(1:norbs,1:ndubl), norbs, 0.0d0, xmat, norbs)
            if (nsingl > ndubl) then
              call dgemm('N','T', norbs, norbs, nsingl-ndubl, frac*sign, &
                   c(1:norbs,ndubl+1:nsingl), norbs, c(1:norbs,ndubl+1:nsingl), norbs, 1.0d0, xmat, norbs)
            end if
            if (need_host_density) then
              forall (i=1:norbs)
                 xmat(i,i) = xmat(i,i) + cst
              endforall
              call dtrttp('u', norbs, xmat, norbs, pp, i )
              if (debug_density) then
                allocate(xmat_ref(norbs,norbs), stat=istat_loc)
                xmat_ref = 0.d0
                if (ndubl > 0) then
                  call dgemm('N','T', norbs, norbs, ndubl, 2.0d0*sign, &
                       c(1:norbs,1:ndubl), norbs, c(1:norbs,1:ndubl), norbs, 0.0d0, xmat_ref, norbs)
                end if
                if (nsingl > ndubl) then
                  call dgemm('N','T', norbs, norbs, nsingl-ndubl, frac*sign, &
                       c(1:norbs,ndubl+1:nsingl), norbs, c(1:norbs,ndubl+1:nsingl), norbs, 1.0d0, xmat_ref, norbs)
                end if
                do idx = 1, norbs
                  xmat_ref(idx,idx) = xmat_ref(idx,idx) + cst
                end do
                allocate(pp_ref(mpack), stat=istat_loc)
                call dtrttp('u', norbs, xmat_ref, norbs, pp_ref, info_ref)
                max_diff = 0.d0
                rms_acc = 0.d0
                do idx = 1, mpack
                  diff = pp(idx) - pp_ref(idx)
                  if (abs(diff) > max_diff) max_diff = abs(diff)
                  rms_acc = rms_acc + diff*diff
                end do
                if (mpack > 0) rms_acc = sqrt(rms_acc / mpack)
                write(*,'(1x,"[GPU density debug] case=4-fallback max=",1pe12.5," rms=",1pe12.5)') max_diff, rms_acc
                if (max_diff > 1.d-6 .and. mpack >= 5) then
                  write(*,'(1x,"[GPU density debug] case=4-fallback sample ",5(1x,1pe12.5))') &
                       pp(1), pp_ref(1), pp(2), pp_ref(2), pp(3)
                end if
                call flush(6)
                deallocate(pp_ref, xmat_ref, stat=istat_loc)
              end if
            else
              pp(:) = 0.d0
            end if
#ifdef GPU
            if (use_resident) then
              call mopac_cuda_clear_density_cache()
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
