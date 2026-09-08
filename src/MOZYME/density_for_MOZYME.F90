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

subroutine density_for_MOZYME (p, mode, nclose_loc, partp)
   !***********************************************************************
   !
   !   DENSIT COMPUTES THE DENSITY MATRIX GIVEN THE EIGENVECTOR MATRIX, AND
   !          INFORMATION ABOUT THE M.O. OCCUPANCY.
   !
   !  INPUT:  COCC  = COMPRESSED OCCUPIED EIGENVECTOR MATRIX
   !
   !   ON EXIT: P   = DENSITY MATRIX
   !
   !***********************************************************************
    use molkst_C, only: numat, mpack, keywrd
    use MOZYME_C, only : lijbo, nijbo, ncf, ncocc, &
      nncf, iorbs, cocc, icocc, icocc_dim, cocc_dim, gpu_occ_enabled, &
      gpu_virt_enabled, at_res, nnce, nce, ncvir, cvir, icvir
    use chanel_C, only: iw
    use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required
#ifdef GPU
    use iso_c_binding, only: c_int, c_double
    use mozyme_gpu_int_utils, only: mozyme_c_int_nonnegative_or_zero, &
      mozyme_c_int_positive_or_zero
    use mod_vars_cuda, only: lgpu, mozyme_gpu, mozyme_gpu_min_block, ngpus, mozyme_force_2gpu
    use mopac_cublas_interfaces
#endif
    implicit none
    integer, intent (in) :: mode, nclose_loc
    double precision, dimension (mpack), intent (in) :: partp
    double precision, dimension (mpack), intent (inout) :: p
    logical :: first = .true.
    logical, save :: prnt
    logical :: active_occ, active_virt
    logical :: strict_resident_density
    integer :: i, j, j1, ja, jj, k, k1, k2, ka, kk, l, loop, nj
    integer :: lbase, nb, nk
#ifdef GPU
    integer :: gpu_syrk_calls, gpu_gemm_calls
    integer :: density_batch_calls, density_batch_terms
    integer :: cpu_diag_blocks, cpu_offdiag_blocks
    integer :: skipped_diag_blocks, skipped_offdiag_blocks
    integer :: max_diag_block, max_offdiag_j, max_offdiag_k
    integer :: max_ncf_batch
    integer :: task_count, value_count, task_idx, value_idx, terms
    integer :: gpu_alloc_stat
    integer(c_int) :: gpu_code
    real(c_double) :: gpu_wall_ms
    logical :: density_batch_gpu_done
    logical :: all_occ_gpu_enabled
    integer(c_int), allocatable :: task_lbase(:), task_j_offset(:), task_k_offset(:)
    integer(c_int), allocatable :: task_nj(:), task_nk(:), task_diag(:), task_value_offset(:)
    double precision, allocatable :: density_values(:)
    integer :: env_len, env_status
    logical :: trace_gpu_density
    character(len=16) :: gpu_profile_env, gpu_verbose_env

    interface
      function mopac_cuda_mozyme_density_values(task_count_c, value_count_c, &
          cocc_dim_c, task_j_offset_c, task_k_offset_c, task_nj_c, &
          task_nk_c, task_diag_c, task_value_offset_c, cocc_c, values_c, &
          wall_ms_c) bind(C,name='mopac_cuda_mozyme_density_values') result(code)
        import :: c_int, c_double
        integer(c_int), value :: task_count_c, value_count_c, cocc_dim_c
        integer(c_int), intent(in) :: task_j_offset_c(*), task_k_offset_c(*)
        integer(c_int), intent(in) :: task_nj_c(*), task_nk_c(*), task_diag_c(*)
        integer(c_int), intent(in) :: task_value_offset_c(*)
        real(c_double), intent(in) :: cocc_c(*)
        real(c_double) :: values_c(*), wall_ms_c
        integer(c_int) :: code
      end function mopac_cuda_mozyme_density_values
    end interface
#endif
    double precision, allocatable :: xmat(:,:)
    double precision, allocatable :: xblk(:,:)
    double precision, allocatable :: avec(:,:), bvec(:,:)
    double precision :: spinfa, sum
    external :: mozyme_gpu_strict_abort
    integer, external :: ijbo
    strict_resident_density = mozyme_gpu_scf_no_fallback_required()
    if (strict_resident_density) then
      call mozyme_gpu_strict_abort('strict_density_direct_host_rebuild', &
        'MOZYME GPU strict resident SCF does not support direct host density rebuild')
      return
    end if
#ifdef GPU
    gpu_syrk_calls = 0
    gpu_gemm_calls = 0
    density_batch_calls = 0
    density_batch_terms = 0
    cpu_diag_blocks = 0
    cpu_offdiag_blocks = 0
    skipped_diag_blocks = 0
    skipped_offdiag_blocks = 0
    max_diag_block = 0
    max_offdiag_j = 0
    max_offdiag_k = 0
    max_ncf_batch = 0
    task_count = 0
    value_count = 0
    task_idx = 0
    value_idx = 0
    terms = 0
    density_batch_gpu_done = .false.
    all_occ_gpu_enabled = .false.
    trace_gpu_density = .false.
    gpu_profile_env = ' '
    gpu_verbose_env = ' '
    call get_environment_variable('MOPAC_GPU_PROFILE', gpu_profile_env, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(gpu_profile_env) /= '0') trace_gpu_density = .true.
    call get_environment_variable('MOPAC_GPU_VERBOSE', gpu_verbose_env, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(gpu_verbose_env) /= '0') trace_gpu_density = .true.
#endif
    if (first) then
      first = .false.
      prnt = Index (keywrd, " DIAG ") /= 0
    end if

#ifdef GPU
    if (mozyme_density_batch_gpu_enabled() .and. lijbo .and. allocated(nijbo) .and. nclose_loc > 0) then
      max_ncf_batch = maxval(ncf(1:nclose_loc))
      all_occ_gpu_enabled = nclose_loc <= size(gpu_occ_enabled)
      if (all_occ_gpu_enabled) all_occ_gpu_enabled = all(gpu_occ_enabled(1:nclose_loc))
      if (max_ncf_batch > 0 .and. all_occ_gpu_enabled) then
        task_count = 0
        value_count = 0
        do i = 1, nclose_loc
          loop = ncocc(i)
          ja = 0
          do jj = nncf(i) + 1, nncf(i) + ncf(i)
            j = icocc(jj)
            nj = iorbs(j)
            ka = loop
            do kk = nncf(i) + 1, nncf(i) + ncf(i)
              k = icocc(kk)
              l = nijbo(j, k)
              if (j == k) then
                task_count = task_count + 1
                value_count = value_count + (nj * (nj + 1)) / 2
              else if (j > k .and. l >= 0) then
                task_count = task_count + 1
                value_count = value_count + nj * iorbs(k)
              end if
              ka = ka + iorbs(k)
            end do
            ja = ja + nj
          end do
        end do
        if (task_count > 0 .and. value_count > 0) then
          allocate(task_lbase(task_count), task_j_offset(task_count), &
            task_k_offset(task_count), task_nj(task_count), task_nk(task_count), &
            task_diag(task_count), task_value_offset(task_count), &
            density_values(value_count), stat=gpu_alloc_stat)
          if (gpu_alloc_stat == 0) then
            task_idx = 0
            value_idx = 0
            do i = 1, nclose_loc
              loop = ncocc(i)
              ja = 0
              do jj = nncf(i) + 1, nncf(i) + ncf(i)
                j = icocc(jj)
                nj = iorbs(j)
                ka = loop
                do kk = nncf(i) + 1, nncf(i) + ncf(i)
                  k = icocc(kk)
                  l = nijbo(j, k)
                  if (j == k) then
                    task_idx = task_idx + 1
                    terms = (nj * (nj + 1)) / 2
                    task_lbase(task_idx) = mozyme_c_int_nonnegative_or_zero(l)
                    task_j_offset(task_idx) = &
                      mozyme_c_int_nonnegative_or_zero(loop + ja)
                    task_k_offset(task_idx) = mozyme_c_int_nonnegative_or_zero(ka)
                    task_nj(task_idx) = mozyme_c_int_positive_or_zero(nj)
                    task_nk(task_idx) = mozyme_c_int_positive_or_zero(iorbs(k))
                    task_diag(task_idx) = 1_c_int
                    task_value_offset(task_idx) = &
                      mozyme_c_int_nonnegative_or_zero(value_idx)
                    value_idx = value_idx + terms
                  else if (j > k .and. l >= 0) then
                    task_idx = task_idx + 1
                    terms = nj * iorbs(k)
                    task_lbase(task_idx) = mozyme_c_int_nonnegative_or_zero(l)
                    task_j_offset(task_idx) = &
                      mozyme_c_int_nonnegative_or_zero(loop + ja)
                    task_k_offset(task_idx) = mozyme_c_int_nonnegative_or_zero(ka)
                    task_nj(task_idx) = mozyme_c_int_positive_or_zero(nj)
                    task_nk(task_idx) = mozyme_c_int_positive_or_zero(iorbs(k))
                    task_diag(task_idx) = 0_c_int
                    task_value_offset(task_idx) = &
                      mozyme_c_int_nonnegative_or_zero(value_idx)
                    value_idx = value_idx + terms
                  end if
                  ka = ka + iorbs(k)
                end do
                ja = ja + nj
              end do
            end do
            gpu_wall_ms = 0.0_c_double
            gpu_code = mopac_cuda_mozyme_density_values( &
              mozyme_c_int_nonnegative_or_zero(task_count), &
              mozyme_c_int_nonnegative_or_zero(value_count), &
              mozyme_c_int_positive_or_zero(cocc_dim), task_j_offset, &
              task_k_offset, task_nj, task_nk, task_diag, task_value_offset, &
              cocc, density_values, gpu_wall_ms)
            if (gpu_code == 0_c_int) then
              density_batch_gpu_done = .true.
              density_batch_calls = task_count
              density_batch_terms = value_count
              if (trace_gpu_density) then
                write(iw,'(1x,a," success code=",i0," mode=",i0," blocks=",i0," terms=",i0," ms=",f12.6)') &
                  '[MOZYME GPU density_batch]', int(gpu_code), mode, &
                  density_batch_calls, density_batch_terms, gpu_wall_ms
              end if
            else if (trace_gpu_density) then
              write(iw,'(1x,a," fallback_cpu code=",i0," mode=",i0)') &
                '[MOZYME GPU density_batch]', int(gpu_code), mode
            end if
          else if (trace_gpu_density) then
            write(iw,'(1x,a," fallback_cpu code=",i0," mode=",i0)') &
              '[MOZYME GPU density_batch]', -2, mode
          end if
          if (.not. density_batch_gpu_done) then
            if (allocated(task_lbase)) deallocate(task_lbase)
            if (allocated(task_j_offset)) deallocate(task_j_offset)
            if (allocated(task_k_offset)) deallocate(task_k_offset)
            if (allocated(task_nj)) deallocate(task_nj)
            if (allocated(task_nk)) deallocate(task_nk)
            if (allocated(task_diag)) deallocate(task_diag)
            if (allocated(task_value_offset)) deallocate(task_value_offset)
            if (allocated(density_values)) deallocate(density_values)
          end if
        end if
      else if (trace_gpu_density .and. .not. all_occ_gpu_enabled) then
        write(iw,'(1x,a," fallback_cpu code=",i0," mode=",i0)') &
          '[MOZYME GPU density_batch]', -4, mode
      end if
    end if

    if (strict_resident_density .and. .not. density_batch_gpu_done) then
      call mozyme_gpu_strict_abort('strict_density_batch_fallback', &
        'MOZYME GPU strict resident SCF does not support CPU density rebuild')
      return
    end if
#else
    if (strict_resident_density) then
      call mozyme_gpu_strict_abort('strict_density_batch_fallback', &
        'MOZYME GPU strict resident SCF does not support CPU density rebuild')
      return
    end if
#endif
    if (mode == 0) then
      !
      !  INITIALIZE P:  FULL DENSITY CALCULATION
      !
      p(:) = 0.d0
    else if (mode ==-1) then
      !
      !   FULL DENSITY MATRIX IS IN PARTP, REMOVE DENSITY DUE TO
      !   OLD LMO's USED IN SCF
      !
      p(:) = -0.5d0 * partp(:)
    else
      !
      !   PARTIAL DENSITY MATRIX IS IN PARTP, BUILD THE REST OF P
      !
      p(:) = 0.5d0 * partp(:)
    end if

#ifdef GPU
    if (density_batch_gpu_done) then
      do task_idx = 1, task_count
        l = task_lbase(task_idx)
        value_idx = task_value_offset(task_idx)
        if (task_diag(task_idx) == 1_c_int) then
          do j1 = 1, task_nj(task_idx)
            do k1 = 1, j1
              l = l + 1
              value_idx = value_idx + 1
              p(l) = p(l) + density_values(value_idx)
            end do
          end do
        else
          do j1 = 1, task_nj(task_idx)
            do k1 = 1, task_nk(task_idx)
              l = l + 1
              value_idx = value_idx + 1
              p(l) = p(l) + density_values(value_idx)
            end do
          end do
        end if
      end do
      if (allocated(task_lbase)) deallocate(task_lbase)
      if (allocated(task_j_offset)) deallocate(task_j_offset)
      if (allocated(task_k_offset)) deallocate(task_k_offset)
      if (allocated(task_nj)) deallocate(task_nj)
      if (allocated(task_nk)) deallocate(task_nk)
      if (allocated(task_diag)) deallocate(task_diag)
      if (allocated(task_value_offset)) deallocate(task_value_offset)
      if (allocated(density_values)) deallocate(density_values)
    else
#endif
    do i = 1, nclose_loc
#ifdef GPU
      active_occ = mozyme_gpu .and. lgpu
      if (active_occ) then
        if (i < 1 .or. i > size(gpu_occ_enabled)) then
          active_occ = .false.
        else if (.not. gpu_occ_enabled(i)) then
          active_occ = .false.
        end if
      end if
#endif
      loop = ncocc(i)
      ja = 0
      if (lijbo) then
        do jj = nncf(i) + 1, nncf(i) + ncf(i)
          j = icocc(jj)
          nj = iorbs(j)
          ka = loop
          do kk = nncf(i) + 1, nncf(i) + ncf(i)
            k = icocc(kk)
            if (j == k) then
#ifdef GPU
              max_diag_block = max(max_diag_block, nj)
              if (active_occ .and. (nj >= mozyme_gpu_min_block)) then
                ! Use SYRK on GPU to form outer product v*v^T for the diagonal block
                nb = nj
                gpu_syrk_calls = gpu_syrk_calls + 1
                lbase = nijbo(j, k)
                allocate(xmat(nb, nb))
                xmat = 0.0d0
                allocate(avec(nb,1))
                avec(:,1) = cocc(ja+1+loop:ja+nb+loop)
                if (ngpus > 1 .or. mozyme_force_2gpu) then
                  call syrk_cublas_2gpu('L','N', nb, 1, 1.0d0, avec, nb, 0.0d0, xmat, nb)
                else
                  call syrk_cublas('L','N', nb, 1, 1.0d0, avec, nb, 0.0d0, xmat, nb)
                end if
                deallocate(avec)
                l = lbase
                do j1 = 1, nb
                  do k1 = 1, j1
                    l = l + 1
                    p(l) = p(l) + xmat(j1, k1)
                  end do
                end do
                deallocate(xmat)
              else if (active_occ .and. (nj >= mozyme_gpu_min_block)) then
                nb = nj
                lbase = nijbo(j, k)
                allocate(xmat(nb, nb))
                xmat = 0.0d0
                call dsyrk('L','N', nb, 1, 1.0d0, cocc(ja+1+loop), nb, 0.0d0, xmat, nb)
                l = lbase
                do j1 = 1, nb
                  do k1 = 1, j1
                    l = l + 1
                    p(l) = p(l) + xmat(j1, k1)
                  end do
                end do
                deallocate(xmat)
              else
                if (active_occ) then
                  skipped_diag_blocks = skipped_diag_blocks + 1
                else
                  cpu_diag_blocks = cpu_diag_blocks + 1
                end if
#endif
                l = nijbo (j, k)
                do j1 = 1, nj
                  sum = cocc(ja+j1+loop)
                  do k1 = 1, j1
                    k2 = ka + k1
                    l = l + 1
                    p(l) = p(l) + cocc(k2) * sum
                  end do
                end do
#ifdef GPU
              end if
#endif
            else if (j > k .and. nijbo (j, k) >= 0) then
              l = nijbo (j, k)
#ifdef GPU
              max_offdiag_j = max(max_offdiag_j, nj)
              max_offdiag_k = max(max_offdiag_k, iorbs(k))
              if (active_occ .and. (nj >= mozyme_gpu_min_block) .and. (iorbs(k) >= mozyme_gpu_min_block)) then
                ! Off-diagonal block: form outer product a(nj) * b(nk)^T with GEMM (k=1)
                nk = iorbs(k)
                gpu_gemm_calls = gpu_gemm_calls + 1
                allocate(xblk(nj, nk))
                xblk = 0.0d0
                allocate(avec(nj,1), bvec(nk,1))
                avec(:,1) = cocc(ja+1+loop:ja+nj+loop)
                bvec(:,1) = cocc(ka+1:ka+nk)
                if (ngpus > 1 .or. mozyme_force_2gpu) then
                  call gemm_cublas_2gpu('N','T', nj, nk, 1, 1.0d0, avec, nj, bvec, nk, 0.0d0, xblk, nj)
                else
                  call gemm_cublas('N','T', nj, nk, 1, 1.0d0, avec, nj, bvec, nk, 0.0d0, xblk, nj)
                end if
                deallocate(avec, bvec)
                do j1 = 1, nj
                  do k1 = 1, nk
                    l = l + 1
                    p(l) = p(l) + xblk(j1, k1)
                  end do
                end do
                deallocate(xblk)
              else if (active_occ .and. (nj >= mozyme_gpu_min_block) .and. (iorbs(k) >= mozyme_gpu_min_block)) then
                nk = iorbs(k)
                allocate(xblk(nj, nk))
                xblk = 0.0d0
                call dgemm('N','T', nj, nk, 1, 1.0d0, cocc(ja+1+loop), nj, &
                           cocc(ka+1), nk, 0.0d0, xblk, nj)
                do j1 = 1, nj
                  do k1 = 1, nk
                    l = l + 1
                    p(l) = p(l) + xblk(j1, k1)
                  end do
                end do
                deallocate(xblk)
              else
                if (active_occ) then
                  skipped_offdiag_blocks = skipped_offdiag_blocks + 1
                else
                  cpu_offdiag_blocks = cpu_offdiag_blocks + 1
                end if
#endif
                do j1 = 1, nj
                  sum = cocc(ja+j1+loop)
                  do k1 = 1, iorbs(k)
                    k2 = ka + k1
                    l = l + 1
                    p(l) = p(l) + cocc(k2) * sum
                  end do
                end do
#ifdef GPU
              end if
#endif
            end if
            ka = ka + iorbs(k)
          end do
          ja = ja + nj
        end do
      else
        do jj = nncf(i) + 1, nncf(i) + ncf(i)
          j = icocc(jj)
          nj = iorbs(j)
          ka = loop
          do kk = nncf(i) + 1, nncf(i) + ncf(i)
            k = icocc(kk)
            l = ijbo (j, k)
            if (j == k) then
#ifdef GPU
              max_diag_block = max(max_diag_block, nj)
              if (active_occ .and. (nj >= mozyme_gpu_min_block)) then
                nb = nj
                gpu_syrk_calls = gpu_syrk_calls + 1
                lbase = l
                allocate(xmat(nb, nb))
                xmat = 0.0d0
                allocate(avec(nb,1))
                avec(:,1) = cocc(ja+1+loop:ja+nb+loop)
                if (ngpus > 1 .or. mozyme_force_2gpu) then
                  call syrk_cublas_2gpu('L','N', nb, 1, 1.0d0, avec, nb, 0.0d0, xmat, nb)
                else
                  call syrk_cublas('L','N', nb, 1, 1.0d0, avec, nb, 0.0d0, xmat, nb)
                end if
                deallocate(avec)
                l = lbase
                do j1 = 1, nb
                  do k1 = 1, j1
                    l = l + 1
                    p(l) = p(l) + xmat(j1, k1)
                  end do
                end do
                deallocate(xmat)
              else if (active_occ .and. (nj >= mozyme_gpu_min_block)) then
                nb = nj
                lbase = l
                allocate(xmat(nb, nb))
                xmat = 0.0d0
                call dsyrk('L','N', nb, 1, 1.0d0, cocc(ja+1+loop), nb, 0.0d0, xmat, nb)
                l = lbase
                do j1 = 1, nb
                  do k1 = 1, j1
                    l = l + 1
                    p(l) = p(l) + xmat(j1, k1)
                  end do
                end do
                deallocate(xmat)
              else
                if (active_occ) then
                  skipped_diag_blocks = skipped_diag_blocks + 1
                else
                  cpu_diag_blocks = cpu_diag_blocks + 1
                end if
#endif
                do j1 = 1, nj
                  sum = cocc(ja+j1+loop)
                  do k1 = 1, j1
                    k2 = ka + k1
                    l = l + 1
                    p(l) = p(l) + cocc(k2) * sum
                  end do
                end do
#ifdef GPU
              end if
#endif
            else if (j > k .and. l >= 0) then
#ifdef GPU
              max_offdiag_j = max(max_offdiag_j, nj)
              max_offdiag_k = max(max_offdiag_k, iorbs(k))
              if (active_occ .and. (nj >= mozyme_gpu_min_block) .and. (iorbs(k) >= mozyme_gpu_min_block)) then
                nk = iorbs(k)
                gpu_gemm_calls = gpu_gemm_calls + 1
                allocate(xblk(nj, nk))
                xblk = 0.0d0
                allocate(avec(nj,1), bvec(nk,1))
                avec(:,1) = cocc(ja+1+loop:ja+nj+loop)
                bvec(:,1) = cocc(ka+1:ka+nk)
                if (ngpus > 1 .or. mozyme_force_2gpu) then
                  call gemm_cublas_2gpu('N','T', nj, nk, 1, 1.0d0, avec, nj, bvec, nk, 0.0d0, xblk, nj)
                else
                  call gemm_cublas('N','T', nj, nk, 1, 1.0d0, avec, nj, bvec, nk, 0.0d0, xblk, nj)
                end if
                deallocate(avec, bvec)
                do j1 = 1, nj
                  do k1 = 1, nk
                    l = l + 1
                    p(l) = p(l) + xblk(j1, k1)
                  end do
                end do
                deallocate(xblk)
              else if (active_occ .and. (nj >= mozyme_gpu_min_block) .and. (iorbs(k) >= mozyme_gpu_min_block)) then
                nk = iorbs(k)
                allocate(xblk(nj, nk))
                xblk = 0.0d0
                call dgemm('N','T', nj, nk, 1, 1.0d0, cocc(ja+1+loop), nj, &
                           cocc(ka+1), nk, 0.0d0, xblk, nj)
                do j1 = 1, nj
                  do k1 = 1, nk
                    l = l + 1
                    p(l) = p(l) + xblk(j1, k1)
                  end do
                end do
                deallocate(xblk)
              else
                if (active_occ) then
                  skipped_offdiag_blocks = skipped_offdiag_blocks + 1
                else
                  cpu_offdiag_blocks = cpu_offdiag_blocks + 1
                end if
#endif
                do j1 = 1, nj
                  sum = cocc(ja+j1+loop)
                  do k1 = 1, iorbs(k)
                    k2 = ka + k1
                    l = l + 1
                    p(l) = p(l) + cocc(k2) * sum
                  end do
                end do
#ifdef GPU
              end if
#endif
            end if
            ka = ka + iorbs(k)
          end do
          ja = ja + nj
        end do
      end if
    end do
#ifdef GPU
    end if
#endif
#ifdef GPU
    if (mozyme_gpu .and. trace_gpu_density) then
      write(iw,'(1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
        '[MOZYME GPU density] mode=', mode, 'minblk=', mozyme_gpu_min_block, &
        'gpu_syrk=', gpu_syrk_calls, 'gpu_gemm=', gpu_gemm_calls
      write(iw,'(1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
        '[MOZYME GPU density] skipped_diag=', skipped_diag_blocks, &
        'skipped_offdiag=', skipped_offdiag_blocks, 'cpu_diag=', cpu_diag_blocks, &
        'cpu_offdiag=', cpu_offdiag_blocks
      write(iw,'(1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
        '[MOZYME GPU density] max_diag=', max_diag_block, 'max_offdiag_j=', max_offdiag_j, &
        'max_offdiag_k=', max_offdiag_k
    end if
#endif
    if (mode == 0 .or. mode == 1) then
      !
      !    FULL DENSITY CALCULATION.  MULTIPLY BY 2 FOR SPIN
      !
      spinfa = 2.d0
    else if (mode ==-1) then
      !
      !   MAKING PARTIAL DENSITY MATRIX. REVERSE SIGN ONCE MORE
      !
      spinfa = -2.d0
    else
      spinfa = 1.d0
    end if
   !
    if (Abs(spinfa - 1.d0) > 1.d-10) then
      p(:) = spinfa * p(:)
    end if
    if (prnt) then
      sum = 0.d0
      if (lijbo) then
        do i = 1, numat
          j = nijbo(i, i) + 1
          if (iorbs(i) > 0) then
            sum = sum + p(j)
          end if
          if (iorbs(i) == 4) then
            sum = sum + p(j+2) + p(j+5) + p(j+9)
          end if
        end do
      else
        do i = 1, numat
          j = ijbo (i, i) + 1
          if (iorbs(i) > 0) then
            sum = sum + p(j)
          end if
          if (iorbs(i) == 4) then
            sum = sum + p(j+2) + p(j+5) + p(j+9)
          end if
        end do
      end if
      write (iw,*) " COMPUTED NUMBER OF ELECTRONS:", sum
    end if
#ifdef GPU
contains
  logical function mozyme_density_batch_gpu_enabled()
    implicit none
    integer :: env_len_local, env_status_local
    character(len=16) :: env_value

    mozyme_density_batch_gpu_enabled = lgpu .and. mozyme_gpu
    if (.not. mozyme_density_batch_gpu_enabled) return
    env_value = ' '
    call get_environment_variable('MOPAC_MOZYME_DENSITY_BATCH_GPU', &
      env_value, length=env_len_local, status=env_status_local)
    if (env_status_local == 0 .and. env_len_local > 0) then
      select case (trim(env_value))
      case ('0', 'off', 'OFF', 'false', 'FALSE', 'no', 'NO')
        mozyme_density_batch_gpu_enabled = .false.
      case default
        mozyme_density_batch_gpu_enabled = .true.
      end select
    end if
  end function mozyme_density_batch_gpu_enabled
#endif
end subroutine density_for_MOZYME
