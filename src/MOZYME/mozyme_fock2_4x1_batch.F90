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

module mozyme_fock2_4x1_batch
  use iso_c_binding, only: c_int
  use mozyme_gpu_int_utils, only: mozyme_c_int_positive_or_zero
  implicit none
  integer(c_int), allocatable, save :: task_heavy_offset(:)
  integer(c_int), allocatable, save :: task_light_offset(:)
  integer(c_int), allocatable, save :: task_cross_offset(:)
  double precision, allocatable, save :: task_wj(:,:)
  double precision, allocatable, save :: task_wk(:,:)
  integer(c_int), save :: task_count = 0_c_int
  integer, save :: batch_chunk_size = 65536
  logical, save :: gpu_failed_this_run = .false.
contains

  logical function mozyme_fock2_4x1_batch_enabled()
#ifdef GPU
    use mod_vars_cuda, only: mozyme_gpu_requested, mozyme_fock2_4x1_batch_gpu
#endif
    implicit none
#ifdef GPU
    mozyme_fock2_4x1_batch_enabled = mozyme_gpu_requested .and. mozyme_fock2_4x1_batch_gpu &
      .and. .not. gpu_failed_this_run
#else
    mozyme_fock2_4x1_batch_enabled = .false.
#endif
  end function mozyme_fock2_4x1_batch_enabled

  subroutine mozyme_fock2_4x1_batch_begin(max_tasks)
    implicit none
    integer, intent(in) :: max_tasks
    integer :: capacity

    batch_chunk_size = mozyme_fock2_4x1_batch_chunk_size(max_tasks)
    capacity = max(1, batch_chunk_size)
    if (allocated(task_heavy_offset)) then
      if (size(task_heavy_offset) < capacity) call mozyme_fock2_4x1_batch_release()
    end if
    if (.not. allocated(task_heavy_offset)) then
      allocate(task_heavy_offset(capacity), task_light_offset(capacity), &
        task_cross_offset(capacity), task_wj(10, capacity), task_wk(16, capacity))
    end if
    task_count = 0_c_int
    gpu_failed_this_run = .false.
  end subroutine mozyme_fock2_4x1_batch_begin

  subroutine mozyme_fock2_4x1_batch_release()
    implicit none
    if (allocated(task_heavy_offset)) deallocate(task_heavy_offset)
    if (allocated(task_light_offset)) deallocate(task_light_offset)
    if (allocated(task_cross_offset)) deallocate(task_cross_offset)
    if (allocated(task_wj)) deallocate(task_wj)
    if (allocated(task_wk)) deallocate(task_wk)
    task_count = 0_c_int
  end subroutine mozyme_fock2_4x1_batch_release

  logical function mozyme_fock2_4x1_batch_record(f, ptot, heavy_offset, light_offset, cross_offset, wj_values, wk_values)
    implicit none
    double precision, intent(inout) :: f(*)
    double precision, intent(in) :: ptot(*), wj_values(10), wk_values(16)
    integer, intent(in) :: heavy_offset, light_offset, cross_offset

    mozyme_fock2_4x1_batch_record = .false.
    if (.not. mozyme_fock2_4x1_batch_enabled()) return
    if (.not. allocated(task_heavy_offset)) return
    if (task_count >= size(task_heavy_offset)) call mozyme_fock2_4x1_batch_flush(f, ptot)
    if (task_count >= size(task_heavy_offset)) return

    task_count = task_count + 1_c_int
    task_heavy_offset(task_count) = mozyme_c_int_positive_or_zero(heavy_offset)
    task_light_offset(task_count) = mozyme_c_int_positive_or_zero(light_offset)
    task_cross_offset(task_count) = mozyme_c_int_positive_or_zero(cross_offset)
    task_wj(:, task_count) = wj_values(:)
    task_wk(:, task_count) = wk_values(:)
    mozyme_fock2_4x1_batch_record = .true.
  end function mozyme_fock2_4x1_batch_record

  subroutine mozyme_fock2_4x1_batch_flush(f, ptot)
    use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required
#ifdef GPU
    use chanel_C, only: iw
    use gpu_fock_interfaces, only: mopac_cuda_mozyme_fock2_4x1_batch
#endif
    implicit none
    double precision, intent(inout) :: f(*)
    double precision, intent(in) :: ptot(*)
    integer :: code, idx
    logical :: trace_gpu
    external :: mozyme_gpu_strict_abort

    if (task_count <= 0_c_int) return

#ifdef GPU
    if (mozyme_fock2_4x1_batch_enabled()) then
      trace_gpu = mozyme_fock2_4x1_batch_trace()
      if (trace_gpu) then
        write(iw,'(1x,a,1x,a,1x,i0)') '[MOZYME GPU fock2_4x1_batch]', 'attempt tasks=', task_count
        call flush(iw)
      end if
      code = mopac_cuda_mozyme_fock2_4x1_batch(task_count, task_heavy_offset, task_light_offset, &
        task_cross_offset, task_wj, task_wk, ptot, f)
      if (code == 0) then
        if (trace_gpu) then
          write(iw,'(1x,a,1x,a,1x,i0)') '[MOZYME GPU fock2_4x1_batch]', 'success tasks=', task_count
          call flush(iw)
        end if
        task_count = 0_c_int
        return
      end if
      gpu_failed_this_run = .true.
      if (trace_gpu) then
        write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0)') &
          '[MOZYME GPU fock2_4x1_batch]', 'fallback code=', code, 'tasks=', task_count
        call flush(iw)
      end if
    end if
#endif

    if (mozyme_gpu_scf_no_fallback_required()) then
      call mozyme_gpu_strict_abort('strict_fock2_4x1_batch_cpu_fallback', &
        'MOZYME GPU strict resident SCF does not support CPU 4x1 Fock batch fallback')
      return
    end if

    do idx = 1, task_count
      call mozyme_fock2_4x1_apply_cpu(f, ptot, task_heavy_offset(idx), &
        task_light_offset(idx), task_cross_offset(idx), task_wj(:, idx), task_wk(:, idx))
    end do
    task_count = 0_c_int
  end subroutine mozyme_fock2_4x1_batch_flush

  subroutine mozyme_fock2_4x1_apply_cpu(f, ptot, heavy_offset, light_offset, cross_offset, wj_values, wk_values)
    implicit none
    double precision, intent(inout) :: f(*)
    double precision, intent(in) :: ptot(*), wj_values(10), wk_values(16)
    integer(c_int), intent(in) :: heavy_offset, light_offset, cross_offset
    integer :: i, j, k, li
    double precision :: sum, sumdia, sumoff

    sumdia = 0.d0
    sumoff = 0.d0
    li = heavy_offset - 1
    k = 0
    do i = 1, 4
      do j = 1, i - 1
        li = li + 1
        k = k + 1
        f(li) = f(li) + ptot(light_offset) * wj_values(k)
        sumoff = sumoff + ptot(li) * wj_values(k)
      end do
      li = li + 1
      k = k + 1
      f(li) = f(li) + ptot(light_offset) * wj_values(k)
      sumdia = sumdia + ptot(li) * wj_values(k)
    end do
    f(light_offset) = f(light_offset) + sumoff * 2.d0 + sumdia

    k = 0
    do i = 1, 4
      sum = 0.d0
      do j = 1, 4
        k = k + 1
        sum = sum + ptot(cross_offset + j - 1) * wk_values(k)
      end do
      f(cross_offset + i - 1) = f(cross_offset + i - 1) - sum * 0.5d0
    end do
  end subroutine mozyme_fock2_4x1_apply_cpu

  logical function mozyme_fock2_4x1_batch_trace()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value
    mozyme_fock2_4x1_batch_trace = .false.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_PROFILE', env_value, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') mozyme_fock2_4x1_batch_trace = .true.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_VERBOSE', env_value, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') mozyme_fock2_4x1_batch_trace = .true.
  end function mozyme_fock2_4x1_batch_trace

  integer function mozyme_fock2_4x1_batch_chunk_size(max_tasks)
    implicit none
    integer, intent(in) :: max_tasks
    integer :: env_status, value
    character(len=32) :: env_value
    mozyme_fock2_4x1_batch_chunk_size = min(max(1, max_tasks), 65536)
    env_value = ' '
    call get_environment_variable('MOPAC_MOZYME_FOCK2_4X1_BATCH_CHUNK', env_value, status=env_status)
    if (env_status == 0 .and. len_trim(env_value) > 0) then
      read(env_value, *, err=100, end=100) value
      mozyme_fock2_4x1_batch_chunk_size = min(max(1, value), max(1, max_tasks))
    end if
100 continue
  end function mozyme_fock2_4x1_batch_chunk_size

end module mozyme_fock2_4x1_batch
