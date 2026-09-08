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

module mozyme_fock1_batch
  use iso_c_binding, only: c_int
  use mozyme_gpu_int_utils, only: mozyme_c_int_positive_or_zero
  implicit none
  integer(c_int), allocatable, save :: task_f_offset(:)
  integer(c_int), allocatable, save :: task_w_offset(:)
  integer(c_int), allocatable, save :: task_iab(:)
  integer(c_int), allocatable, save :: task_ilim(:)
  integer(c_int), save :: task_count = 0_c_int
  integer, save :: batch_chunk_size = 64
contains

  logical function mozyme_fock1_batch_enabled()
#ifdef GPU
    use mod_vars_cuda, only: mozyme_gpu_requested, mozyme_fock1_batch_gpu
#endif
    implicit none
#ifdef GPU
    mozyme_fock1_batch_enabled = mozyme_gpu_requested .and. mozyme_fock1_batch_gpu
#else
    mozyme_fock1_batch_enabled = .false.
#endif
  end function mozyme_fock1_batch_enabled

  subroutine mozyme_fock1_batch_begin(max_tasks)
    implicit none
    integer, intent(in) :: max_tasks
    integer :: capacity
    capacity = max(1, max_tasks)
    if (allocated(task_f_offset)) then
      if (size(task_f_offset) < capacity) call mozyme_fock1_batch_release()
    end if
    if (.not. allocated(task_f_offset)) then
      allocate(task_f_offset(capacity), task_w_offset(capacity), task_iab(capacity), task_ilim(capacity))
    end if
    task_count = 0_c_int
    batch_chunk_size = mozyme_fock1_batch_chunk_size(capacity)
  end subroutine mozyme_fock1_batch_begin

  subroutine mozyme_fock1_batch_release()
    implicit none
    if (allocated(task_f_offset)) deallocate(task_f_offset)
    if (allocated(task_w_offset)) deallocate(task_w_offset)
    if (allocated(task_iab)) deallocate(task_iab)
    if (allocated(task_ilim)) deallocate(task_ilim)
    task_count = 0_c_int
  end subroutine mozyme_fock1_batch_release

  logical function mozyme_fock1_batch_add(f_offset, w_offset, iab, ilim)
    implicit none
    integer, intent(in) :: f_offset, w_offset, iab, ilim
    mozyme_fock1_batch_add = .false.
    if (.not. allocated(task_f_offset)) return
    if (task_count >= size(task_f_offset)) return
    task_count = task_count + 1_c_int
    task_f_offset(task_count) = mozyme_c_int_positive_or_zero(f_offset)
    task_w_offset(task_count) = mozyme_c_int_positive_or_zero(w_offset)
    task_iab(task_count) = mozyme_c_int_positive_or_zero(iab)
    task_ilim(task_count) = mozyme_c_int_positive_or_zero(ilim)
    mozyme_fock1_batch_add = .true.
  end function mozyme_fock1_batch_add

  subroutine mozyme_fock1_batch_record_or_run(f, ptot, w, kr, f_offset, w_offset, iab, ilim)
    implicit none
    double precision, intent(inout) :: f(*)
    double precision, intent(in) :: ptot(*), w(*)
    integer, intent(inout) :: kr
    integer, intent(in) :: f_offset, w_offset, iab, ilim
    logical :: recorded
    external :: fock1_for_MOZYME

    if (mozyme_fock1_batch_enabled()) then
      recorded = mozyme_fock1_batch_add(f_offset, w_offset, iab, ilim)
      if (recorded) then
        kr = kr + ilim ** 2
        if (task_count >= batch_chunk_size) call mozyme_fock1_batch_flush(f, ptot, w)
        return
      end if
    end if
    call fock1_for_MOZYME(f(f_offset), ptot(f_offset), w(w_offset), kr, iab, ilim)
  end subroutine mozyme_fock1_batch_record_or_run

  subroutine mozyme_fock1_batch_flush(f, ptot, w)
#ifdef GPU
    use chanel_C, only: iw
    use gpu_fock_interfaces, only: mopac_cuda_mozyme_fock1_batch
#endif
    implicit none
    double precision, intent(inout) :: f(*)
    double precision, intent(in) :: ptot(*), w(*)
    integer :: code, idx, kr_tmp, total_pairs
    logical :: trace_gpu
    external :: fock1_for_MOZYME

    if (task_count <= 0_c_int) return
    total_pairs = 0
    do idx = 1, task_count
      total_pairs = total_pairs + task_ilim(idx)
    end do

#ifdef GPU
    if (mozyme_fock1_batch_enabled()) then
      trace_gpu = mozyme_fock1_batch_trace()
      if (trace_gpu) then
        write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0)') &
          '[MOZYME GPU fock1_batch]', 'attempt tasks=', task_count, 'pairs=', total_pairs
        call flush(iw)
      end if
      code = mopac_cuda_mozyme_fock1_batch(task_count, task_f_offset, task_w_offset, task_iab, task_ilim, ptot, f, w)
      if (code == 0) then
        if (trace_gpu) then
          write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0)') &
            '[MOZYME GPU fock1_batch]', 'success tasks=', task_count, 'pairs=', total_pairs
          call flush(iw)
        end if
        task_count = 0_c_int
        return
      end if
      if (trace_gpu) then
        write(iw,'(1x,a,1x,a,1x,i0,1x,a,1x,i0,1x,a,1x,i0)') &
          '[MOZYME GPU fock1_batch]', 'fallback code=', code, 'tasks=', task_count, 'pairs=', total_pairs
        call flush(iw)
      end if
    end if
#endif

    do idx = 1, task_count
      kr_tmp = 0
      call fock1_for_MOZYME(f(task_f_offset(idx)), ptot(task_f_offset(idx)), &
        w(task_w_offset(idx)), kr_tmp, task_iab(idx), task_ilim(idx))
    end do
    task_count = 0_c_int
  end subroutine mozyme_fock1_batch_flush

  logical function mozyme_fock1_batch_trace()
    implicit none
    integer :: env_len, env_status
    character(len=16) :: env_value
    mozyme_fock1_batch_trace = .false.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_PROFILE', env_value, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') mozyme_fock1_batch_trace = .true.
    env_value = ' '
    call get_environment_variable('MOPAC_GPU_VERBOSE', env_value, length=env_len, status=env_status)
    if (env_status == 0 .and. env_len > 0 .and. trim(env_value) /= '0') mozyme_fock1_batch_trace = .true.
  end function mozyme_fock1_batch_trace

  integer function mozyme_fock1_batch_chunk_size(capacity)
    implicit none
    integer, intent(in) :: capacity
    integer :: env_status, value
    character(len=32) :: env_value
    mozyme_fock1_batch_chunk_size = min(max(1, capacity), 64)
    env_value = ' '
    call get_environment_variable('MOPAC_MOZYME_FOCK1_BATCH_CHUNK', env_value, status=env_status)
    if (env_status == 0 .and. len_trim(env_value) > 0) then
      read(env_value, *, err=100, end=100) value
      mozyme_fock1_batch_chunk_size = min(max(1, value), max(1, capacity))
    end if
100 continue
  end function mozyme_fock1_batch_chunk_size

end module mozyme_fock1_batch
