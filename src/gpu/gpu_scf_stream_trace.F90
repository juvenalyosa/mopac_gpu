! Developed by Dr. Juvenal Yosa Reyes, UMCG Groningen, Universidad Simon Bolivar - Barranquilla - Colombia
module gpu_scf_stream_trace
  implicit none
  private
  logical :: trace_initialized = .false.
  logical :: trace_enabled = .false.
  integer :: trace_unit = -1
  public :: gpu_stream_trace_block
contains
  subroutine gpu_stream_trace_block(tag, ia, ib, ja, jb, len)
    character(len=*), intent(in) :: tag
    integer, intent(in) :: ia, ib, ja, jb, len
    if (.not. trace_initialized) call gpu_stream_trace_init()
    if (.not. trace_enabled) return
    if (trace_unit < 0) then
      trace_unit = 97
      open(unit=trace_unit, file='gpu_stream_trace.log', status='unknown', position='append', action='write')
    end if
    write(trace_unit,'(a,1x,4i8,1x,i8)') trim(tag), ia, ib, ja, jb, len
    flush(trace_unit)
  end subroutine gpu_stream_trace_block

  subroutine gpu_stream_trace_init()
    character(len=32) :: env
    integer :: stat
    trace_initialized = .true.
    trace_enabled = .false.
    env = ''
    stat = 1
    call get_environment_variable('MOPAC_GPU_STREAM_TRACE', env, status=stat)
    if (stat == 0) then
      env = adjustl(env)
      if (len_trim(env) > 0) then
        select case (env(1:1))
        case('0','n','N','f','F')
          trace_enabled = .false.
        case default
          trace_enabled = .true.
        end select
      end if
    end if
  end subroutine gpu_stream_trace_init
end module gpu_scf_stream_trace
