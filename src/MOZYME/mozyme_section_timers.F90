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

module mozyme_section_timers
  implicit none
  private

  integer, parameter :: max_sections = 128
  integer, parameter :: section_name_len = 48

  logical, save :: env_checked = .false.
  logical, save :: profile_enabled = .false.
  integer, save :: section_count = 0
  character(len=section_name_len), save :: section_names(max_sections) = ' '
  integer(kind=8), save :: section_calls(max_sections) = 0_8
  double precision, save :: section_ms(max_sections) = 0.d0

  ! Whole-run phase tokens: run_mopac starts them right after reading the
  ! input; compfg closes 'run_setup' on its first call, run_mopac closes
  ! 'run_total' before the final report.  -1 = not started / already closed.
  double precision, save, public :: run_setup_token = -1.d0
  double precision, save, public :: run_total_token = -1.d0

  public :: mozyme_section_timer_begin
  public :: mozyme_section_timer_end
  public :: mozyme_section_timer_report
  public :: mozyme_section_timer_report_all
  public :: mozyme_section_timers_enabled
  public :: mozyme_section_timer_add_c

contains

  logical function mozyme_section_timers_enabled()
    implicit none

    if (.not. env_checked) call mozyme_section_timer_init()
    mozyme_section_timers_enabled = profile_enabled
  end function mozyme_section_timers_enabled

  subroutine mozyme_section_timer_begin(name, token)
    implicit none
    character(len=*), intent(in) :: name
    double precision, intent(out) :: token

    token = -1.d0
    if (.not. mozyme_section_timers_enabled()) return
    token = wall_seconds()
  end subroutine mozyme_section_timer_begin

  ! Wall clock, not cpu_time: GPU stages that wait asynchronously would be undercounted otherwise.
  double precision function wall_seconds()
    implicit none
    integer(kind=8) :: count, rate

    call system_clock(count, rate)
    wall_seconds = dble(count) / dble(max(rate, 1_8))
  end function wall_seconds

  subroutine mozyme_section_timer_end(name, token)
    implicit none
    character(len=*), intent(in) :: name
    double precision, intent(in) :: token

    integer :: idx
    double precision :: now, elapsed_ms

    if (.not. profile_enabled) return
    if (token < 0.d0) return

    now = wall_seconds()
    elapsed_ms = max(0.d0, now - token) * 1000.d0
    idx = mozyme_section_index(name, .true.)
    if (idx <= 0) return

    section_calls(idx) = section_calls(idx) + 1_8
    section_ms(idx) = section_ms(idx) + elapsed_ms
  end subroutine mozyme_section_timer_end

  subroutine mozyme_section_timer_report(name)
    use chanel_C, only: iw
    implicit none
    character(len=*), intent(in) :: name

    integer :: idx
    character(len=32) :: ms_text

    if (.not. mozyme_section_timers_enabled()) return
    idx = mozyme_section_index(name, .false.)
    if (idx <= 0) return
    if (section_calls(idx) <= 0_8) return

    write(ms_text, '(f16.3)') section_ms(idx)
    ms_text = adjustl(ms_text)
    write(iw,'(a,1x,a,1x,a,i0,1x,a,a)') '[PROFILE] MOZYME_SECTION', &
      'name='//trim(section_names(idx)), 'calls=', section_calls(idx), &
      'ms=', trim(ms_text)
    call flush(iw)
  end subroutine mozyme_section_timer_report

  subroutine mozyme_section_timer_report_all()
    implicit none
    integer :: idx

    if (.not. mozyme_section_timers_enabled()) return
    do idx = 1, section_count
      call mozyme_section_timer_report(section_names(idx))
    end do
  end subroutine mozyme_section_timer_report_all

  subroutine mozyme_section_timer_init()
    implicit none

    profile_enabled = env_enabled('MOPAC_MOZYME_SECTION_PROFILE') .or. &
      env_enabled('MOPAC_MOZYME_PROFILE') .or. &
      env_enabled('MOPAC_GPU_PROFILE')
    env_checked = .true.
  end subroutine mozyme_section_timer_init

  logical function env_enabled(var_name)
    implicit none
    character(len=*), intent(in) :: var_name

    integer :: env_len, env_status
    character(len=32) :: env_value
    character(len=32) :: value

    env_enabled = .false.
    env_value = ' '
    call get_environment_variable(var_name, env_value, length=env_len, &
      status=env_status)
    if (env_status /= 0 .or. env_len <= 0) return

    value = adjustl(env_value)
    select case (trim(value))
    case ('0', 'f', 'F', 'false', 'FALSE', 'False', &
          'n', 'N', 'no', 'NO', 'No', 'off', 'OFF', 'Off')
      env_enabled = .false.
    case default
      env_enabled = .true.
    end select
  end function env_enabled

  ! C-callable accumulator (used by the CUDA driver): adds one call of `ms`
  ! milliseconds to the section `name(1:name_len)`.
  subroutine mozyme_section_timer_add_c(name, name_len, ms) bind(C, name='mozyme_section_timer_add_c')
    use iso_c_binding, only : c_char, c_int, c_double
    implicit none
    character(kind=c_char), intent(in) :: name(*)
    integer(c_int), value :: name_len
    real(c_double), value :: ms
    character(len=section_name_len) :: fname
    integer :: i, n, idx

    if (.not. mozyme_section_timers_enabled()) return
    fname = ' '
    n = min(int(name_len), section_name_len)
    do i = 1, n
      fname(i:i) = name(i)
    end do
    idx = mozyme_section_index(fname, .true.)
    if (idx <= 0) return
    section_calls(idx) = section_calls(idx) + 1_8
    section_ms(idx) = section_ms(idx) + ms
  end subroutine mozyme_section_timer_add_c

  integer function mozyme_section_index(name, create_if_missing)
    implicit none
    character(len=*), intent(in) :: name
    logical, intent(in) :: create_if_missing

    integer :: idx
    character(len=section_name_len) :: stored_name

    stored_name = adjustl(name)
    do idx = 1, section_count
      if (trim(section_names(idx)) == trim(stored_name)) then
        mozyme_section_index = idx
        return
      end if
    end do

    if (.not. create_if_missing .or. section_count >= max_sections) then
      mozyme_section_index = 0
      return
    end if

    section_count = section_count + 1
    section_names(section_count) = stored_name
    section_calls(section_count) = 0_8
    section_ms(section_count) = 0.d0
    mozyme_section_index = section_count
  end function mozyme_section_index

end module mozyme_section_timers
