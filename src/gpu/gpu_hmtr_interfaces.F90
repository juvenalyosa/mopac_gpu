! Molecular Orbital PACkage (MOPAC)
! Copyright 2025-2026 Juvenal Yosa, PhD (juvenal.yosa@gmail.com)
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

module gpu_hmtr_interfaces
  use iso_c_binding
  implicit none

  type, bind(C) :: mopac_hmtr_config
     integer(c_int) :: torsion_dim
     integer(c_int) :: population_size
     integer(c_int) :: wrap_angles
     real(c_double) :: inertia
     real(c_double) :: cognitive
     real(c_double) :: social
     real(c_double) :: max_velocity
  end type mopac_hmtr_config

  interface
    function mopac_cuda_hmtr_configure(cfg) &
         bind(C, name='mopac_cuda_hmtr_configure') result(code)
      import :: mopac_hmtr_config, c_int
      type(mopac_hmtr_config), intent(in) :: cfg
      integer(c_int) :: code
    end function mopac_cuda_hmtr_configure

    function mopac_cuda_hmtr_upload_population(torsions, velocities, pbest) &
         bind(C, name='mopac_cuda_hmtr_upload_population') result(code)
      import :: c_double, c_int
      real(c_double), intent(in) :: torsions(*)
      real(c_double), intent(in) :: velocities(*)
      real(c_double), intent(in) :: pbest(*)
      integer(c_int) :: code
    end function mopac_cuda_hmtr_upload_population

    function mopac_cuda_hmtr_set_gbest(gbest) &
         bind(C, name='mopac_cuda_hmtr_set_gbest') result(code)
      import :: c_double, c_int
      real(c_double), intent(in) :: gbest(*)
      integer(c_int) :: code
    end function mopac_cuda_hmtr_set_gbest

    function mopac_cuda_hmtr_pso_step(rand1, rand2) &
         bind(C, name='mopac_cuda_hmtr_pso_step') result(code)
      import :: c_double, c_int
      real(c_double), intent(in) :: rand1(*)
      real(c_double), intent(in) :: rand2(*)
      integer(c_int) :: code
    end function mopac_cuda_hmtr_pso_step

    function mopac_cuda_hmtr_download_population(torsions, velocities) &
         bind(C, name='mopac_cuda_hmtr_download_population') result(code)
      import :: c_double, c_int
      real(c_double), intent(out) :: torsions(*)
      real(c_double), intent(out) :: velocities(*)
      integer(c_int) :: code
    end function mopac_cuda_hmtr_download_population

    subroutine mopac_cuda_hmtr_release() bind(C, name='mopac_cuda_hmtr_release')
    end subroutine mopac_cuda_hmtr_release

    subroutine mopac_cuda_hmtr_bind_thread(device, thread_id, stream_ptr, device_changed) &
         bind(C, name='mopac_cuda_hmtr_bind_thread')
      import :: c_int, c_ptr
      integer(c_int), value :: device
      integer(c_int), value :: thread_id
      type(c_ptr), intent(out) :: stream_ptr
      integer(c_int), intent(out) :: device_changed
    end subroutine mopac_cuda_hmtr_bind_thread

    subroutine mopac_cuda_hmtr_clear_streams() bind(C, name='mopac_cuda_hmtr_clear_streams')
    end subroutine mopac_cuda_hmtr_clear_streams
  end interface

end module gpu_hmtr_interfaces
