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
  end interface

end module gpu_hmtr_interfaces
