module gpu_info
          use iso_c_binding
          implicit none
            interface
                subroutine gpuInfo(hasGpu, hasDouble, nDevices, name,name_size, totalMem, clockRate, major, minor) &
                bind(c, name="getGPUInfo")
                  import :: c_bool, c_int, c_char, c_size_t
                  logical(c_bool)                :: hasGpu
                  integer(c_int)                 :: nDevices
                  logical(c_bool),dimension(6)   :: hasDouble
                  integer(c_int),dimension(6)    :: clockRate, major, minor, name_size
                  character(kind=c_char),dimension(6) :: name
                  integer(c_size_t),dimension(6) :: totalMem
                end subroutine
            end interface
        end module
        
! ******************************

        module settingGPUcard
          use iso_c_binding
          implicit none
            interface setDevice_C
                subroutine setGPU(idevice, stat) bind(c, name='setDevice')
                  import :: c_bool, c_int
                  logical(c_bool)           :: stat
                  integer(c_int), value     :: idevice
                end subroutine
            end interface
            interface setPair_C
                subroutine setMGpuPair(dev0, dev1) bind(c, name='set_mozyme_gpu_pair')
                  import :: c_int
                  integer(c_int), value     :: dev0
                  integer(c_int), value     :: dev1
                end subroutine
            end interface
        end module
