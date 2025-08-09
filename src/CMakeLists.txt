# Molecular Orbital PACkage (MOPAC)
# Copyright 2021 Virginia Polytechnic Institute and State University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#===============================================
# source files for the executable stubs
target_sources(mopac PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/mopac.F90)
target_sources(mopac-param PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/param.F90)
#===============================================
# Take a list of source file names
set(main_src_list 
    conref_C run_mopac funcon_C
    molkst_C setup_mopac_arrays overlaps_C 
    molmec_C Common_arrays_C elemts_C run_param
    mopend moldat chanel_C compfg
   )

# Add GPU source files if GPU option is enabled
if(GPU)
    # Add GPU-related Fortran modules and CUDA interop wrappers
    list(APPEND main_src_list
        deprecated/mod_gpu_info
        deprecated/mod_vars_cuda
        deprecated/pulay_for_gpu
        matrix/eigenvectors_cuda
        matrix/call_rot_cuda
        matrix/density_cuda_i
        matrix/mamult_cuda_i
    )
    # Add CUDA C++ wrappers for cuBLAS/cuSOLVER interop
    target_sources(mopac-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cuda_wrappers.cu)
    # Minimal Fortran interfaces for cuBLAS wrappers
    target_sources(mopac-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cublas_interfaces.F90)
endif()

# Small API control module is added from src/interface/CMakeLists.txt

#-----------------------------------------------
# Add a list of source files to the target
foreach(idx IN LISTS main_src_list)
	target_sources(mopac-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${idx}.F90)
endforeach()

# Set language for GPU source files
if(GPU)
    # Ensure Fortran files still compile as Fortran (no CUDA Fortran required)
    # and compile the CUDA .cu wrapper via CUDA language
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cuda_wrappers.cu
        PROPERTIES LANGUAGE CUDA
    )
endif()

#===============================================
# Make a list of subdirectories to be added to the build structure  
set(dir_list
    INDO MOZYME PARAM SCF bz chemistry corrections forces geometry
    input integrals interface makpol matrix meci models optimization
    output potentials properties reactions solvation symmetry utilities
   )
#-----------------------------------------------
# Add the subdirectories to the build directory
foreach(idx IN LISTS dir_list)
	add_subdirectory(${idx})
endforeach()
if(MDI)
  add_subdirectory(mdi)
endif()
#===============================================
