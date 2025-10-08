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

#Basic input/output comparison tests

set(HOF_ERROR 0.01) # Default numerical threshold to flag HOF errors

macro(add_mopac_test _name _files)
  # _files is not a list, it's a string... Transform it into a list
  set(files)
  string(REPLACE ";" " " _files "${_files}")
  foreach(_label "${_files}")
    list(APPEND files ${_label})
  endforeach()
  unset(_files)

  add_test(NAME "${_name}" COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/run_test.py ${CMAKE_CURRENT_SOURCE_DIR}
           ${CMAKE_BINARY_DIR}/mopac${CMAKE_EXECUTABLE_SUFFIX} ${HOF_ERROR} ${_files})
endmacro()

target_sources(mopac-api-test PRIVATE mopac_api_test.F90)
if(NOT TARGET mopac-gpu-grad-test)
  add_executable(mopac-gpu-grad-test gpu_grad_compare.F90)
  target_link_libraries(mopac-gpu-grad-test PRIVATE mopac-core)
endif()
add_test(NAME "gpu-grad-compare" COMMAND mopac-gpu-grad-test)
if(NOT TARGET mopac-gpu-grad-ho-test)
  add_executable(mopac-gpu-grad-ho-test gpu_grad_compare_HO.F90)
  target_link_libraries(mopac-gpu-grad-ho-test PRIVATE mopac-core)
endif()
add_test(NAME "gpu-grad-ho-compare" COMMAND mopac-gpu-grad-ho-test)
if(NOT TARGET mopac-gpu-grad-oo-test)
  add_executable(mopac-gpu-grad-oo-test gpu_grad_compare_OO.F90)
  target_link_libraries(mopac-gpu-grad-oo-test PRIVATE mopac-core)
endif()
add_test(NAME "gpu-grad-oo-compare" COMMAND mopac-gpu-grad-oo-test)
if(NOT TARGET mopac-gpu-grad-ch4-test)
  add_executable(mopac-gpu-grad-ch4-test gpu_grad_compare_CH4.F90)
  target_link_libraries(mopac-gpu-grad-ch4-test PRIVATE mopac-core)
endif()
add_test(NAME "gpu-grad-ch4-compare" COMMAND mopac-gpu-grad-ch4-test)
if(NOT TARGET mopac-gpu-grad-nh3-test)
  add_executable(mopac-gpu-grad-nh3-test gpu_grad_compare_NH3.F90)
  target_link_libraries(mopac-gpu-grad-nh3-test PRIVATE mopac-core)
endif()
add_test(NAME "gpu-grad-nh3-compare" COMMAND mopac-gpu-grad-nh3-test)
if(NOT TARGET mopac-gpu-grad-h2o2-test)
  add_executable(mopac-gpu-grad-h2o2-test gpu_grad_compare_H2O2.F90)
  target_link_libraries(mopac-gpu-grad-h2o2-test PRIVATE mopac-core)
endif()
add_test(NAME "gpu-grad-h2o2-compare" COMMAND mopac-gpu-grad-h2o2-test)
if(NOT TARGET mopac-gpu-grad-no-uhf-test)
  add_executable(mopac-gpu-grad-no-uhf-test gpu_grad_compare_NO_UHF.F90)
  target_link_libraries(mopac-gpu-grad-no-uhf-test PRIVATE mopac-core)
endif()
add_test(NAME "gpu-grad-no-uhf-compare" COMMAND mopac-gpu-grad-no-uhf-test)
if(NOT TARGET mopac-gpu-scf-compare)
  add_executable(mopac-gpu-scf-compare gpu_scf_compare.F90)
  target_link_libraries(mopac-gpu-scf-compare PRIVATE mopac-core)
endif()
add_test(NAME "gpu-scf-compare" COMMAND mopac-gpu-scf-compare)
set_tests_properties(gpu-scf-compare PROPERTIES ENVIRONMENT "MOPAC_FORCEGPU=1;MOPAC_GPU_SCF_EXPERIMENTAL=on")
if(NOT TARGET mopac-gpu-scf-ch4)
  add_executable(mopac-gpu-scf-ch4 gpu_scf_compare_CH4.F90)
  target_link_libraries(mopac-gpu-scf-ch4 PRIVATE mopac-core)
endif()
add_test(NAME "gpu-scf-ch4" COMMAND mopac-gpu-scf-ch4)
set_tests_properties(gpu-scf-ch4 PROPERTIES ENVIRONMENT "MOPAC_FORCEGPU=1;MOPAC_GPU_SCF_EXPERIMENTAL=on")
if(NOT TARGET mopac-gpu-scf-nh3)
  add_executable(mopac-gpu-scf-nh3 gpu_scf_compare_NH3.F90)
  target_link_libraries(mopac-gpu-scf-nh3 PRIVATE mopac-core)
endif()
add_test(NAME "gpu-scf-nh3" COMMAND mopac-gpu-scf-nh3)
set_tests_properties(gpu-scf-nh3 PROPERTIES ENVIRONMENT "MOPAC_FORCEGPU=1;MOPAC_GPU_SCF_EXPERIMENTAL=on")
if(NOT TARGET mopac-gpu-scf-h2o2)
  add_executable(mopac-gpu-scf-h2o2 gpu_scf_compare_H2O2.F90)
  target_link_libraries(mopac-gpu-scf-h2o2 PRIVATE mopac-core)
endif()
add_test(NAME "gpu-scf-h2o2" COMMAND mopac-gpu-scf-h2o2)
set_tests_properties(gpu-scf-h2o2 PROPERTIES ENVIRONMENT "MOPAC_FORCEGPU=1;MOPAC_GPU_SCF_EXPERIMENTAL=on")
if(NOT TARGET mopac-gpu-scf-no-uhf)
  add_executable(mopac-gpu-scf-no-uhf gpu_scf_compare_NO_UHF.F90)
  target_link_libraries(mopac-gpu-scf-no-uhf PRIVATE mopac-core)
endif()
add_test(NAME "gpu-scf-no-uhf" COMMAND mopac-gpu-scf-no-uhf)
set_tests_properties(gpu-scf-no-uhf PROPERTIES ENVIRONMENT "MOPAC_FORCEGPU=1;MOPAC_GPU_SCF_EXPERIMENTAL=on")

if(NOT TARGET mopac-gpu-scf-stub-smoke)
  add_executable(mopac-gpu-scf-stub-smoke gpu_scf_stub_smoke.F90)
  target_link_libraries(mopac-gpu-scf-stub-smoke PRIVATE mopac-core)
endif()
add_test(NAME "gpu-scf-stub-smoke" COMMAND mopac-gpu-scf-stub-smoke)
set_tests_properties(gpu-scf-stub-smoke PROPERTIES ENVIRONMENT "MOPAC_GPU_SCF_STUB_LOG=1;MOPAC_GPU_SCF_EXPERIMENTAL=off")

add_test(NAME "gpu-scf-disk-fallback"
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/check_gpu_scf_fallback.py
          $<TARGET_FILE:mopac>)
set_tests_properties(gpu-scf-disk-fallback PROPERTIES LABELS "gpu")

if(NOT TARGET mopac-gpu-fock-pair-compare)
  add_executable(mopac-gpu-fock-pair-compare gpu_fock_pair_compare.F90)
  target_link_libraries(mopac-gpu-fock-pair-compare PRIVATE mopac-core)
endif()
add_test(NAME "gpu-fock-pair-compare" COMMAND mopac-gpu-fock-pair-compare)

if(GPU)
  if(NOT TARGET mopac-gpu-numeric-reference)
    add_executable(mopac-gpu-numeric-reference gpu_numeric_reference.F90)
    target_link_libraries(mopac-gpu-numeric-reference PRIVATE mopac-core)
  endif()
  add_test(NAME "gpu-numeric-reference" COMMAND mopac-gpu-numeric-reference)
  set_tests_properties(gpu-numeric-reference PROPERTIES
    DISABLED TRUE
    ENVIRONMENT "MOPAC_GPU_EXACT_SC=1")
endif()

add_subdirectory(INDO-dev)
add_subdirectory(keywords)

add_mopac_test(port "port.mop;port.parameters;try.txt;aa.txt;mol.in;SETUP")
add_mopac_test(crambin "Crambin_1SCF.mop")
add_mopac_test(mozyme "test_Lewis_for_Proteins.mop")
add_mopac_test(xeno "Test_XENO.mop")
add_mopac_test(hetero_group "Test_Hetero_group.mop")
add_mopac_test(convert_edited_pdb "convert_an_edited_PDB_file.mop")
add_mopac_test(convert_unedited_pdb "convert_an_un-edited_PDB_file.mop")
add_mopac_test(ters_preserved "Check_that_TERs_are_preserved.mop;Check_TERs.pdb")
add_mopac_test(compare_proteins "Compare_two_protein_structures.mop;Structure-1.txt;Structure-2.txt")
add_mopac_test(pdb1cbn "PDB1CBN.ent")
add_mopac_test(pdb1ejg "PDB1EJG.ent")
add_mopac_test(add-h "ADD-H_Bacteriorhodopsin_1AP9.mop;Bacteriorhodopsin_1AP9.ent")
add_mopac_test(add-h2 "ADD-H_Bacteriorhodopsin_1AT9.mop;Bacteriorhodopsin_1AT9.ent")
add_mopac_test(add-h3 "Au_Add-H.mop")
add_mopac_test(compare_add-h "Compare_ADD-H_1EJG_and_1CBN.mop")
add_mopac_test(pdb_compare "Compare_PDB_1EJG_and_1CBN.mop")
set_tests_properties(compare_add-h pdb_compare PROPERTIES DEPENDS "pdb1cbn;pdb1ejg") # uses PDB1EJG.arc/ent & PDB1CBN.arc/ent
add_mopac_test(pdb_compare2 "Compare_PDB_Bacteriorhodopsin_1AP9_and_2AT9.mop;Bacteriorhodopsin_2AT9.ent")
add_mopac_test(pdb_compare3 "Compare_PDB_Bacteriorhodopsin_1AP9_and_1C3W.mop;Bacteriorhodopsin_1C3W.ent")
add_mopac_test(pdb_compare4 "Compare_PDB_Bacteriorhodopsin_1AP9_and_1BRX.mop;Bacteriorhodopsin_1BRX.ent")
add_mopac_test(pdb_compare5 "Compare_PDB_Bacteriorhodopsin_1AP9_and_1BRR.mop;Bacteriorhodopsin_1BRR.ent")
add_mopac_test(pdb_compare6 "Compare_PDB_Bacteriorhodopsin_1AP9_and_1AT9.mop")
set_tests_properties(pdb_compare2 pdb_compare3 pdb_compare4 pdb_compare5 pdb_compare6 PROPERTIES DEPENDS "add-h;add-h2") # uses ADD-H_Bacteriorhodopsin_1AP9.arc & ADD-H_Bacteriorhodopsin_1AT9.arc or Bacteriorhodopsin_1AP9.ent & Bacteriorhodopsin_1AT9.ent
add_mopac_test(locate_ts "Locate_TS.mop;Cyclobutadiene_1.arc;Cyclobutadiene_2.arc")
add_mopac_test(ionize "IONIZE.mop;Neutral.mop")
add_mopac_test(salt "SALT.mop")
set_tests_properties(salt PROPERTIES DEPENDS ionize) # uses Neutral.mop
add_mopac_test(salt2 "SALT_2.mop")
add_mopac_test(residues "SITE_Individual_residues.mop")
set_tests_properties(residues PROPERTIES DEPENDS ionize) # uses IONIZE.arc

set(HOF_ERROR 1.0) # Raise threshold, these tests are only looking for crashes
add_mopac_test(erratic-1 "erratic1.mop")
add_mopac_test(erratic-2 "erratic2.mop")
add_mopac_test(erratic-3 "erratic3.mop")

add_test(NAME "mopac-api-test" COMMAND mopac-api-test)
