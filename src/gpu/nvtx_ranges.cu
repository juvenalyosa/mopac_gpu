// Molecular Orbital PACkage (MOPAC)
// Copyright 2021 Virginia Polytechnic Institute and State University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// NVTX ranges for the MOZYME section timers (MOPAC_NVTX=1, src/MOZYME/mozyme_section_timers.F90), so that
// Nsight Systems timelines show which part of MOPAC runs while the GPU waits.  NVTX 3 is header-only (CUDA 12);
// without the header the calls do nothing.

#include <string>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define MOPAC_HAVE_NVTX 1
#endif
#endif

extern "C" void mopac_nvtx_push(const char *name, int name_len) {
#ifdef MOPAC_HAVE_NVTX
  std::string s(name, name_len > 0 ? static_cast<size_t>(name_len) : 0);
  nvtxRangePushA(s.c_str());
#else
  (void)name;
  (void)name_len;
#endif
}

extern "C" void mopac_nvtx_pop() {
#ifdef MOPAC_HAVE_NVTX
  nvtxRangePop();
#endif
}
