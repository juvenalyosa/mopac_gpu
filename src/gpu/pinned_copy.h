// Molecular Orbital PACkage (MOPAC)
// Copyright 2025-2026 Juvenal Yosa, PhD (juvenal.yosa@gmail.com)
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

// Large host <-> device copies through pinned double-buffered staging (mozyme_scf_context.cu).
// Pageable cudaMemcpy of the mpack / norbs^2 arrays of a solid ran at ~1.5 GB/s on an A100.
// The copies run on a blocking stream: synchronize any non-default stream that produced the
// device data before mopac_pinned_d2h.  Both return 1 once the data has arrived.
#pragma once
#include <cstddef>

extern "C" int mopac_pinned_d2h(void *host, const void *device, size_t bytes);
extern "C" int mopac_pinned_h2d(void *device, const void *host, size_t bytes);
