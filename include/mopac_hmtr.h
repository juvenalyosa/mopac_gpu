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

// Hierarchical memetic trust-region optimizer (HMTR) GPU interface

#pragma once

#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int torsion_dim;
  int population_size;
  int wrap_angles;
  double inertia;
  double cognitive;
  double social;
  double max_velocity;
} MopacHmtrConfig;

int mopac_cuda_hmtr_configure(const MopacHmtrConfig* cfg);

int mopac_cuda_hmtr_upload_population(const double* torsions,
                                      const double* velocities,
                                      const double* pbest);

int mopac_cuda_hmtr_set_gbest(const double* gbest);

int mopac_cuda_hmtr_pso_step(const double* rand1,
                             const double* rand2);

int mopac_cuda_hmtr_download_population(double* torsions,
                                        double* velocities);

void mopac_cuda_hmtr_release(void);

void mopac_cuda_hmtr_bind_thread(int device,
                                 int thread_id,
                                 void **stream_out,
                                 int *device_changed);

void mopac_cuda_hmtr_clear_streams(void);

#ifdef __cplusplus
}
#endif
