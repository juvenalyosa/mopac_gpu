// Molecular Orbital PACkage (MOPAC)
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
