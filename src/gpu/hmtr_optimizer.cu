// HMTR optimizer CUDA support: population updates and PSO-style steps

#include "mopac_hmtr.h"

// Developed by Dr. Juvenal Yosa Reyes, UMCG Groningen, Universidad Simon Bolivar - Barranquilla - Colombia
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace {

constexpr double DEG_FULL = 360.0;
constexpr double DEG_HALF = 180.0;

struct DevicePopulation {
  double* torsions = nullptr;
  double* velocities = nullptr;
  double* pbest = nullptr;
  double* gbest = nullptr;
  double* rand1 = nullptr;
  double* rand2 = nullptr;
  size_t capacity = 0;
} g_pop;

MopacHmtrConfig g_cfg{};
bool g_configured = false;

inline int check_cuda(cudaError_t err) {
  if (err == cudaSuccess) return 0;
  std::fprintf(stderr, "[GPU][HMTR] CUDA error: %s\n", cudaGetErrorString(err));
  return 2;
}

void release_population() {
  if (g_pop.torsions) cudaFree(g_pop.torsions);
  if (g_pop.velocities) cudaFree(g_pop.velocities);
  if (g_pop.pbest) cudaFree(g_pop.pbest);
  if (g_pop.gbest) cudaFree(g_pop.gbest);
  if (g_pop.rand1) cudaFree(g_pop.rand1);
  if (g_pop.rand2) cudaFree(g_pop.rand2);
  g_pop = DevicePopulation{};
}

cudaError_t ensure_capacity(size_t need) {
  if (need <= g_pop.capacity) return cudaSuccess;
  release_population();
  size_t bytes = need * sizeof(double);
  if (cudaMalloc(&g_pop.torsions, bytes) != cudaSuccess) return cudaGetLastError();
  if (cudaMalloc(&g_pop.velocities, bytes) != cudaSuccess) return cudaGetLastError();
  if (cudaMalloc(&g_pop.pbest, bytes) != cudaSuccess) return cudaGetLastError();
  if (cudaMalloc(&g_pop.rand1, bytes) != cudaSuccess) return cudaGetLastError();
  if (cudaMalloc(&g_pop.rand2, bytes) != cudaSuccess) return cudaGetLastError();
  if (cudaMalloc(&g_pop.gbest, g_cfg.torsion_dim * sizeof(double)) != cudaSuccess) return cudaGetLastError();
  g_pop.capacity = need;
  return cudaSuccess;
}

__device__ inline double wrap_delta(double x) {
  x = fmod(x + DEG_HALF, DEG_FULL);
  if (x < 0.0) x += DEG_FULL;
  return x - DEG_HALF;
}

__device__ inline double wrap_angle(double x) {
  x = fmod(x, DEG_FULL);
  if (x < 0.0) x += DEG_FULL;
  return x;
}

__global__ void pso_update_kernel(int population,
                                  int dim,
                                  double inertia,
                                  double cognitive,
                                  double social,
                                  double max_velocity,
                                  int wrap_flag,
                                  const double* __restrict__ pbest,
                                  const double* __restrict__ gbest,
                                  const double* __restrict__ rand1,
                                  const double* __restrict__ rand2,
                                  double* torsions,
                                  double* velocities) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = static_cast<size_t>(population) * static_cast<size_t>(dim);
  if (idx >= total) return;
  int coord = static_cast<int>(idx / static_cast<size_t>(population));
  double pos = torsions[idx];
  double vel = velocities[idx];
  double rp = rand1[idx];
  double rg = rand2[idx];
  double pb = pbest[idx];
  double gb = gbest[coord];
  bool wrap = (wrap_flag != 0);
  double delta_p = wrap ? wrap_delta(pb - pos) : (pb - pos);
  double delta_g = wrap ? wrap_delta(gb - pos) : (gb - pos);
  double new_vel = inertia * vel + cognitive * rp * delta_p + social * rg * delta_g;
  if (new_vel > max_velocity) new_vel = max_velocity;
  if (new_vel < -max_velocity) new_vel = -max_velocity;
  double new_pos = wrap ? wrap_angle(pos + new_vel) : (pos + new_vel);
  torsions[idx] = new_pos;
  velocities[idx] = new_vel;
}

} // namespace

extern "C" int mopac_cuda_hmtr_configure(const MopacHmtrConfig* cfg) {
  if (!cfg || cfg->torsion_dim <= 0 || cfg->population_size <= 0) return 1;
  g_cfg = *cfg;
  size_t need = static_cast<size_t>(cfg->torsion_dim) * static_cast<size_t>(cfg->population_size);
  cudaError_t err = ensure_capacity(need);
  if (err != cudaSuccess) {
    release_population();
    g_configured = false;
    return check_cuda(err);
  }
  g_configured = true;
  return 0;
}

extern "C" int mopac_cuda_hmtr_upload_population(const double* torsions,
                                                  const double* velocities,
                                                  const double* pbest) {
  if (!g_configured) return 1;
  size_t total = static_cast<size_t>(g_cfg.population_size) * static_cast<size_t>(g_cfg.torsion_dim);
  size_t bytes = total * sizeof(double);
  if (torsions) {
    if (check_cuda(cudaMemcpy(g_pop.torsions, torsions, bytes, cudaMemcpyHostToDevice))) return 2;
  }
  if (velocities) {
    if (check_cuda(cudaMemcpy(g_pop.velocities, velocities, bytes, cudaMemcpyHostToDevice))) return 2;
  } else {
    if (check_cuda(cudaMemset(g_pop.velocities, 0, bytes))) return 2;
  }
  if (pbest) {
    if (check_cuda(cudaMemcpy(g_pop.pbest, pbest, bytes, cudaMemcpyHostToDevice))) return 2;
  }
  return 0;
}

extern "C" int mopac_cuda_hmtr_set_gbest(const double* gbest) {
  if (!g_configured || !gbest) return 1;
  size_t bytes = static_cast<size_t>(g_cfg.torsion_dim) * sizeof(double);
  return check_cuda(cudaMemcpy(g_pop.gbest, gbest, bytes, cudaMemcpyHostToDevice));
}

extern "C" int mopac_cuda_hmtr_pso_step(const double* rand1,
                                         const double* rand2) {
  if (!g_configured) return 1;
  size_t total = static_cast<size_t>(g_cfg.population_size) * static_cast<size_t>(g_cfg.torsion_dim);
  size_t bytes = total * sizeof(double);
  if (rand1) {
    if (check_cuda(cudaMemcpy(g_pop.rand1, rand1, bytes, cudaMemcpyHostToDevice))) return 2;
  } else {
    if (check_cuda(cudaMemset(g_pop.rand1, 0, bytes))) return 2;
  }
  if (rand2) {
    if (check_cuda(cudaMemcpy(g_pop.rand2, rand2, bytes, cudaMemcpyHostToDevice))) return 2;
  } else {
    if (check_cuda(cudaMemset(g_pop.rand2, 0, bytes))) return 2;
  }
  int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  pso_update_kernel<<<blocks, threads>>>(
      g_cfg.population_size,
      g_cfg.torsion_dim,
      g_cfg.inertia,
      g_cfg.cognitive,
      g_cfg.social,
      g_cfg.max_velocity,
      g_cfg.wrap_angles,
      g_pop.pbest,
      g_pop.gbest,
      g_pop.rand1,
      g_pop.rand2,
      g_pop.torsions,
      g_pop.velocities);
  return check_cuda(cudaGetLastError());
}

extern "C" int mopac_cuda_hmtr_download_population(double* torsions,
                                                    double* velocities) {
  if (!g_configured) return 1;
  size_t total = static_cast<size_t>(g_cfg.population_size) * static_cast<size_t>(g_cfg.torsion_dim);
  size_t bytes = total * sizeof(double);
  if (torsions) {
    if (check_cuda(cudaMemcpy(torsions, g_pop.torsions, bytes, cudaMemcpyDeviceToHost))) return 2;
  }
  if (velocities) {
    if (check_cuda(cudaMemcpy(velocities, g_pop.velocities, bytes, cudaMemcpyDeviceToHost))) return 2;
  }
  return 0;
}

extern "C" void mopac_cuda_hmtr_release(void) {
  release_population();
  g_configured = false;
}
