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

// Two-centre part of fock2 (src/SCF/fock2.F90) for solids (id /= 0, the
// branch that takes the J integrals from wj and the K integrals from wk) on
// the device:
//
//   pairs jj < ii : one warp per pair; the ii-ii triangle, the jj-jj triangle
//                   and the ii-jj block are summed in shared memory and then
//                   added to f (the ii-jj block belongs to the pair alone)
//   pairs ii = ii : an atom with its own images, one thread per atom, the
//                   CPU loop as written (kl > ij skipped, i = k diagonal term)
//
// The one-centre terms (fock1) stay on the host.  w and wk (n2elec each) stay
// on the device between calls with the same generation number (one hcore).
// The device hcore of a solid (mopac_cuda_solrot_pairs) can hand over the w / wk
// it built on the device (mopac_cuda_conv_fock_offer_w), which hcore confirms for
// its generation (mopac_cuda_conv_fock_confirm_w); the first Fock call of that
// generation then takes them instead of uploading the host copies.  When the
// device hcore did not download them either, the host copies of the pair
// segments are stale until mopac_cuda_conv_w_sync_host fetches them (for the
// CPU code that reads w: fock2's CPU path, C.I., ENPART, POLAR, ...).

#include <cuda_runtime.h>

#include "pinned_copy.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

// Fortran section-timer hook (mozyme_section_timers).
extern "C" void mozyme_section_timer_add_c(const char *name, int name_len, double ms);

namespace {

using Clock = std::chrono::steady_clock;
void add_section(const char *name, const Clock::time_point &t0, const Clock::time_point &t1) {
  mozyme_section_timer_add_c(name, static_cast<int>(std::strlen(name)),
                             std::chrono::duration<double, std::milli>(t1 - t0).count());
}

__constant__ unsigned char c_ti[45];
__constant__ unsigned char c_tj[45];

constexpr int kWarpsPerBlock = 4;
constexpr int kShared = 45 + 45 + 81;

struct FockArgs {
  int npairs;               // pairs jj < ii
  const int *pair_ia, *pair_ib, *pair_ja, *pair_jb;
  const long long *pair_off;  // 0-based start of the pair's integrals in w / wk
  int nself;                // pairs ii = jj
  const int *self_ia, *self_ib;
  const long long *self_off;
  const double *w, *wk, *ptot, *p;
  double *f;
};

__device__ __forceinline__ long long tri(long long i) { return (i * (i - 1)) / 2; }

__global__ void fock2_pairs_kernel(FockArgs a) {
  __shared__ double sh[kWarpsPerBlock][kShared];
  const int lane = threadIdx.x & 31;
  const int wib = threadIdx.x >> 5;
  const int pair = blockIdx.x * kWarpsPerBlock + wib;
  if (pair >= a.npairs) return;
  double *fii = sh[wib];
  double *fjj = fii + 45;
  double *fij = fjj + 45;
  for (int t = lane; t < kShared; t += 32) fii[t] = 0.0;
  __syncwarp();
  const int ia = a.pair_ia[pair];
  const int ib = a.pair_ib[pair];
  const int ja = a.pair_ja[pair];
  const int jb = a.pair_jb[pair];
  const int ni = ib - ia + 1;
  const int nj = jb - ja + 1;
  const int nii = (ni * (ni + 1)) / 2;
  const int njj = (nj * (nj + 1)) / 2;
  const long long off = a.pair_off[pair];
  for (int t = lane; t < nii * njj; t += 32) {
    const int ijx = t / njj;
    const int klx = t - ijx * njj;
    const int iloc = c_ti[ijx], jloc = c_tj[ijx], kloc = c_ti[klx], lloc = c_tj[klx];
    const long long i = ia + iloc, j = ia + jloc, k = ja + kloc, l = ja + lloc;
    const double aa = (i == j) ? 1.0 : 2.0;
    const double bb = (k == l) ? 1.0 : 2.0;
    const double aj = a.w[off + t];
    const double ak = a.wk[off + t];
    const long long ij = tri(i) + j, kl = tri(k) + l;
    const long long ik = tri(i) + k, il = tri(i) + l, jk = tri(j) + k, jl = tri(j) + l;
    atomicAdd(&fii[ijx], bb * aj * a.ptot[kl - 1]);
    atomicAdd(&fjj[klx], aa * aj * a.ptot[ij - 1]);
    const double e = ak * aa * bb * 0.25;
    atomicAdd(&fij[iloc * nj + kloc], -e * a.p[jl - 1]);
    atomicAdd(&fij[iloc * nj + lloc], -e * a.p[jk - 1]);
    atomicAdd(&fij[jloc * nj + kloc], -e * a.p[il - 1]);
    atomicAdd(&fij[jloc * nj + lloc], -e * a.p[ik - 1]);
  }
  __syncwarp();
  for (int t = lane; t < nii; t += 32) {
    const long long i = ia + c_ti[t], j = ia + c_tj[t];
    atomicAdd(&a.f[tri(i) + j - 1], fii[t]);
  }
  for (int t = lane; t < njj; t += 32) {
    const long long k = ja + c_ti[t], l = ja + c_tj[t];
    atomicAdd(&a.f[tri(k) + l - 1], fjj[t]);
  }
  for (int t = lane; t < ni * nj; t += 32) {
    const long long i = ia + t / nj, k = ja + t % nj;
    a.f[tri(i) + k - 1] += fij[t];
  }
}

__global__ void fock2_self_kernel(FockArgs a) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= a.nself) return;
  const long long ia = a.self_ia[s], ib = a.self_ib[s];
  long long kk = a.self_off[s];
  for (long long i = ia; i <= ib; ++i) {
    const long long ka = tri(i);
    for (long long j = ia; j <= i; ++j) {
      const long long kb = tri(j);
      const long long ij = ka + j;
      const double aa = (i == j) ? 1.0 : 2.0;
      for (long long k = ia; k <= ib; ++k) {
        const long long kc = tri(k);
        const long long ik = (i >= k) ? ka + k : 0;
        const long long jk = (j >= k) ? kb + k : 0;
        for (long long l = ia; l <= k; ++l) {
          const long long il = (i >= l) ? ka + l : 0;
          const long long jl = (j >= l) ? kb + l : 0;
          const long long kl = kc + l;
          const double bb = (k == l) ? 1.0 : 2.0;
          const double aj = a.w[kk];
          const double ak = a.wk[kk];
          ++kk;
          if (kl > ij) continue;
          if (i == k && aa + bb < 2.1) {
            atomicAdd(&a.f[ij - 1], aj * a.ptot[kl - 1]);
          } else {
            atomicAdd(&a.f[ij - 1], bb * aj * a.ptot[kl - 1]);
            atomicAdd(&a.f[kl - 1], aa * aj * a.ptot[ij - 1]);
            const double e = ak * aa * bb * 0.25;
            if (jl > 0 && ik > 0) atomicAdd(&a.f[ik - 1], -e * a.p[jl - 1]);
            if (jk > 0 && il > 0) atomicAdd(&a.f[il - 1], -e * a.p[jk - 1]);
            if (jk > 0) atomicAdd(&a.f[jk - 1], -e * a.p[il - 1]);
            if (jl > 0) atomicAdd(&a.f[jl - 1], -e * a.p[ik - 1]);
          }
        }
      }
    }
  }
}

// Device copies kept between calls.
struct Cache {
  long long gen = -1;
  long long n2elec = 0;
  int npairs = 0, nself = 0;
  double *w = nullptr, *wk = nullptr;
  int *pia = nullptr, *pib = nullptr, *pja = nullptr, *pjb = nullptr, *sia = nullptr, *sib = nullptr;
  long long *poff = nullptr, *soff = nullptr;
  long long mpack = 0;
  double *ptot = nullptr, *p = nullptr, *f = nullptr;
  bool tables = false;
  void release() {
    void *ptrs[] = {w, wk, pia, pib, pja, pjb, sia, sib, poff, soff, ptot, p, f};
    for (void *q : ptrs) {
      if (q) cudaFree(q);
    }
    w = wk = ptot = p = f = nullptr;
    pia = pib = pja = pjb = sia = sib = nullptr;
    poff = soff = nullptr;
    gen = -1;
    n2elec = mpack = 0;
    npairs = nself = 0;
  }
};
Cache g_cache;

// w / wk offered by the device hcore: unconfirmed (gen = -1) until hcore confirms them.
struct Offer {
  double *w = nullptr, *wk = nullptr;
  long long n2elec = 0;
  long long gen = -1;
  bool host_current = true;   // the host w / wk hold the same pair segments
  void release() {
    if (w) cudaFree(w);
    if (wk) cudaFree(wk);
    w = wk = nullptr;
    n2elec = 0;
    gen = -1;
    host_current = true;
  }
};
Offer g_offer;

// Host w / wk of generation g_stale_gen lack their pair segments (device hcore without download);
// the one-centre blocks (built on the host) are g_blocks: (0-based offset, length) in w.
bool g_w_host_stale = false;
long long g_stale_gen = -1;
std::vector<std::pair<long long, long long>> g_blocks;

template <typename T>
bool up(T **dst, const T *src, size_t n) {
  if (n == 0) return true;
  if (cudaMalloc(reinterpret_cast<void **>(dst), n * sizeof(T)) != cudaSuccess) {
    *dst = nullptr;
    return false;
  }
  return mopac_pinned_h2d(*dst, src, n * sizeof(T)) == 1;
}

}  // namespace

// Fortran entry point.  Pairs jj < ii (pair_*), pairs ii = jj (self_*), with
// orbital ranges ia..ib (atom ii), ja..jb (atom jj) and the 0-based start of
// their integrals in w / wk; gen identifies the w / wk of the current hcore
// (they are uploaded again when it changes).  f (mpack) is accumulated in
// place with the two-centre terms.  Returns 0 on success, 2 on CUDA failure
// (f untouched).
extern "C" int mopac_cuda_conv_fock2(long long mpack, long long n2elec, int npairs, const int *pair_ia,
                                     const int *pair_ib, const int *pair_ja, const int *pair_jb,
                                     const long long *pair_off, int nself, const int *self_ia,
                                     const int *self_ib, const long long *self_off, const double *w,
                                     const double *wk, long long gen, const double *ptot, const double *p,
                                     double *f, double *ms_out) {
  const auto t0 = std::chrono::steady_clock::now();
  Cache &c = g_cache;
  if (!c.tables) {
    unsigned char ti[45], tj[45];
    int t = 0;
    for (int i = 0; i < 9; ++i) {
      for (int j = 0; j <= i; ++j, ++t) {
        ti[t] = static_cast<unsigned char>(i);
        tj[t] = static_cast<unsigned char>(j);
      }
    }
    if (cudaMemcpyToSymbol(c_ti, ti, sizeof(ti)) != cudaSuccess ||
        cudaMemcpyToSymbol(c_tj, tj, sizeof(tj)) != cudaSuccess) {
      return 2;
    }
    c.tables = true;
  }
  if (c.gen != gen || c.n2elec != n2elec || c.npairs != npairs || c.nself != nself || c.mpack != mpack) {
    c.release();
    const size_t n = static_cast<size_t>(n2elec);
    const auto tu = std::chrono::steady_clock::now();
    bool w_ok;
    if (g_offer.w && g_offer.gen == gen && g_offer.n2elec == n2elec) {
      c.w = g_offer.w;
      c.wk = g_offer.wk;
      g_offer.w = g_offer.wk = nullptr;
      g_offer.release();
      w_ok = true;
    } else {
      g_offer.release();
      // never upload host pair segments that the device hcore did not fill
      if (g_w_host_stale && g_stale_gen == gen) return 2;
      w_ok = up(&c.w, w, n) && up(&c.wk, wk, n);
      add_section("conv_fock_w_upload", tu, Clock::now());
    }
    if (!w_ok || !up(&c.pia, pair_ia, npairs) || !up(&c.pib, pair_ib, npairs) ||
        !up(&c.pja, pair_ja, npairs) || !up(&c.pjb, pair_jb, npairs) || !up(&c.poff, pair_off, npairs) ||
        !up(&c.sia, self_ia, nself) || !up(&c.sib, self_ib, nself) || !up(&c.soff, self_off, nself) ||
        cudaMalloc(reinterpret_cast<void **>(&c.ptot), mpack * sizeof(double)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&c.p), mpack * sizeof(double)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&c.f), mpack * sizeof(double)) != cudaSuccess) {
      c.release();
      cudaGetLastError();
      return 2;
    }
    c.gen = gen;
    c.n2elec = n2elec;
    c.npairs = npairs;
    c.nself = nself;
    c.mpack = mpack;
  }
  const size_t bytes = static_cast<size_t>(mpack) * sizeof(double);
  const auto th0 = Clock::now();
  if (mopac_pinned_h2d(c.ptot, ptot, bytes) != 1 || mopac_pinned_h2d(c.p, p, bytes) != 1 ||
      mopac_pinned_h2d(c.f, f, bytes) != 1) {
    return 2;
  }
  const auto th1 = Clock::now();
  add_section("conv_fock_h2d", th0, th1);
  FockArgs a;
  a.npairs = npairs;
  a.pair_ia = c.pia;
  a.pair_ib = c.pib;
  a.pair_ja = c.pja;
  a.pair_jb = c.pjb;
  a.pair_off = c.poff;
  a.nself = nself;
  a.self_ia = c.sia;
  a.self_ib = c.sib;
  a.self_off = c.soff;
  a.w = c.w;
  a.wk = c.wk;
  a.ptot = c.ptot;
  a.p = c.p;
  a.f = c.f;
  if (npairs > 0) {
    fock2_pairs_kernel<<<(npairs + kWarpsPerBlock - 1) / kWarpsPerBlock, 32 * kWarpsPerBlock>>>(a);
  }
  if (nself > 0) fock2_self_kernel<<<(nself + 63) / 64, 64>>>(a);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[GPU ERROR] conventional Fock: %s\n", cudaGetErrorString(err));
    return 2;
  }
  const auto tk = Clock::now();
  add_section("conv_fock_kernels", th1, tk);
  if (mopac_pinned_d2h(f, c.f, bytes) != 1) return 2;
  add_section("conv_fock_d2h", tk, Clock::now());
  if (ms_out) {
    *ms_out = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  return 0;
}

// Frees the device copies (end of a run or when memory is needed elsewhere).
extern "C" void mopac_cuda_conv_fock2_release() { g_cache.release(); }

// Device hcore of a solid: w and wk (device buffers of n2elec doubles, the pair segments as
// hcore lays them out) for the Fock builds of the coming generation.  Ownership passes here.
// host_current = 0: the pair segments were not copied to the host w / wk.
extern "C" void mopac_cuda_conv_fock_offer_w(double *w, double *wk, long long n2elec, int host_current) {
  g_offer.release();
  g_offer.w = w;
  g_offer.wk = wk;
  g_offer.n2elec = n2elec;
  g_offer.gen = -1;
  g_offer.host_current = host_current != 0;
}

// hcore, after bumping the generation: keep = 1 when its w / wk are the offered ones (device
// hcore, no check mode), else the offer is dropped.  blk_off / blk_len (nblk): the one-centre
// blocks of w (0-based offsets), which only the host has.
extern "C" void mopac_cuda_conv_fock_confirm_w(long long gen, int keep, int nblk, const long long *blk_off,
                                               const long long *blk_len) {
  g_w_host_stale = false;
  if (keep && g_offer.w) {
    g_offer.gen = gen;
    if (!g_offer.host_current) {
      g_w_host_stale = true;
      g_stale_gen = gen;
      g_blocks.assign(static_cast<size_t>(nblk), {0, 0});
      for (int b = 0; b < nblk; ++b) g_blocks[b] = {blk_off[b], blk_len[b]};
    }
  } else {
    g_offer.release();
  }
}

// Fills the pair segments of the host w / wk (n2elec) from the device copies when they are stale,
// keeping the host's one-centre blocks.  Returns 1 on success (or nothing to do), 0 if the device
// copies are gone or a copy failed (the host w / wk are then unusable).
extern "C" int mopac_cuda_conv_w_sync_host(double *w, double *wk, long long n2elec) {
  if (!g_w_host_stale) return 1;
  const double *dw = nullptr, *dwk = nullptr;
  if (g_cache.w && g_cache.gen == g_stale_gen && g_cache.n2elec == n2elec) {
    dw = g_cache.w;
    dwk = g_cache.wk;
  } else if (g_offer.w && g_offer.gen == g_stale_gen && g_offer.n2elec == n2elec) {
    dw = g_offer.w;
    dwk = g_offer.wk;
  }
  if (!dw || !dwk) return 0;
  const auto t0 = Clock::now();
  size_t total = 0;
  for (const auto &b : g_blocks) total += static_cast<size_t>(b.second);
  std::vector<double> saved(total);
  size_t k = 0;
  for (const auto &b : g_blocks) {
    std::memcpy(saved.data() + k, w + b.first, static_cast<size_t>(b.second) * sizeof(double));
    k += static_cast<size_t>(b.second);
  }
  const size_t bytes = static_cast<size_t>(n2elec) * sizeof(double);
  if (cudaDeviceSynchronize() != cudaSuccess || mopac_pinned_d2h(w, dw, bytes) != 1 ||
      mopac_pinned_d2h(wk, dwk, bytes) != 1) {
    cudaGetLastError();
    return 0;
  }
  k = 0;
  for (const auto &b : g_blocks) {
    std::memcpy(w + b.first, saved.data() + k, static_cast<size_t>(b.second) * sizeof(double));
    std::memset(wk + b.first, 0, static_cast<size_t>(b.second) * sizeof(double));
    k += static_cast<size_t>(b.second);
  }
  g_w_host_stale = false;
  add_section("conv_w_sync_host", t0, Clock::now());
  return 1;
}

// Frees the device w / wk of the last generation (a new hcore replaces them), so the device
// hcore has room for its own full-size buffers.
extern "C" void mopac_cuda_conv_fock_drop_w() {
  g_cache.release();
  g_offer.release();
  g_w_host_stale = false;
}
