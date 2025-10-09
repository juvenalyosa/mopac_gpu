#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstdint>

#include "packed_utils.h"

extern "C" bool mopac_cuda_fock2_scf(int norbs, int mpack, int numat,
                                      const int *nfirst, const int *nlast,
                                      const double *ptot, const double *p,
                                      const double *w, const double *wj, const double *wk,
                                      int periodic_flag,
                                      double *fout);

extern "C" void mopac_cuda_dsyevd_keep(int n, double *A, int lda, double *W, int *info);
extern "C" void mopac_cuda_fetch_eigenvectors(int n, double *A, int lda);
extern "C" void mopac_cuda_density_from_dev_syrk(int n, int ndubl, double alpha, double *C, int ldc);
extern "C" bool mopac_cuda_density_copy_cached(double *dest, size_t len, const double *host_ptr);
extern "C" bool mopac_cuda_fock_copy_cached(double *dest, size_t len, const double *host_ptr);
extern "C" void mopac_cuda_register_fock_device(int linear, double *host_ptr, const double *src_dev);
extern "C" bool mopac_cuda_fetch_fock(double *host_ptr, size_t linear);
extern "C" void mopac_cuda_clear_fock_cache();
extern "C" int mopac_cuda_get_resident_mode();
extern "C" bool mopac_cuda_launch_pairs_kernel(int npairs,
                                               const int *pair_i,
                                               const int *pair_j,
                                               const int *pair_type,
                                               const int *pair_w_off,
                                               const int *pair_wj_off,
                                               const int *pair_wk_off,
                                               const int *nfirst,
                                               const int *nlast,
                                               const double *ptot,
                                               const double *p,
                                               const double *w,
                                               const double *wj,
                                               const double *wk,
                                               double *f,
                                               int debug_flag);

namespace {
struct MopacGpuScfContext {
  int    norbs;
  int    nalpha;
  int    nbeta;
  int    mpack;
  int    max_iter;
  int    numat;
  int    n2elec;
  int    periodic;
  int    iterations;
  int    flags;
  double energy_tol;
  double density_tol;
  double energy_total;
  double energy_delta;
  double density_rms;
  void  *h_core;
  void  *overlap;
  void  *density_alpha;
  void  *density_beta;
  void  *density_total;
  void  *fock_alpha;
  void  *fock_beta;
  void  *coeff_alpha;
  void  *coeff_beta;
  void  *eigvals_alpha;
  void  *eigvals_beta;
  void  *nfirst;
  void  *nlast;
  void  *two_e_w;
  void  *two_e_wj;
  void  *two_e_wk;
  void  *work;
  void  *log_buffer;
};

constexpr int GPU_SCF_FLAG_RHF = 2;
constexpr int GPU_SCF_FLAG_UHF = 4;

std::string &last_error() {
  static std::string err_msg = "GPU SCF driver not initialised";
  return err_msg;
}

std::mutex &error_mutex() {
  static std::mutex mtx;
  return mtx;
}

void set_last_error(const std::string &msg) {
  std::lock_guard<std::mutex> guard(error_mutex());
  last_error() = msg;
}

bool gpu_logging_enabled() {
  const char *env = std::getenv("MOPAC_GPU_SCF_DEBUG");
  if (!env || *env == '\0') env = std::getenv("MOPAC_GPU_SCF_STUB_LOG");
  if (!env || *env == '\0') return false;
  return !(std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0 || std::strcmp(env, "false") == 0);
}

inline size_t packed_length(int n) {
  return static_cast<size_t>(n) * static_cast<size_t>(n + 1) / 2;
}

void unpack_packed_lower(const double *packed, int n, double *full) {
  size_t idx = 0;
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row <= col; ++row, ++idx) {
      double val = packed[idx];
      full[row + col * static_cast<size_t>(n)] = val;
      if (row != col) {
        full[col + row * static_cast<size_t>(n)] = val;
      }
    }
  }
}

void pack_full_lower(const double *full, int n, double *packed) {
  size_t idx = 0;
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row <= col; ++row, ++idx) {
      packed[idx] = full[row + col * static_cast<size_t>(n)];
    }
  }
}

double spin_energy_from_packed(int n, const double *p, const double *h, const double *f) {
  if (!p || !h || !f) return 0.0;
  double ed = 0.0;
  double ee = 0.0;
  int k = 0;
  const int nn = n + 1;
  for (int i = 2; i <= nn; ++i) {
    ++k;
    int base = k - 1;
    int jj = i - 1;
    ed += p[base] * (h[base] + f[base]);
    if (i == nn) continue;
    if (jj > 0) {
      for (int off = 1; off <= jj; ++off) {
        int pos = base + off;
        ee += p[pos] * (h[pos] + f[pos]);
      }
      k += jj;
    }
  }
  ee += 0.5 * ed;
  return ee;
}

double rms_density_diff(const double *a, const double *b, size_t len) {
  if (!a || !b || len == 0) return 0.0;
  long double acc = 0.0;
  for (size_t i = 0; i < len; ++i) {
    long double diff = static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
    acc += diff * diff;
  }
  return std::sqrt(static_cast<double>(acc / static_cast<long double>(len)));
}

struct GpuScfStreamCookie {
  int32_t norbs;
  int32_t mpack;
  int32_t numat;
  int32_t periodic_flag;
  int32_t has_exchange;
  void *ptot;
  void *p;
  void *f;
  void *nfirst;
  void *nlast;
};

enum PairTypeCodes {
  PAIR_LIGHT_LIGHT = 0,
  PAIR_HEAVY_LIGHT = 1,
  PAIR_LIGHT_HEAVY = 2,
  PAIR_HEAVY_HEAVY = 3,
  PAIR_GENERAL     = 4,
  PAIR_PERIODIC    = 5
};

constexpr int STREAM_STATUS_SUCCESS      =  0;
constexpr int STREAM_STATUS_NOT_READY    = -1;
constexpr int STREAM_STATUS_BAD_ARGS     = -2;
constexpr int STREAM_STATUS_COPY_FAILED  = -3;
constexpr int STREAM_STATUS_INTERNAL     = -4;

struct PairChunkLengths {
  int type = PAIR_GENERAL;
  int chunk_w = 0;
  int chunk_wj = 0;
  int chunk_wk = 0;
};

struct ExpectedBlock {
  int ii = 0;
  int jj = 0;
  int ia = 0;
  int ib = 0;
  int ja = 0;
  int jb = 0;
  int span_i = 0;
  int span_j = 0;
  int type = PAIR_GENERAL;
  int len_w = 0;
  int len_wj = 0;
  int len_wk = 0;
  size_t w_off = 0;
  size_t wj_off = 0;
  size_t wk_off = 0;
};

struct StreamSession {
  GpuScfStreamCookie cookie{};
  int norbs = 0;
  int mpack = 0;
  int numat = 0;
  bool periodic = false;
  bool has_exchange = false;
  const double *ptot_host = nullptr;
  const double *p_host = nullptr;
  double *f_host = nullptr;
  const int *nfirst_host = nullptr;
  const int *nlast_host = nullptr;

  std::vector<int> pair_i;
  std::vector<int> pair_j;
  std::vector<int> pair_type;
  std::vector<int> pair_w_off;
  std::vector<int> pair_wj_off;
  std::vector<int> pair_wk_off;
  std::vector<ExpectedBlock> blocks;
  std::vector<char> filled;

  size_t total_w_len = 0;
  size_t total_wj_len = 0;
  size_t total_wk_len = 0;
  size_t max_w_len = 0;
  size_t max_wj_len = 0;
  size_t max_wk_len = 0;

  double *d_ptot = nullptr;
  double *d_p = nullptr;
  double *d_f = nullptr;
  int *d_nfirst = nullptr;
  int *d_nlast = nullptr;
  double *d_w = nullptr;
  double *d_wj = nullptr;
  double *d_wk = nullptr;
  int *d_pair_i = nullptr;
  int *d_pair_j = nullptr;
  int *d_pair_type = nullptr;
  int *d_pair_w_off = nullptr;
  int *d_pair_wj_off = nullptr;
  int *d_pair_wk_off = nullptr;

  cudaStream_t stream = nullptr;
  bool active = false;
  bool error = false;
  int error_code = 0;
  std::string error_msg;
  size_t next_block = 0;
  size_t processed_blocks = 0;
};

std::unordered_map<void*, std::unique_ptr<StreamSession>> &stream_sessions() {
  static std::unordered_map<void*, std::unique_ptr<StreamSession>> sessions;
  return sessions;
}

std::mutex &stream_mutex() {
  static std::mutex mtx;
  return mtx;
}

PairChunkLengths classify_pair(bool periodic, int span_i, int span_j) {
  PairChunkLengths info;
  if (periodic) {
    info.type = PAIR_PERIODIC;
    int pairs_i = mopac_gpu::pair_count(span_i);
    int pairs_j = mopac_gpu::pair_count(span_j);
    info.chunk_w = pairs_i * pairs_j;
    info.chunk_wj = info.chunk_w;
    info.chunk_wk = info.chunk_w;
    return info;
  }

  bool has_d = (span_i >= 7) || (span_j >= 7);
  if (has_d) {
    info.type = PAIR_GENERAL;
    info.chunk_w = mopac_gpu::pair_count(span_i) * mopac_gpu::pair_count(span_j);
    return info;
  }

  if (span_i >= 4 && span_j >= 4) {
    info.type = PAIR_HEAVY_HEAVY;
    info.chunk_w = 100;
    return info;
  }
  if (span_i >= 4 && span_j == 1) {
    info.type = PAIR_HEAVY_LIGHT;
    info.chunk_w = 10;
    return info;
  }
  if (span_j >= 4 && span_i == 1) {
    info.type = PAIR_LIGHT_HEAVY;
    info.chunk_w = 10;
    return info;
  }
  if (span_i == 1 && span_j == 1) {
    info.type = PAIR_LIGHT_LIGHT;
    info.chunk_w = 1;
    return info;
  }
  info.type = PAIR_GENERAL;
  info.chunk_w = mopac_gpu::pair_count(span_i) * mopac_gpu::pair_count(span_j);
  return info;
}

void destroy_session(StreamSession &session) {
  if (session.d_ptot) cudaFree(session.d_ptot);
  if (session.d_p) cudaFree(session.d_p);
  if (session.d_f) cudaFree(session.d_f);
  if (session.d_nfirst) cudaFree(session.d_nfirst);
  if (session.d_nlast) cudaFree(session.d_nlast);
  if (session.d_w) cudaFree(session.d_w);
  if (session.d_wj) cudaFree(session.d_wj);
  if (session.d_wk) cudaFree(session.d_wk);
  if (session.d_pair_i) cudaFree(session.d_pair_i);
  if (session.d_pair_j) cudaFree(session.d_pair_j);
  if (session.d_pair_type) cudaFree(session.d_pair_type);
  if (session.d_pair_w_off) cudaFree(session.d_pair_w_off);
  if (session.d_pair_wj_off) cudaFree(session.d_pair_wj_off);
  if (session.d_pair_wk_off) cudaFree(session.d_pair_wk_off);
  if (session.stream) cudaStreamDestroy(session.stream);
  session.d_ptot = session.d_p = session.d_f = nullptr;
  session.d_nfirst = session.d_nlast = nullptr;
  session.d_w = session.d_wj = session.d_wk = nullptr;
  session.d_pair_i = session.d_pair_j = nullptr;
  session.d_pair_type = nullptr;
  session.d_pair_w_off = session.d_pair_wj_off = session.d_pair_wk_off = nullptr;
  session.stream = nullptr;
  session.active = false;
}

void set_stream_error(StreamSession &session, const std::string &msg, int code) {
  session.error = true;
  session.error_code = code;
  session.error_msg = msg;
  if (gpu_logging_enabled()) {
    std::fprintf(stderr, "[GPU STREAM] error: %s (code=%d)\n", msg.c_str(), code);
    std::fflush(stderr);
  }
  set_last_error(msg);
}

std::string cuda_error_message(const char *where, cudaError_t err) {
  std::ostringstream oss;
  oss << where << ": " << cudaGetErrorString(err);
  return oss.str();
}

bool build_expected_blocks(StreamSession &session, std::string &err) {
  if (!session.nfirst_host || !session.nlast_host) {
    err = "GPU SCF stream: atom ranges are null";
    return false;
  }
  const size_t max_index = static_cast<size_t>(std::numeric_limits<int>::max());
  size_t w_off = 0;
  size_t wj_off = 0;
  size_t wk_off = 0;

  for (int ii = 1; ii <= session.numat; ++ii) {
    int ia = session.nfirst_host[ii - 1];
    int ib = session.nlast_host[ii - 1];
    int span_i = mopac_gpu::span_count(ia, ib);
    if (span_i <= 0) continue;
    for (int jj = 1; jj < ii; ++jj) {
      int ja = session.nfirst_host[jj - 1];
      int jb = session.nlast_host[jj - 1];
      int span_j = mopac_gpu::span_count(ja, jb);
      if (span_j <= 0) continue;

      PairChunkLengths chunks = classify_pair(session.periodic, span_i, span_j);
      if (chunks.chunk_w < 0 || chunks.chunk_wj < 0 || chunks.chunk_wk < 0) {
        err = "GPU SCF stream: negative block length detected";
        return false;
      }
      if (chunks.chunk_w == 0 && chunks.chunk_wj == 0 && chunks.chunk_wk == 0) continue;

      if (w_off + static_cast<size_t>(chunks.chunk_w) > max_index ||
          wj_off + static_cast<size_t>(chunks.chunk_wj) > max_index ||
          wk_off + static_cast<size_t>(chunks.chunk_wk) > max_index) {
        err = "GPU SCF stream: integral offset exceeds 32-bit index range";
        return false;
      }

      ExpectedBlock blk;
      blk.ii = ii;
      blk.jj = jj;
      blk.ia = ia;
      blk.ib = ib;
      blk.ja = ja;
      blk.jb = jb;
      blk.span_i = span_i;
      blk.span_j = span_j;
      blk.type = chunks.type;
      blk.len_w = chunks.chunk_w;
      blk.len_wj = chunks.chunk_wj;
      blk.len_wk = chunks.chunk_wk;
      blk.w_off = w_off;
      blk.wj_off = wj_off;
      blk.wk_off = wk_off;

      session.blocks.push_back(blk);
      session.pair_i.push_back(ii);
      session.pair_j.push_back(jj);
      session.pair_type.push_back(chunks.type);
      session.pair_w_off.push_back(static_cast<int>(w_off));
      session.pair_wj_off.push_back(static_cast<int>(wj_off));
      session.pair_wk_off.push_back(static_cast<int>(wk_off));
      session.max_w_len = std::max(session.max_w_len, static_cast<size_t>(blk.len_w));
      session.max_wj_len = std::max(session.max_wj_len, static_cast<size_t>(blk.len_wj));
      session.max_wk_len = std::max(session.max_wk_len, static_cast<size_t>(blk.len_wk));

      w_off += static_cast<size_t>(chunks.chunk_w);
      wj_off += static_cast<size_t>(chunks.chunk_wj);
      wk_off += static_cast<size_t>(chunks.chunk_wk);
    }
  }

  session.total_w_len = w_off;
  session.total_wj_len = wj_off;
  session.total_wk_len = wk_off;
  session.filled.assign(session.blocks.size(), 0);
  return true;
}

bool allocate_stream_buffers(StreamSession &session, std::string &err) {
  size_t atoms_e = static_cast<size_t>(session.numat);
  size_t mpack_e = static_cast<size_t>(session.mpack);

  if (session.d_ptot || session.d_p || session.d_f) {
    err = "GPU SCF stream: buffers already allocated";
    return false;
  }

  cudaError_t errc;
  errc = cudaMalloc(reinterpret_cast<void **>(&session.d_ptot), sizeof(double) * mpack_e);
  if (errc != cudaSuccess) { err = cuda_error_message("cudaMalloc ptot", errc); return false; }
  errc = cudaMalloc(reinterpret_cast<void **>(&session.d_p), sizeof(double) * mpack_e);
  if (errc != cudaSuccess) { err = cuda_error_message("cudaMalloc p", errc); return false; }
  errc = cudaMalloc(reinterpret_cast<void **>(&session.d_f), sizeof(double) * mpack_e);
  if (errc != cudaSuccess) { err = cuda_error_message("cudaMalloc f", errc); return false; }
  errc = cudaMalloc(reinterpret_cast<void **>(&session.d_nfirst), sizeof(int) * atoms_e);
  if (errc != cudaSuccess) { err = cuda_error_message("cudaMalloc nfirst", errc); return false; }
  errc = cudaMalloc(reinterpret_cast<void **>(&session.d_nlast), sizeof(int) * atoms_e);
  if (errc != cudaSuccess) { err = cuda_error_message("cudaMalloc nlast", errc); return false; }

  size_t coulomb_cap = std::max(session.max_w_len, session.max_wj_len);
  if (coulomb_cap > 0) {
    errc = cudaMalloc(reinterpret_cast<void **>(&session.d_w), sizeof(double) * coulomb_cap);
    if (errc != cudaSuccess) { err = cuda_error_message("cudaMalloc stream coulomb", errc); return false; }
  }
  if (session.max_wk_len > 0) {
    errc = cudaMalloc(reinterpret_cast<void **>(&session.d_wk), sizeof(double) * session.max_wk_len);
    if (errc != cudaSuccess) { err = cuda_error_message("cudaMalloc stream exchange", errc); return false; }
  }
  if (!session.blocks.empty()) {
    errc = cudaMalloc(reinterpret_cast<void **>(&session.d_pair_i), sizeof(int));
    if (errc != cudaSuccess) { err = cuda_error_message("cudaMalloc pair_i", errc); return false; }
    errc = cudaMalloc(reinterpret_cast<void **>(&session.d_pair_j), sizeof(int));
    if (errc != cudaSuccess) { err = cuda_error_message("cudaMalloc pair_j", errc); return false; }
    errc = cudaMalloc(reinterpret_cast<void **>(&session.d_pair_type), sizeof(int));
    if (errc != cudaSuccess) { err = cuda_error_message("cudaMalloc pair_type", errc); return false; }
    errc = cudaMalloc(reinterpret_cast<void **>(&session.d_pair_w_off), sizeof(int));
    if (errc != cudaSuccess) { err = cuda_error_message("cudaMalloc pair_w_off", errc); return false; }
    errc = cudaMalloc(reinterpret_cast<void **>(&session.d_pair_wj_off), sizeof(int));
    if (errc != cudaSuccess) { err = cuda_error_message("cudaMalloc pair_wj_off", errc); return false; }
    errc = cudaMalloc(reinterpret_cast<void **>(&session.d_pair_wk_off), sizeof(int));
    if (errc != cudaSuccess) { err = cuda_error_message("cudaMalloc pair_wk_off", errc); return false; }
  }

  errc = cudaMemset(session.d_f, 0, sizeof(double) * mpack_e);
  if (errc != cudaSuccess) { err = cuda_error_message("cudaMemset f", errc); return false; }

  if (!mopac_cuda_density_copy_cached(session.d_ptot, mpack_e, session.ptot_host)) {
    errc = cudaMemcpy(session.d_ptot, session.ptot_host, sizeof(double) * mpack_e, cudaMemcpyHostToDevice);
    if (errc != cudaSuccess) { err = cuda_error_message("cudaMemcpy ptot", errc); return false; }
  }
  if (!mopac_cuda_density_copy_cached(session.d_p, mpack_e, session.p_host)) {
    errc = cudaMemcpy(session.d_p, session.p_host, sizeof(double) * mpack_e, cudaMemcpyHostToDevice);
    if (errc != cudaSuccess) { err = cuda_error_message("cudaMemcpy p", errc); return false; }
  }

  errc = cudaMemcpy(session.d_nfirst, session.nfirst_host, sizeof(int) * atoms_e, cudaMemcpyHostToDevice);
  if (errc != cudaSuccess) { err = cuda_error_message("cudaMemcpy nfirst", errc); return false; }
  errc = cudaMemcpy(session.d_nlast, session.nlast_host, sizeof(int) * atoms_e, cudaMemcpyHostToDevice);
  if (errc != cudaSuccess) { err = cuda_error_message("cudaMemcpy nlast", errc); return false; }

  return true;
}
} // namespace

extern "C" bool mopac_cuda_scf_run(MopacGpuScfContext *ctx) {
  if (!ctx) {
    set_last_error("GPU SCF context pointer is null");
    return false;
  }

  const bool verbose = gpu_logging_enabled();

  const int norbs = ctx->norbs;
  const int mpack = ctx->mpack;
  if (norbs <= 0 || mpack <= 0) {
    set_last_error("GPU SCF: invalid matrix dimensions");
    return false;
  }
  if (static_cast<size_t>(mpack) != packed_length(norbs)) {
    set_last_error("GPU SCF: inconsistent packed matrix length");
    return false;
  }
  if (ctx->max_iter <= 0) {
    set_last_error("GPU SCF: max_iter must be positive");
    return false;
  }
  if (ctx->numat <= 0) {
    set_last_error("GPU SCF stub: number of atoms not provided");
    return false;
  }

  auto *h_core = static_cast<double*>(ctx->h_core);
  auto *p_alpha = static_cast<double*>(ctx->density_alpha);
  auto *p_beta  = static_cast<double*>(ctx->density_beta);
  auto *p_total = static_cast<double*>(ctx->density_total);
  auto *f_alpha = static_cast<double*>(ctx->fock_alpha);
  auto *f_beta  = static_cast<double*>(ctx->fock_beta);
  auto *coeff_alpha = static_cast<double*>(ctx->coeff_alpha);
  auto *coeff_beta  = static_cast<double*>(ctx->coeff_beta);
  auto *eigvals_alpha = static_cast<double*>(ctx->eigvals_alpha);
  auto *eigvals_beta  = static_cast<double*>(ctx->eigvals_beta);
  auto *nfirst = static_cast<const int*>(ctx->nfirst);
  auto *nlast  = static_cast<const int*>(ctx->nlast);
  auto *w      = static_cast<const double*>(ctx->two_e_w);
  const double *wj = ctx->two_e_wj ? static_cast<const double*>(ctx->two_e_wj) : w;
  const double *wk = ctx->two_e_wk ? static_cast<const double*>(ctx->two_e_wk)
                                   : (ctx->two_e_w ? static_cast<const double*>(ctx->two_e_w) : nullptr);

  if (!h_core || !p_alpha || !p_total || !f_alpha || !nfirst || !nlast || !w) {
    set_last_error("GPU SCF: required matrix pointers are null");
    return false;
  }

  const bool uhf = (ctx->flags & GPU_SCF_FLAG_UHF) != 0;
  const bool rhf = (ctx->flags & GPU_SCF_FLAG_RHF) != 0 || !uhf;
  if (uhf) {
    if (!p_beta || !f_beta) {
      set_last_error("GPU SCF: beta density/Fock pointers missing for UHF run");
      return false;
    }
  }
  if (!rhf && !uhf) {
    set_last_error("GPU SCF: spin configuration not specified");
    return false;
  }

  const int numat = ctx->numat;
  const int max_iter = ctx->max_iter;
  const int periodic_flag = (ctx->periodic != 0) ? 1 : 0;
  const double energy_tol = (ctx->energy_tol > 0.0) ? ctx->energy_tol : 1.0e-8;
  const double density_tol = (ctx->density_tol > 0.0) ? ctx->density_tol : 1.0e-6;

  if (verbose) {
    std::fprintf(stderr,
                 "[GPU SCF] start: norbs=%d mpack=%d max_iter=%d periodic=%d rhf=%d uhf=%d\n",
                 norbs, mpack, max_iter, periodic_flag, rhf ? 1 : 0, uhf ? 1 : 0);
    std::fflush(stderr);
  }

  std::vector<double> alpha_full(static_cast<size_t>(norbs) * static_cast<size_t>(norbs), 0.0);
  std::vector<double> beta_full;
  if (uhf) beta_full.assign(alpha_full.size(), 0.0);

  std::vector<double> density_full(static_cast<size_t>(norbs) * static_cast<size_t>(norbs), 0.0);
  std::vector<double> density_full_beta;
  if (uhf) density_full_beta.assign(density_full.size(), 0.0);

  std::vector<double> alpha_eigs(norbs, 0.0);
  std::vector<double> beta_eigs;
  if (uhf) beta_eigs.assign(norbs, 0.0);

  std::vector<double> prev_density(static_cast<size_t>(mpack));
  std::copy(p_total, p_total + mpack, prev_density.begin());

  std::vector<double> ptot_iter(static_cast<size_t>(mpack));
  std::vector<double> pa_iter(static_cast<size_t>(mpack));
  std::vector<double> pb_iter;
  if (uhf) pb_iter.assign(static_cast<size_t>(mpack), 0.0);

  double prev_energy = std::numeric_limits<double>::infinity();
  double total_energy = prev_energy;
  double energy_delta = prev_energy;
  double density_rms = prev_energy;

  for (int iter = 1; iter <= max_iter; ++iter) {
    std::copy(p_total, p_total + mpack, ptot_iter.begin());
    std::copy(p_alpha, p_alpha + mpack, pa_iter.begin());
    if (uhf) std::copy(p_beta, p_beta + mpack, pb_iter.begin());

    std::memcpy(f_alpha, h_core, sizeof(double) * static_cast<size_t>(mpack));
    if (!mopac_cuda_fock2_scf(norbs, mpack, numat,
                              nfirst, nlast,
                              ptot_iter.data(), pa_iter.data(),
                              w, wj, wk, periodic_flag,
                              f_alpha)) {
      set_last_error("GPU SCF: alpha Fock build failed");
      return false;
    }

    unpack_packed_lower(f_alpha, norbs, alpha_full.data());
    int info = 0;
    mopac_cuda_dsyevd_keep(norbs, alpha_full.data(), norbs, alpha_eigs.data(), &info);
    if (info != 0) {
      char buf[96];
      std::snprintf(buf, sizeof(buf), "GPU SCF: alpha diagonalisation failed (info=%d)", info);
      set_last_error(buf);
      return false;
    }
    mopac_cuda_density_from_dev_syrk(norbs, std::max(0, ctx->nalpha), 1.0, density_full.data(), norbs);
    pack_full_lower(density_full.data(), norbs, p_alpha);
    if (coeff_alpha) mopac_cuda_fetch_eigenvectors(norbs, coeff_alpha, norbs);
    if (eigvals_alpha) std::memcpy(eigvals_alpha, alpha_eigs.data(), sizeof(double) * static_cast<size_t>(norbs));

    if (uhf) {
      std::memcpy(f_beta, h_core, sizeof(double) * static_cast<size_t>(mpack));
      if (!mopac_cuda_fock2_scf(norbs, mpack, numat,
                                nfirst, nlast,
                                ptot_iter.data(), pb_iter.data(),
                                w, wj, wk, periodic_flag,
                                f_beta)) {
        set_last_error("GPU SCF: beta Fock build failed");
        return false;
      }
      unpack_packed_lower(f_beta, norbs, beta_full.data());
      info = 0;
      mopac_cuda_dsyevd_keep(norbs, beta_full.data(), norbs, beta_eigs.data(), &info);
      if (info != 0) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "GPU SCF: beta diagonalisation failed (info=%d)", info);
        set_last_error(buf);
        return false;
      }
      mopac_cuda_density_from_dev_syrk(norbs, std::max(0, ctx->nbeta), 1.0, density_full_beta.data(), norbs);
      pack_full_lower(density_full_beta.data(), norbs, p_beta);
      if (coeff_beta) mopac_cuda_fetch_eigenvectors(norbs, coeff_beta, norbs);
      if (eigvals_beta) std::memcpy(eigvals_beta, beta_eigs.data(), sizeof(double) * static_cast<size_t>(norbs));
    } else if (p_beta) {
      std::memcpy(p_beta, p_alpha, sizeof(double) * static_cast<size_t>(mpack));
    }

    if (uhf) {
      for (int i = 0; i < mpack; ++i) {
        p_total[i] = p_alpha[i] + (p_beta ? p_beta[i] : 0.0);
      }
    } else {
      for (int i = 0; i < mpack; ++i) {
        double spin_val = p_alpha[i];
        p_total[i] = spin_val * 2.0;
        if (p_beta) p_beta[i] = spin_val;
      }
    }

    double alpha_energy = spin_energy_from_packed(norbs, p_alpha, h_core, f_alpha);
    double beta_energy = uhf ? spin_energy_from_packed(norbs, p_beta, h_core, f_beta) : 0.0;
    total_energy = uhf ? (alpha_energy + beta_energy) : (2.0 * alpha_energy);

    density_rms = rms_density_diff(p_total, prev_density.data(), static_cast<size_t>(mpack));
    energy_delta = (!std::isfinite(prev_energy)) ? std::abs(total_energy)
                                                 : std::abs(total_energy - prev_energy);

    prev_energy = total_energy;
    std::copy(p_total, p_total + mpack, prev_density.begin());

    ctx->iterations = iter;
    ctx->energy_total = total_energy;
    ctx->energy_delta = energy_delta;
    ctx->density_rms = density_rms;

    if (verbose) {
      std::fprintf(stderr, "[GPU SCF] iter=%d |dE|=%+.6e rmsP=%.6e\n", iter, energy_delta, density_rms);
      std::fflush(stderr);
    }

    if (energy_delta <= energy_tol && density_rms <= density_tol) {
      std::ostringstream oss;
      oss.setf(std::ios::scientific);
      oss.precision(6);
      oss << "iterations=" << iter << " |dE|=" << energy_delta << " rmsP=" << density_rms;
      set_last_error(oss.str());
      return true;
    }
  }

  char buf[160];
  std::snprintf(buf, sizeof(buf),
                "GPU SCF did not converge (iters=%d |dE|=%.3e rmsP=%.3e)",
                ctx->iterations, energy_delta, density_rms);
  set_last_error(buf);
  return false;
}

extern "C" void mopac_cuda_scf_release() {
  set_last_error("GPU SCF driver reset");
}

extern "C" size_t mopac_cuda_scf_last_error(char *buf, size_t len) {
  std::string copy;
  {
    std::lock_guard<std::mutex> guard(error_mutex());
    copy = last_error();
  }
  if (buf && len > 0) {
    size_t usable = (len > 0) ? len - 1 : 0;
    size_t to_copy = std::min(usable, copy.size());
    if (to_copy > 0) {
      std::memcpy(buf, copy.data(), to_copy);
    }
    if (len > 0) {
      buf[to_copy] = '\0';
    }
  }
  return copy.size();
}

extern "C" bool mopac_cuda_scf_stream_supported() {
  const bool verbose = gpu_logging_enabled();
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (verbose) {
    std::fprintf(stderr, "[GPU STREAM] support probe: err=%d count=%d\n",
                 static_cast<int>(err), count);
    std::fflush(stderr);
  }
  if (err != cudaSuccess || count <= 0) {
    set_last_error("GPU SCF stream unsupported: CUDA device not available");
    return false;
  }
  if (verbose) {
    std::fprintf(stderr, "[GPU STREAM] streaming supported (devices=%d)\n", count);
    std::fflush(stderr);
  }
  return true;
}

extern "C" void mopac_cuda_scf_stream_register(void *cookie_ptr) {
  const bool verbose = gpu_logging_enabled();
  if (verbose) {
    std::fprintf(stderr, "[GPU STREAM] register cookie=%p\n", cookie_ptr);
    std::fflush(stderr);
  }
  if (!cookie_ptr) {
    set_last_error("GPU SCF stream register: cookie pointer is null");
    return;
  }

  std::unique_ptr<StreamSession> session(new StreamSession());
  session->cookie = *static_cast<GpuScfStreamCookie*>(cookie_ptr);
  session->norbs = session->cookie.norbs;
  session->mpack = session->cookie.mpack;
  session->numat = session->cookie.numat;
  session->periodic = (session->cookie.periodic_flag != 0);
  session->has_exchange = (session->cookie.has_exchange != 0);
  session->ptot_host = static_cast<const double*>(session->cookie.ptot);
  session->p_host = static_cast<const double*>(session->cookie.p);
  session->f_host = static_cast<double*>(session->cookie.f);
  session->nfirst_host = static_cast<const int*>(session->cookie.nfirst);
  session->nlast_host = static_cast<const int*>(session->cookie.nlast);

  if (session->norbs <= 0 || session->mpack <= 0 || session->numat <= 0) {
    set_last_error("GPU SCF stream register: invalid problem dimensions");
    return;
  }
  if (verbose) {
    std::fprintf(stderr,
                 "[GPU STREAM] dims: norbs=%d mpack=%d numat=%d periodic=%d\n",
                 session->norbs, session->mpack, session->numat,
                 session->periodic ? 1 : 0);
    std::fflush(stderr);
  }
  if (!session->ptot_host || !session->p_host || !session->f_host ||
      !session->nfirst_host || !session->nlast_host) {
    set_last_error("GPU SCF stream register: required host pointers are null");
    return;
  }

  std::string err_msg;
  if (!build_expected_blocks(*session, err_msg)) {
    if (!err_msg.empty()) set_last_error(err_msg);
    return;
  }
  if (!allocate_stream_buffers(*session, err_msg)) {
    destroy_session(*session);
    if (!err_msg.empty()) set_last_error(err_msg);
    return;
  }

  session->active = true;
  session->error = false;
  session->next_block = 0;
  session->processed_blocks = 0;

  {
    std::lock_guard<std::mutex> guard(stream_mutex());
    auto &map = stream_sessions();
    auto it = map.find(cookie_ptr);
    if (it != map.end()) {
      destroy_session(*it->second);
      map.erase(it);
    }
    map.emplace(cookie_ptr, std::move(session));
  }
}

extern "C" void mopac_cuda_scf_stream_publish(void *cookie_ptr,
                                              int ia, int ib,
                                              int ja, int jb,
                                              int len,
                                              const double *wj,
                                              const double *wk,
                                              int *status) {
  const bool verbose = gpu_logging_enabled();
  if (status) *status = STREAM_STATUS_BAD_ARGS;
  if (!cookie_ptr) {
    set_last_error("GPU SCF stream publish: cookie pointer is null");
    return;
  }

  std::lock_guard<std::mutex> guard(stream_mutex());
  auto it = stream_sessions().find(cookie_ptr);
  if (it == stream_sessions().end()) {
    set_last_error("GPU SCF stream publish: session not registered");
    return;
  }
  StreamSession &session = *it->second;
  if (session.error) {
    if (status) *status = session.error_code != 0 ? session.error_code : STREAM_STATUS_INTERNAL;
    return;
  }
  if (!wj || len <= 0) {
    set_stream_error(session, "GPU SCF stream publish: invalid block payload", STREAM_STATUS_BAD_ARGS);
    if (status) *status = session.error_code;
    return;
  }
  if (verbose) {
    std::fprintf(stderr,
                 "[GPU STREAM] publish #%zu cookie=%p ia=%d ib=%d ja=%d jb=%d len=%d\n",
                 session.next_block, cookie_ptr, ia, ib, ja, jb, len);
    std::fflush(stderr);
  }
  if (session.next_block >= session.blocks.size()) {
    set_stream_error(session, "GPU SCF stream publish: excess blocks supplied", STREAM_STATUS_INTERNAL);
    if (status) *status = session.error_code;
    return;
  }

  ExpectedBlock &blk = session.blocks[session.next_block];
  if (ia != blk.ia || ib != blk.ib || ja != blk.ja || jb != blk.jb) {
    set_stream_error(session, "GPU SCF stream publish: block atom ranges mismatch", STREAM_STATUS_BAD_ARGS);
    if (status) *status = session.error_code;
    return;
  }
  if (len != blk.len_w) {
    set_stream_error(session, "GPU SCF stream publish: block length mismatch", STREAM_STATUS_BAD_ARGS);
    if (status) *status = session.error_code;
    return;
  }

  cudaError_t err;
  size_t coulomb_len = static_cast<size_t>((blk.len_w > 0) ? blk.len_w : blk.len_wj);
  if (coulomb_len > 0) {
    size_t coulomb_cap = std::max(session.max_w_len, session.max_wj_len);
    if (coulomb_len > coulomb_cap) {
      set_stream_error(session, "GPU SCF stream publish: coulomb block exceeds staging capacity", STREAM_STATUS_INTERNAL);
      if (status) *status = session.error_code;
      return;
    }
    if (!session.d_w) {
      set_stream_error(session, "GPU SCF stream publish: coulomb staging unavailable", STREAM_STATUS_INTERNAL);
      if (status) *status = session.error_code;
      return;
    }
    err = cudaMemcpy(session.d_w, wj, sizeof(double) * coulomb_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      set_stream_error(session, cuda_error_message("cudaMemcpy stream coulomb", err), STREAM_STATUS_COPY_FAILED);
      if (status) *status = session.error_code;
      return;
    }
  }

  size_t exchange_len = static_cast<size_t>(blk.len_wk);
  if (exchange_len > 0) {
    if (exchange_len > session.max_wk_len) {
      set_stream_error(session, "GPU SCF stream publish: exchange block exceeds staging capacity", STREAM_STATUS_INTERNAL);
      if (status) *status = session.error_code;
      return;
    }
    const double *wk_src = wk ? wk : wj;
    if (!wk_src) {
      set_stream_error(session, "GPU SCF stream publish: exchange payload missing", STREAM_STATUS_BAD_ARGS);
      if (status) *status = session.error_code;
      return;
    }
    if (!session.d_wk) {
      set_stream_error(session, "GPU SCF stream publish: exchange staging unavailable", STREAM_STATUS_INTERNAL);
      if (status) *status = session.error_code;
      return;
    }
    err = cudaMemcpy(session.d_wk, wk_src, sizeof(double) * exchange_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      set_stream_error(session, cuda_error_message("cudaMemcpy stream exchange", err), STREAM_STATUS_COPY_FAILED);
      if (status) *status = session.error_code;
      return;
    }
  }

  int pair_i_val = blk.ii;
  int pair_j_val = blk.jj;
  int pair_type_val = blk.type;
  int zero = 0;
  if (session.d_pair_i) {
    err = cudaMemcpy(session.d_pair_i, &pair_i_val, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      set_stream_error(session, cuda_error_message("cudaMemcpy pair_i", err), STREAM_STATUS_COPY_FAILED);
      if (status) *status = session.error_code;
      return;
    }
  }
  if (session.d_pair_j) {
    err = cudaMemcpy(session.d_pair_j, &pair_j_val, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      set_stream_error(session, cuda_error_message("cudaMemcpy pair_j", err), STREAM_STATUS_COPY_FAILED);
      if (status) *status = session.error_code;
      return;
    }
  }
  if (session.d_pair_type) {
    err = cudaMemcpy(session.d_pair_type, &pair_type_val, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      set_stream_error(session, cuda_error_message("cudaMemcpy pair_type", err), STREAM_STATUS_COPY_FAILED);
      if (status) *status = session.error_code;
      return;
    }
  }
  if (session.d_pair_w_off) {
    err = cudaMemcpy(session.d_pair_w_off, &zero, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      set_stream_error(session, cuda_error_message("cudaMemcpy pair_w_off", err), STREAM_STATUS_COPY_FAILED);
      if (status) *status = session.error_code;
      return;
    }
  }
  if (session.d_pair_wj_off) {
    err = cudaMemcpy(session.d_pair_wj_off, &zero, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      set_stream_error(session, cuda_error_message("cudaMemcpy pair_wj_off", err), STREAM_STATUS_COPY_FAILED);
      if (status) *status = session.error_code;
      return;
    }
  }
  if (session.d_pair_wk_off) {
    err = cudaMemcpy(session.d_pair_wk_off, &zero, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      set_stream_error(session, cuda_error_message("cudaMemcpy pair_wk_off", err), STREAM_STATUS_COPY_FAILED);
      if (status) *status = session.error_code;
      return;
    }
  }

  double *w_ptr = (blk.len_w > 0) ? session.d_w : nullptr;
  double *wj_ptr = (blk.len_wj > 0) ? session.d_w : ((blk.len_w > 0) ? session.d_w : nullptr);
  double *wk_ptr = (blk.len_wk > 0) ? session.d_wk : nullptr;
  if (!mopac_cuda_launch_pairs_kernel(1,
                                      session.d_pair_i,
                                      session.d_pair_j,
                                      session.d_pair_type,
                                      session.d_pair_w_off,
                                      session.d_pair_wj_off,
                                      session.d_pair_wk_off,
                                      session.d_nfirst,
                                      session.d_nlast,
                                      session.d_ptot,
                                      session.d_p,
                                      w_ptr,
                                      wj_ptr,
                                      wk_ptr,
                                      session.d_f,
                                      0)) {
    set_stream_error(session, "GPU SCF stream publish: pair kernel launch failed", STREAM_STATUS_INTERNAL);
    if (status) *status = session.error_code;
    return;
  }

  session.filled[session.next_block] = 1;
  session.next_block += 1;
  session.processed_blocks += 1;
  if (verbose) {
    std::fprintf(stderr, "[GPU STREAM] publish ok (block %zu/%zu)\n",
                 session.next_block, session.blocks.size());
    std::fflush(stderr);
  }
  if (status) *status = STREAM_STATUS_SUCCESS;
}

extern "C" void mopac_cuda_scf_stream_finalize(void *cookie_ptr, int *status) {
  const bool verbose = gpu_logging_enabled();
  if (status) *status = STREAM_STATUS_INTERNAL;
  if (!cookie_ptr) {
    set_last_error("GPU SCF stream finalize: cookie pointer is null");
    return;
  }

  std::unique_lock<std::mutex> guard(stream_mutex());
  auto map_it = stream_sessions().find(cookie_ptr);
  if (map_it == stream_sessions().end()) {
    set_last_error("GPU SCF stream finalize: session not found");
    if (status) *status = STREAM_STATUS_NOT_READY;
    return;
  }
  if (verbose) {
    std::fprintf(stderr, "[GPU STREAM] finalize cookie=%p\n", cookie_ptr);
    std::fflush(stderr);
  }

  StreamSession &session = *map_it->second;
  int rc = STREAM_STATUS_SUCCESS;

  if (session.error) {
    rc = session.error_code != 0 ? session.error_code : STREAM_STATUS_INTERNAL;
  } else if (session.next_block != session.blocks.size()) {
    set_stream_error(session, "GPU SCF stream finalize: missing blocks detected", STREAM_STATUS_INTERNAL);
    rc = session.error_code;
  } else if (session.processed_blocks != session.blocks.size()) {
    set_stream_error(session, "GPU SCF stream finalize: incomplete kernel accumulation", STREAM_STATUS_INTERNAL);
    rc = session.error_code;
  } else {
    size_t bytes = sizeof(double) * static_cast<size_t>(session.mpack);
    bool resident = (mopac_cuda_get_resident_mode() != 0);
    if (resident) {
      mopac_cuda_register_fock_device(session.mpack, session.f_host, session.d_f);
      if (!mopac_cuda_fetch_fock(session.f_host, static_cast<size_t>(session.mpack))) {
        cudaError_t err = cudaMemcpy(session.f_host, session.d_f, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
          set_stream_error(session, cuda_error_message("cudaMemcpy fock resident fallback", err), STREAM_STATUS_COPY_FAILED);
          rc = session.error_code;
        }
      }
    } else {
      cudaError_t err = cudaMemcpy(session.f_host, session.d_f, bytes, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        set_stream_error(session, cuda_error_message("cudaMemcpy fock", err), STREAM_STATUS_COPY_FAILED);
        rc = session.error_code;
      } else {
        mopac_cuda_clear_fock_cache();
      }
    }
  }

  destroy_session(session);
  stream_sessions().erase(map_it);
  guard.unlock();

  if (status) *status = rc;
  if (verbose) {
    std::fprintf(stderr, "[GPU STREAM] finalize complete status=%d\n", rc);
    std::fflush(stderr);
  }
}
