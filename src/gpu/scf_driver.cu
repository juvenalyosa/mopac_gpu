#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

extern "C" bool mopac_cuda_fock2_scf(int norbs, int mpack, int numat,
                                      const int *nfirst, const int *nlast,
                                      const double *ptot, const double *p,
                                      const double *w, const double *wj, const double *wk,
                                      int periodic_flag,
                                      double *fout);

extern "C" void mopac_cuda_dsyevd_keep(int n, double *A, int lda, double *W, int *info);
extern "C" void mopac_cuda_fetch_eigenvectors(int n, double *A, int lda);
extern "C" void mopac_cuda_density_from_dev_syrk(int n, int ndubl, double alpha, double *C, int ldc);

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
    set_last_error("GPU SCF: number of atoms not provided");
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
