#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {
struct MopacGpuScfContext {
  int    norbs;
  int    nalpha;
  int    nbeta;
  int    mpack;
  int    max_iter;
  double energy_tol;
  double density_tol;
  void  *h_core;
  void  *overlap;
  void  *density_alpha;
  void  *density_beta;
  void  *fock_alpha;
  void  *fock_beta;
  void  *work;
  void  *log_buffer;
  int    flags;
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

bool stub_logging_enabled() {
  const char *env = std::getenv("MOPAC_GPU_SCF_STUB_LOG");
  if (!env || *env == '\0') return false;
  return !(std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0 || std::strcmp(env, "false") == 0);
}
} // namespace

extern "C" bool mopac_cuda_scf_run(MopacGpuScfContext *ctx) {
  if (!ctx) {
    set_last_error("GPU SCF context pointer is null");
    return false;
  }
  if (stub_logging_enabled()) {
    std::fprintf(stderr,
                 "[GPU SCF] stub invoked: norbs=%d nalpha=%d nbeta=%d max_iter=%d energy_tol=%g density_tol=%g\n",
                 ctx->norbs, ctx->nalpha, ctx->nbeta, ctx->max_iter,
                 ctx->energy_tol, ctx->density_tol);
    std::fflush(stderr);
  }
  set_last_error("GPU SCF driver stub: functionality not implemented yet");
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
