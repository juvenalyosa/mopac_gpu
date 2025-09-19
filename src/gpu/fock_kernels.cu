// Minimal scaffolding for GPU Fock build (to be implemented)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

extern "C" {

// Packed inputs; this stub currently returns false and leaves F unchanged.
bool mopac_cuda_fock2(int norbs, int mpack, int numat,
                      const int *nfirst, const int *nlast,
                      const double *ptot, const double *p,
                      double *f) {
  (void)norbs; (void)mpack; (void)numat; (void)nfirst; (void)nlast; (void)ptot; (void)p; (void)f;
  // TODO: implement tiled J/K accumulation on GPU.
  return false;
}

}

