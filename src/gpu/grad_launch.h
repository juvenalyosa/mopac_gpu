#pragma once

#include <cuda_runtime.h>

struct AtomRange {
  int first;
  int last;
};

struct AtomPairWork {
  int atom_i;
  int atom_j;
  AtomRange range_i;
  AtomRange range_j;
};

struct GradPairPod {
  int atom_i;
  int atom_j;
  int span_i_first;
  int span_i_last;
  int span_j_first;
  int span_j_last;
  int image_code;
  int flags;
  double displacement[3];
  double distance2;
  double weight;
};

bool resident_grad_launch_impl(int numat,
                               int l123,
                               const double *coord,
                               double *grad,
                               const double *charges,
                               const GradPairPod *near_pairs,
                               int near_count,
                               const GradPairPod *far_pairs,
                               int far_count);

void resident_grad_release_impl();

void generate_pair_work(int numat,
                        int l123,
                        const GradPairPod *pairs,
                        int pair_count,
                        AtomPairWork *out_work,
                        int *out_count);
