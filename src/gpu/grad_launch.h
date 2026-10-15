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
