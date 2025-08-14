/*
 * Copyright (c) 2026, Cuhksz DragonPass. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef __DPRM_RESIZE_CUH__
#define __DPRM_RESIZE_CUH__

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "cudatools.h"

struct AffineMatrix {
    float value[6];
};

extern AffineMatrix input_to_infer;
extern AffineMatrix infer_to_input;

void generate_affine_matrix(int src_width, int src_height, int dst_width, int dst_height);

__global__ void warpaffine_kernel(
    uint8_t* src,
    int src_line_size,
    int src_width,
    int src_height,
    float* dst,
    int dst_width,
    int dst_height,
    uint8_t const_value_st,
    AffineMatrix d2s,
    int edge
);

#endif
