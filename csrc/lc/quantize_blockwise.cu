// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/extension.h"
#include<stdlib.h>
#include<string.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/mman.h>
#include<stdio.h>
#include<algorithm>
#include<cub/device/device_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <mma.h>
#include "common.h"

__device__ unsigned char dQuantizeNF4(float x)
{

  // the values for this tree was generated by test_normal_map_tree
  // in the file tests/test_functional.py
  if(x > 0.03979014977812767f)
    if(x > 0.3893125355243683f) // 1
      if(x > 0.6427869200706482f) // 11
        if(x > 0.8614784181118011f) // 111
          return 0b1111;
        else
          return 0b1110;
      else
        if(x > 0.5016634166240692f) // 110
          return 0b1101;
        else
          return 0b1100;
    else
      if(x > 0.2035212516784668f) // 10
        if(x > 0.2920137718319893f) // 101
          return 0b1011;
        else
          return 0b1010;
      else
        if(x > 0.1202552504837513f) // 100
          return 0b1001;
        else
          return 0b1000;
  else
    if(x > -0.33967943489551544f) // 0
      if(x > -0.13791173323988914f) // 01
        if(x > -0.045525018125772476f) // 011
          return 0b0111;
        else
          return 0b0110;
      else
        if(x > -0.23460740596055984f) // 010
          return 0b0101;
        else
          return 0b0100;
    else
      if(x > -0.6106329262256622f) // 00
        if(x > -0.4599952697753906f) // 001
          return 0b0011;
        else
          return 0b0010;
      else
        if(x > -0.8480964004993439f) // 000
          return 0b0001;
        else
          return 0b0000;
}
__device__ unsigned char dQuantizeFP4(float x)
{
  // FP4 with bias of 3
  // first bit is a sign
  // subnormals
  // 0b000 = 0
  // 0b001 = 0.0625
  // 0b110 = 2
  // 0b111 = 3
  // 0b100 = 4
  // 0b101 = 6
  // 0b010 = 8
  // 0b011 = 12


  // we do a binary search
  // the pivots are divided by 12 (the FP4 absmax)
  // since we assum input data is in [-1.0, 1.0]

  // !be careful here, its easy to make a mistake
  // that is difficult to noice if you add an extra
  // zero somewhere!

  int sign = x < 0 ? 0b1000 : 0b0000;
  x = fabsf(x);
  if(x > 0.29166667f)
    if( x > 0.583333f)
      if( x > 0.8333333f)
        return 0b0011+sign;
      else
        return 0b0010+sign;
    else
      if(x > 0.4166667f)
        return 0b101+sign;
      else
        return 0b100+sign;
  else
    if(x > 0.0859375f)
      if(x > 0.20833333f)
        return 0b0111+sign;
      else
        return 0b0110+sign;
    else
      if(x > 0.00260417f)
        return 0b0001+sign;
      else
        return 0b0000+sign;
}

__device__ unsigned char dQuantize(float* smem_code, float x)
{
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    float lower = -1.0f;
    float upper = 1.0f;

    float val = smem_code[pivot];
    // i>>=1 = {32, 16, 8, 4, 2, 1}
    for(int i = 64; i > 0; i>>=1)
    {
        if(x > val)
        {
            lower_pivot = pivot;
            lower = val;
            pivot+=i;
        }
        else
        {
            upper_pivot = pivot;
            upper = val;
            pivot-=i;
        }
        val = smem_code[pivot];
    }

    if(upper_pivot == 255)
        upper = smem_code[upper_pivot];
    if(lower_pivot == 0)
        lower = smem_code[lower_pivot];

   if(x > val)
   {
     float midpoint = (upper+val)*0.5f;
     if(x > midpoint)
     {
       return upper_pivot;
     }
     else
       return pivot;
   }
   else
   {
     float midpoint = (lower+val)*0.5f;
     if(x < midpoint)
       return lower_pivot;
     else
       return pivot;
   }
}


template<typename T, int BLOCK_SIZE, int NUM_PER_TH, int DATA_TYPE>
//__launch_bounds__(TH, 4)
__global__ void kQuantizeBlockwise(const float * code, const T * __restrict__ A, float *absmax, unsigned char *out, int n)
{
  const int n_full = gridDim.x * BLOCK_SIZE;
  int valid_items = 0;
  const int base_idx = (blockIdx.x * BLOCK_SIZE);

  T vals[NUM_PER_TH];
  unsigned char qvals[(DATA_TYPE > 0) ? NUM_PER_TH/2 : NUM_PER_TH];
  //float local_abs_max = -FLT_MAX;
  float local_abs_max = 0.0f;

  typedef cub::BlockLoad<T, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
  typedef cub::BlockStore<unsigned char, BLOCK_SIZE/NUM_PER_TH, (DATA_TYPE > 0) ? NUM_PER_TH/2 : NUM_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
  typedef cub::BlockReduce<float, BLOCK_SIZE/NUM_PER_TH> BlockReduce;
  typedef cub::BlockLoad<float, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;

  __shared__ typename LoadT::TempStorage loadt;
  __shared__ typename LoadFloat::TempStorage loadf;
  __shared__ typename StoreChar::TempStorage storec;
  __shared__ typename BlockReduce::TempStorage reduce;
  __shared__ float smem_code[256];
  __shared__ float smem_absmax_value[1];

  if(DATA_TYPE == General8bit)
    for(int i = threadIdx.x; i < 256; i+=blockDim.x)
      smem_code[i] = code[i];

  for (unsigned int i = base_idx; i < n_full; i += gridDim.x*BLOCK_SIZE)
  {
    valid_items = n - i > BLOCK_SIZE ? BLOCK_SIZE : n - i;
    local_abs_max = -FLT_MAX;

    __syncthreads();
    LoadT(loadt).Load(&(A[i]), vals, valid_items, (T)0.0f);

    // 1. compute local max
    // 2. broadcast local max
    // 3. normalize inputs and quantize

    #pragma unroll NUM_PER_TH
    for(int j = 0; j < NUM_PER_TH; j++)
       local_abs_max = fmaxf(local_abs_max, fabsf((float)vals[j]));

    local_abs_max = BlockReduce(reduce).Reduce(local_abs_max, cub::Max(), valid_items);

    if(threadIdx.x == 0)
      smem_absmax_value[0] = local_abs_max;

    __syncthreads();

    if(threadIdx.x == 0)
      absmax[i/BLOCK_SIZE] = local_abs_max;
    else
      local_abs_max = smem_absmax_value[0];

    __syncwarp();

    local_abs_max = 1.0f/local_abs_max;

    unsigned char packed_4bit = 0;
    switch(DATA_TYPE)
    {
        case General8bit:
            #pragma unroll NUM_PER_TH
            for(int j = 0; j < NUM_PER_TH; j++)
            {
               qvals[j] = dQuantize(smem_code, ((float)vals[j])*local_abs_max);
            }
            break;
        case FP4:
            #pragma unroll NUM_PER_TH
            for(int j = 0; j < NUM_PER_TH/2; j++)
            {
              packed_4bit |= dQuantizeFP4(((float)vals[2*j])*local_abs_max) << 4;
              packed_4bit |= dQuantizeFP4(((float)vals[2*j+1])*local_abs_max);
              qvals[j] = packed_4bit;
            }
            break;
        case NF4:
            #pragma unroll NUM_PER_TH
            for(int j = 0; j < NUM_PER_TH/2; j++)
            {
              packed_4bit = 0;
              packed_4bit |= dQuantizeNF4(((float)vals[2*j])*local_abs_max) << 4;
              packed_4bit |= dQuantizeNF4(((float)vals[2*j+1])*local_abs_max);
              qvals[j] = packed_4bit;
            }
            break;
    }

    __syncthreads();
    StoreChar(storec).Store(&(out[(DATA_TYPE > 0) ? i/2 : i]), qvals, (DATA_TYPE > 0) ? (valid_items+1)/2 : valid_items);
  }
}


#define MAKE_kQuantizeBlockwise(dtype, blocksize, num_per_thread, data_type_name) \
template __global__ void kQuantizeBlockwise<dtype, blocksize, num_per_thread, data_type_name>(const float * code, const dtype * __restrict__ A, float *absmax, unsigned char *out, int n); \

MAKE_kQuantizeBlockwise(half,  4096, 4, General8bit)
MAKE_kQuantizeBlockwise(half,  2048, 4, General8bit)
MAKE_kQuantizeBlockwise(half,  1024, 4, General8bit)
MAKE_kQuantizeBlockwise(half,   512, 2, General8bit)
MAKE_kQuantizeBlockwise(half,   256, 2, General8bit)
MAKE_kQuantizeBlockwise(half,   128, 2, General8bit)
MAKE_kQuantizeBlockwise(half,    64, 2, General8bit)
MAKE_kQuantizeBlockwise(half,  4096, 4, FP4)
MAKE_kQuantizeBlockwise(half,  2048, 4, FP4)
MAKE_kQuantizeBlockwise(half,  1024, 4, FP4)
MAKE_kQuantizeBlockwise(half,   512, 2, FP4)
MAKE_kQuantizeBlockwise(half,   256, 2, FP4)
MAKE_kQuantizeBlockwise(half,   128, 2, FP4)
MAKE_kQuantizeBlockwise(half,    64, 2, FP4)
MAKE_kQuantizeBlockwise(half,  4096, 4, NF4)
MAKE_kQuantizeBlockwise(half,  2048, 4, NF4)
MAKE_kQuantizeBlockwise(half,  1024, 4, NF4)
MAKE_kQuantizeBlockwise(half,   512, 2, NF4)
MAKE_kQuantizeBlockwise(half,   256, 2, NF4)
MAKE_kQuantizeBlockwise(half,   128, 2, NF4)
MAKE_kQuantizeBlockwise(half,    64, 2, NF4)
MAKE_kQuantizeBlockwise(float, 4096, 4, General8bit)
MAKE_kQuantizeBlockwise(float, 2048, 4, General8bit)
MAKE_kQuantizeBlockwise(float, 1024, 4, General8bit)
MAKE_kQuantizeBlockwise(float,  512, 2, General8bit)
MAKE_kQuantizeBlockwise(float,  256, 2, General8bit)
MAKE_kQuantizeBlockwise(float,  128, 2, General8bit)
MAKE_kQuantizeBlockwise(float,   64, 2, General8bit)
MAKE_kQuantizeBlockwise(float, 4096, 4, FP4)
MAKE_kQuantizeBlockwise(float, 2048, 4, FP4)
MAKE_kQuantizeBlockwise(float, 1024, 4, FP4)
MAKE_kQuantizeBlockwise(float,  512, 2, FP4)
MAKE_kQuantizeBlockwise(float,  256, 2, FP4)
MAKE_kQuantizeBlockwise(float,  128, 2, FP4)
MAKE_kQuantizeBlockwise(float,   64, 2, FP4)
MAKE_kQuantizeBlockwise(float, 4096, 4, NF4)
MAKE_kQuantizeBlockwise(float, 2048, 4, NF4)
MAKE_kQuantizeBlockwise(float, 1024, 4, NF4)
MAKE_kQuantizeBlockwise(float,  512, 2, NF4)
MAKE_kQuantizeBlockwise(float,  256, 2, NF4)
MAKE_kQuantizeBlockwise(float,  128, 2, NF4)
MAKE_kQuantizeBlockwise(float,   64, 2, NF4)

MAKE_kQuantizeBlockwise(__nv_bfloat16, 4096, 4, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 2048, 4, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 1024, 4, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  512, 2, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  256, 2, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  128, 2, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16,   64, 2, General8bit)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 4096, 4, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 2048, 4, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 1024, 4, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  512, 2, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  256, 2, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  128, 2, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,   64, 2, FP4)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 4096, 4, NF4)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 2048, 4, NF4)
MAKE_kQuantizeBlockwise(__nv_bfloat16, 1024, 4, NF4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  512, 2, NF4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  256, 2, NF4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,  128, 2, NF4)
MAKE_kQuantizeBlockwise(__nv_bfloat16,   64, 2, NF4)

template <typename T, int DATA_TYPE>
__global__ void kQuantizeChannelwise(const float *code,
                                      const T* A,
                                      unsigned char* out,
                                      float *absmax,
                                      int n,
                                      int cout) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  int num = n / 2;
  for (int i = idx; i < num; i += blockDim.x * gridDim.x) {
    int idx = 2*(i/cout)* cout + i%cout;
    float local_absmax = absmax[i %cout];
    float inv_local_absmax = 1.0f/local_absmax;

    unsigned char packed_4bit = 0;
    switch(DATA_TYPE)
    {
        case FP4:
            packed_4bit |= dQuantizeFP4(((float)A[idx])*inv_local_absmax) << 4;
            packed_4bit |= dQuantizeFP4(((float)A[idx+cout])*inv_local_absmax);
            out[i] = packed_4bit;
            break;
        case NF4:
            packed_4bit |= dQuantizeNF4(((float)A[idx])*inv_local_absmax) << 4;
            packed_4bit |= dQuantizeNF4(((float)A[idx+cout])*inv_local_absmax);
            out[i] = packed_4bit;
            break;
    }
  }
}

template <paddle::DataType D, int DATA_TYPE> void quantize_blockwise(const float *code, const paddle::Tensor& A, paddle::Tensor& absmax, unsigned char *out, int blocksize, int n, int channelwise)
{
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;

  const DataType_* A_data = reinterpret_cast<const DataType_*>(A.data<data_t>()); 
  if (channelwise == 0) {
    if(blocksize == 4096)
      kQuantizeBlockwise<DataType_, 4096, 4, 0><<<num_blocks, 1024>>>(code, A_data, absmax.data<float>(), out, n);
    else if(blocksize == 2048)
      kQuantizeBlockwise<DataType_, 2048, 4, DATA_TYPE><<<num_blocks, 512>>>(code, A_data, absmax.data<float>(), out, n);
    else if(blocksize == 1024)
      kQuantizeBlockwise<DataType_, 1024, 4, DATA_TYPE><<<num_blocks, 256>>>(code, A_data, absmax.data<float>(), out, n);
    else if(blocksize == 512)
      kQuantizeBlockwise<DataType_, 512, 2, DATA_TYPE><<<num_blocks, 256>>>(code, A_data, absmax.data<float>(), out, n);
    else if(blocksize == 256)
      kQuantizeBlockwise<DataType_, 256, 2, DATA_TYPE><<<num_blocks, 128>>>(code, A_data, absmax.data<float>(), out, n);
    else if(blocksize == 128)
      kQuantizeBlockwise<DataType_, 128, 2, DATA_TYPE><<<num_blocks, 64>>>(code, A_data, absmax.data<float>(), out, n);
    else if(blocksize == 64)
      kQuantizeBlockwise<DataType_, 64, 2, DATA_TYPE><<<num_blocks, 32>>>(code, A_data, absmax.data<float>(), out, n);
  }
  else {
    if (DATA_TYPE == General8bit)
        PD_THROW("blocksize is -1 only support NF4 and FP4.");

    int cout = A.shape()[1];
    int max_threads = 1024; 

    absmax = A.abs().max({0});

    int64_t block_size =
        std::min(static_cast<int64_t>(n),
                 static_cast<int64_t>(max_threads/ 4));

    const int64_t max_blocks =
        std::max(((max_threads - 1) / block_size + 1), static_cast<int64_t>(1));
    const int64_t grid_size =
        std::min(max_blocks, (n + block_size - 1) / block_size);

    kQuantizeChannelwise<DataType_, DATA_TYPE><<<grid_size, block_size, 0>>>(
      code, A_data, out, absmax.data<float>(), n, cout);
  }


  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

std::vector<paddle::Tensor> QuantizeBlockwise(const paddle::Tensor& input, const paddle::Tensor& code, int blocksize, std::string quant_type) {
    int n = input.numel();
    int channelwise = 0;
    std::vector<int64_t> out_shape = input.shape();
    if (quant_type != "8bit") { // 4bit
        out_shape = {(n + 1) / 2, 1};
    }
    if (blocksize == -1){
        blocksize = input.shape()[0];
        out_shape = {input.shape()[0]/2, input.shape()[1]};
        channelwise = 1;
    }
    auto out = paddle::empty(out_shape, paddle::DataType::UINT8, input.place());
    int64_t absmax_shape = n / blocksize;
    auto absmax = paddle::empty({absmax_shape}, paddle::DataType::FLOAT32, input.place());
    switch(input.type()) {
        case paddle::DataType::FLOAT32:
            if (quant_type == "8bit")
                quantize_blockwise<paddle::DataType::FLOAT32, General8bit>(code.data<float>(), input, absmax, out.data<unsigned char>(), blocksize, n, channelwise);
            else if (quant_type == "nf4") {
                quantize_blockwise<paddle::DataType::FLOAT32, NF4>(NULL, input, absmax, out.data<unsigned char>(), blocksize, n, channelwise);
            }
            else if (quant_type == "fp4")
                quantize_blockwise<paddle::DataType::FLOAT32, FP4>(NULL, input, absmax, out.data<unsigned char>(), blocksize, n, channelwise);
            return {out, absmax};
        case paddle::DataType::FLOAT16:
            if (quant_type == "8bit")
                quantize_blockwise<paddle::DataType::FLOAT16, General8bit>(code.data<float>(), input, absmax, out.data<unsigned char>(), blocksize, n, channelwise);
            else if (quant_type == "nf4")
                quantize_blockwise<paddle::DataType::FLOAT16, NF4>(NULL, input, absmax, out.data<unsigned char>(), blocksize, n, channelwise);
            else if (quant_type == "fp4")
                quantize_blockwise<paddle::DataType::FLOAT16, FP4>(NULL, input, absmax, out.data<unsigned char>(), blocksize, n, channelwise);
            return {out, absmax};
        case paddle::DataType::BFLOAT16:
            if (quant_type == "8bit")
                quantize_blockwise<paddle::DataType::BFLOAT16, General8bit>(code.data<float>(), input, absmax, out.data<unsigned char>(), blocksize, n, channelwise);
            else if (quant_type == "nf4")
                quantize_blockwise<paddle::DataType::BFLOAT16, NF4>(NULL, input, absmax, out.data<unsigned char>(), blocksize, n, channelwise);
            else if (quant_type == "fp4")
                quantize_blockwise<paddle::DataType::BFLOAT16, FP4>(NULL, input, absmax, out.data<unsigned char>(), blocksize, n, channelwise);
            return {out, absmax};

        default:
            PD_THROW(
                "NOT supported data type. "
                "Only float16, bfloat16 and float32 are supported. ");
            break;
    }
};

std::vector<std::vector<int64_t>> GetQuantizeBlockwiseInferShape(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& code_shape, int blocksize, std::string quant_type){
    int64_t first_shape = (input_shape[0] * input_shape[1] + 1) / 2;
    if (quant_type != "8bit")
        if (blocksize != -1)
          return {{first_shape, 1}};
        else 
          return {{input_shape[0]/2, input_shape[1]}};
    else
        return {input_shape};
}

std::vector<paddle::DataType> GetQuantizeBlockwiseInferDtype(const paddle::DataType& input_dtype, const paddle::DataType& code_dtype){
    return {paddle::DataType::UINT8};
}

PD_BUILD_OP(quant_blockwise)
    .Inputs({"input", "code"})
    .Outputs({"output", "abs_max"})
    .Attrs({"blocksize: int", "quant_type: std::string"})
    .SetKernelFn(PD_KERNEL(QuantizeBlockwise))
    .SetInferShapeFn(PD_INFER_SHAPE(GetQuantizeBlockwiseInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetQuantizeBlockwiseInferDtype));
