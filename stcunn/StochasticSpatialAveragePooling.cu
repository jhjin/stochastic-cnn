#include "utils.h"
#include "common.h"

template <typename Dtype, bool COUNT_INCLUDE_PAD>
__global__ void AvePoolForward(const int nthreads,
    const Dtype* const bottom_mu, const Dtype* const bottom_var,
    Dtype* const top_mu, Dtype* const top_var,
    const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);

    Dtype sum_mu = 0;
    Dtype sum_var = 0;
    const Dtype* const bottom_mu_slice = bottom_mu + (n * channels + c) * height * width;
    const Dtype* const bottom_var_slice = bottom_var + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        sum_mu += bottom_mu_slice[h * width + w];
        sum_var += bottom_var_slice[h * width + w];
      }
    }

    int divide_factor;
    if(COUNT_INCLUDE_PAD)
      divide_factor = pool_size;
    else
      divide_factor = (hend - hstart) * (wend - wstart);

    top_mu[index] = sum_mu / divide_factor;
    top_var[index] = sum_var / (divide_factor * divide_factor);
  }
}

static int stcunn_StochasticSpatialAveragePooling_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input_mu = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *input_var = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  bool ceil_mode = luaT_getfieldcheckboolean(L, 1, "ceil_mode");
  bool count_include_pad = luaT_getfieldcheckboolean(L, 1, "count_include_pad");

  THCudaTensor *output_mu = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "mu", "torch.CudaTensor");
  THCudaTensor *output_var = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "var", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, input_mu, input_var, output_mu, output_var));
  THArgCheck(input_mu->nDimension == 3 || input_mu->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;

  if (input_mu->nDimension == 3) {
    nInputCols = input_mu->size[2];
    nInputRows = input_mu->size[1];
    nInputPlane = input_mu->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input_mu->size[3];
    nInputRows = input_mu->size[2];
    nInputPlane = input_mu->size[1];
    batchSize = input_mu->size[0];
  }

  THArgCheck(nInputCols >= kW - 2*padW && nInputRows >= kH - 2*padH, 2, "input image smaller than kernel size");
  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");

  if(ceil_mode) {
    nOutputCols = ceil(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = ceil(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  }
  else {
    nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  }
  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((nOutputRows - 1)*dH >= nInputRows + padH)
      --nOutputRows;
    if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
      --nOutputCols;
  }

  input_mu = THCudaTensor_newContiguous(state, input_mu);
  input_var = THCudaTensor_newContiguous(state, input_var);
  float* input_mu_data = THCudaTensor_data(state, input_mu);
  float* input_var_data = THCudaTensor_data(state, input_var);

  THCudaTensor_resize4d(state, output_mu, batchSize, nInputPlane, nOutputRows, nOutputCols);
  THCudaTensor_resize4d(state, output_var, batchSize, nInputPlane, nOutputRows, nOutputCols);

  float* output_mu_data = THCudaTensor_data(state, output_mu);
  float* output_var_data = THCudaTensor_data(state, output_var);

  int count = THCudaTensor_nElement(state, output_mu);

  if(count_include_pad)
    AvePoolForward<float, true>
      <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>(
        count, input_mu_data, input_var_data, output_mu_data, output_var_data,
        batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
        kH, kW, dH, dW, padH, padW);
  else
    AvePoolForward<float, false>
      <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>(
        count, input_mu_data, input_var_data, output_mu_data, output_var_data,
        batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
        kH, kW, dH, dW, padH, padW);

  if(input_mu->nDimension == 3) {
    THCudaTensor_resize3d(state, output_mu, nInputPlane, nOutputRows, nOutputCols);
    THCudaTensor_resize3d(state, output_var, nInputPlane, nOutputRows, nOutputCols);
  }

  THCudaTensor_free(state, input_mu);
  THCudaTensor_free(state, input_var);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialAveragePooling.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 2;
}
static const struct luaL_Reg stcunn_StochasticSpatialAveragePooling__ [] = {
  {"StochasticSpatialAveragePooling_updateOutput", stcunn_StochasticSpatialAveragePooling_updateOutput},
  {NULL, NULL}
};

static void stcunn_StochasticSpatialAveragePooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, stcunn_StochasticSpatialAveragePooling__, "nn");
  lua_pop(L,1);
}
