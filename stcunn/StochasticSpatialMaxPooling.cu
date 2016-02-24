#include "utils.h"
#include "common.h"

template <typename Dtype>
__device__ void maxpool_bubblesort(Dtype *a, Dtype *b, int n) {
  int i, j;
  Dtype temp;

  for (i = 1; i < n; i++) {
    for (j = 0; j < n- 1; j++) {
      if (a[j] > a[j + 1]) {
        temp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = temp;
        temp = b[j];
        b[j] = b[j + 1];
        b[j + 1] = temp;
      }
    }
  }
}

// kernels borrowed from Caffe
template <typename Dtype, bool SORT>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* bottom_mu, const Dtype* bottom_var, Dtype* top_mu, Dtype* top_var,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    // assume 3x3 maxpool at most
    Dtype mu_sorted[9];
    Dtype var_sorted[9];

    bottom_mu += (n * channels + c) * height * width;
    bottom_var += (n * channels + c) * height * width;

    int cnt = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        mu_sorted[cnt] = bottom_mu[h * width + w];
        var_sorted[cnt] = bottom_var[h * width + w];
	cnt++;
      }
    }

    // ascending order
    if (SORT)
      maxpool_bubblesort(mu_sorted, var_sorted, cnt);

    Dtype mu1 = mu_sorted[0];
    Dtype var1 = fmaxf(var_sorted[0], 1e-20);
    for(int k = 1; k < cnt; k++) {
      Dtype mu2 = mu_sorted[k];
      Dtype var2 = fmaxf(var_sorted[k], 1e-20);

      Dtype theta = sqrtf(var1 + var2);
      Dtype alpha12 = fdividef(mu1 - mu2, theta);

      Dtype cdf12 = normcdff( alpha12); // 0.5*(1.+erf( alpha12/sqrt(2.)));
      Dtype cdf21 = normcdff(-alpha12); // 0.5*(1.+erf(-alpha12/sqrt(2.)));
      Dtype pdf12 = fdividef(expf(fdividef(-1.*alpha12*alpha12, 2.)), sqrtf(2.*M_PI));

      Dtype t_mu = mu1*cdf12 + mu2*cdf21 + theta*pdf12;
      Dtype t_var = fmaxf((var1+mu1*mu1)*cdf12 + (var2+mu2*mu2)*cdf21 + (mu1+mu2)*theta*pdf12 - t_mu*t_mu, 1e-20);

      mu1 = t_mu;
      var1 = t_var;
    }

    top_mu[index] = mu1;
    top_var[index] = var1;
  }
}


static int stcunn_StochasticSpatialMaxPooling_updateOutput(lua_State *L)
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
  bool sort = luaT_getfieldcheckboolean(L, 1, "sort");
  bool ceil_mode = luaT_getfieldcheckboolean(L, 1, "ceil_mode");

  THCudaTensor *output_mu = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "mu", "torch.CudaTensor");
  THCudaTensor *output_var = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "var", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, input_mu, input_var, output_mu, output_var));
  luaL_argcheck(L, input_mu->nDimension == 3 || input_mu->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

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

  THArgCheck(nInputCols >= kW - padW && nInputRows >= kH - padH, 2, "input image smaller than kernel size");
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

  if(sort)
    MaxPoolForward<float, true>
      <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count, input_mu_data, input_var_data, output_mu_data, output_var_data,
        batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
        kH, kW, dH, dW, padH, padW);
  else
    MaxPoolForward<float, false>
      <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count, input_mu_data, input_var_data, output_mu_data, output_var_data,
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
    printf("error in SpatialMaxPooling.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 2;
}

static const struct luaL_Reg stcunn_StochasticSpatialMaxPooling__ [] = {
  {"StochasticSpatialMaxPooling_updateOutput", stcunn_StochasticSpatialMaxPooling_updateOutput},
  {NULL, NULL}
};

static void stcunn_StochasticSpatialMaxPooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, stcunn_StochasticSpatialMaxPooling__, "nn");
  lua_pop(L,1);
}
