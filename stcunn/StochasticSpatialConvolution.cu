#include "utils.h"
#include "common.h"

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
__global__ void im2col_kernel(const int n, const float* data_im,
    const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col,
    float* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize_h * ksize_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize_h; ++i) {
      for (int j = 0; j < ksize_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
          data_im[i * width + j] : 0;
        data_col += height_col * width_col;
      }
    }
  }
}

void im2col(cudaStream_t stream, const float* data_im, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, float* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - ksize_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - ksize_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // Launch
  im2col_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernels, data_im, height, width, ksize_h, ksize_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_col
  );
}

static int stcunn_StochasticSpatialConvolution_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);
  // Input
  THCudaTensor *input_mu = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *input_var = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

  // Params:
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *weight2 = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight2", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *columns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", "torch.CudaTensor");
  THCudaTensor *mu = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "mu", "torch.CudaTensor");
  THCudaTensor *var = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "var", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 9, input_mu, input_var, mu, var, weight, weight2, bias, columns, ones));
  luaL_argcheck(L, input_mu->nDimension == 3 || input_mu->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  int batch = 1;
  if (input_mu->nDimension == 3) {
    luaL_argcheck(L, input_mu->size[0] == nInputPlane, 2, "input channels and nInputPlane dont match");
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input_mu, 1, input_mu->size[0], input_mu->size[1], input_mu->size[2]);
    THCudaTensor_resize4d(state, input_var, 1, input_var->size[0], input_var->size[1], input_var->size[2]);
  } else {
    luaL_argcheck(L, input_mu->size[1] == nInputPlane, 2, "input channels and nInputPlane dont match");
  }

  long inputWidth   = input_mu->size[3];
  long inputHeight  = input_mu->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;


  // Batch size + input planes
  long batchSize = input_mu->size[0];

  // Resize output
  THCudaTensor_resize4d(state, mu, batchSize, nOutputPlane, outputHeight, outputWidth);
  THCudaTensor_resize4d(state, var, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Resize temporary columns
  THCudaTensor_resize2d(state, columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCudaTensor_resize2d(state, ones, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  // Helpers
  THCudaTensor *input_mu_n = THCudaTensor_new(state);
  THCudaTensor *input_var_n = THCudaTensor_new(state);
  THCudaTensor *output_mu_n = THCudaTensor_new(state);
  THCudaTensor *output_var_n = THCudaTensor_new(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCudaTensor_select(state, input_mu_n, input_mu, 0, elt);
    THCudaTensor_select(state, input_var_n, input_var, 0, elt);
    THCudaTensor_select(state, output_mu_n, mu, 0, elt);
    THCudaTensor_select(state, output_var_n, var, 0, elt);


    // var
    im2col(
      THCState_getCurrentStream(state),
      THCudaTensor_data(state, input_var_n),
      nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
      THCudaTensor_data(state, columns)
    );

    long m = weight->size[0];
    long n = columns->size[1];
    long k = weight->size[1];

    THCudaBlas_gemm(
        state,
        'n', 'n',
        n, m, k,
        1,
        THCudaTensor_data(state, columns), n,
        THCudaTensor_data(state, weight2), k,
        0,
        THCudaTensor_data(state, output_var_n), n
    );


    // mu
    long m_ = nOutputPlane;
    long n_ = outputHeight * outputWidth;
    long k_ = 1;

    THCudaBlas_gemm(
        state,
        't', 'n',
        n_, m_, k_,
        1,
        THCudaTensor_data(state, ones), k_,
        THCudaTensor_data(state, bias), k_,
        0,
        THCudaTensor_data(state, output_mu_n), n_
    );

    im2col(
      THCState_getCurrentStream(state),
      THCudaTensor_data(state, input_mu_n),
      nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
      THCudaTensor_data(state, columns)
    );

    THCudaBlas_gemm(
        state,
        'n', 'n',
        n, m, k,
        1,
        THCudaTensor_data(state, columns), n,
        THCudaTensor_data(state, weight), k,
        1,
        THCudaTensor_data(state, output_mu_n), n
    );
  }

  // Free
  THCudaTensor_free(state, input_mu_n);
  THCudaTensor_free(state, input_var_n);
  THCudaTensor_free(state, output_mu_n);
  THCudaTensor_free(state, output_var_n);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize3d(state, mu, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(state, var, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(state, input_mu, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize3d(state, input_var, nInputPlane, inputHeight, inputWidth);
  }

  // return output
  return 2;
}

static const struct luaL_Reg stcunn_StochasticSpatialConvolution__ [] = {
  {"StochasticSpatialConvolution_updateOutput", stcunn_StochasticSpatialConvolution_updateOutput},
  {NULL, NULL}
};

static void stcunn_StochasticSpatialConvolution_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, stcunn_StochasticSpatialConvolution__, "nn");
  lua_pop(L,1);
}
