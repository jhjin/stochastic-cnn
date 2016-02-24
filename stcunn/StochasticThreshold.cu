#include "utils.h"

__global__ void threshold_prob(const int n, const float val,
                               float *input_mu, float *output_mu,
                               float *input_var, float *output_var)
{
   for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
     float input_var_clamp = fmaxf(input_var[i], 1e-20);
     float std = sqrtf(input_var_clamp);
     float alpha = fdividef(val - input_mu[i], std);
     float pdf = fdividef(expf(fdividef(-1.*alpha*alpha, 2.)), sqrtf(2.*M_PI));
     float cdf = normcdff(alpha); // 0.5*(1.+erf( alpha/sqrt(2.)));
     float cdf_c = fmaxf(1.-cdf, 1e-20);
     float lambda = fdividef(pdf, cdf_c);
     float delta = lambda*(lambda - alpha);

     output_mu[i] = val*cdf + (input_mu[i] + std*lambda)*cdf_c;
     output_var[i] = input_var_clamp*cdf_c*(1-delta + (alpha-lambda)*(alpha-lambda)*cdf);
   }
}

__global__ void threshold_prob_inplace(const int n, const float val,
                                       float *output_mu, float *output_var)
{
   for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
     float input_var_clamp = fmaxf(output_var[i], 1e-20);
     float std = sqrtf(input_var_clamp);
     float alpha = fdividef(val - output_mu[i], std);
     float pdf = fdividef(expf(fdivide(-1.*alpha*alpha, 2.)), sqrtf(2.*M_PI));
     float cdf = normcdff(alpha); // 0.5*(1.+erf( alpha/sqrt(2.)));
     float cdf_c = fmaxf(1.-cdf, 1e-20);
     float lambda = fdividef(pdf, cdf_c);
     float delta = lambda*(lambda - alpha);

     output_mu[i] = val*cdf + (output_mu[i] + std*lambda)*cdf_c;
     output_var[i] = input_var_clamp*cdf_c*(1-delta + (alpha-lambda)*(alpha-lambda)*cdf);
   }
}

static int stcunn_StochasticThreshold_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input_mu = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *input_var = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  double val = luaT_getfieldchecknumber(L, 1, "val");
  double threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  bool   inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");
  THCudaTensor *mu = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "mu", "torch.CudaTensor");
  THCudaTensor *var = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "var", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, input_mu, input_var, mu, var));

  long num_threads_ = THCudaTensor_nElement(state, input_mu);

  if (inPlace) {
    THCudaTensor_set(state, mu, input_mu);
    THCudaTensor_set(state, var, input_var);

    threshold_prob_inplace <<<GET_BLOCKS(num_threads_), CUDA_NUM_THREADS>>> (
      num_threads_, val,
      THCudaTensor_data(state, mu),
      THCudaTensor_data(state, var)
    );
  } else {
    THCudaTensor_resizeAs(state, mu, input_mu);
    THCudaTensor_resizeAs(state, var, input_var);

    threshold_prob <<<GET_BLOCKS(num_threads_), CUDA_NUM_THREADS>>> (
      num_threads_, val,
      THCudaTensor_data(state, input_mu),
      THCudaTensor_data(state, mu),
      THCudaTensor_data(state, input_var),
      THCudaTensor_data(state, var)
    );
  }

  THCudaCheck(cudaGetLastError());
  return 2;
}

static const struct luaL_Reg stcunn_StochasticThreshold__ [] = {
  {"StochasticThreshold_updateOutput", stcunn_StochasticThreshold_updateOutput},
  {NULL, NULL}
};

static void stcunn_StochasticThreshold_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, stcunn_StochasticThreshold__, "nn");
  lua_pop(L,1);
}
