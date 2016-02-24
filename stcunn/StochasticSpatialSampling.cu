#include "utils.h"

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)

__global__ void sampling_prob(const int n,
                              const float *input_mu, float *output_mu,
                              const float *input_var, float *output_var,
                              const float *randi,
                              const int radius, const int iH, const int iW)
{
   for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
      int px_lin = (2*radius + 1);
      int c = i / (iH*iW);
      int h = (i % (iH*iW)) / iW;
      int w = i % iW;

      int number = randi[i];
      int col = MAX( MIN( (number % px_lin) - radius + w, iW-1), 0);
      int row = MAX( MIN( (number / px_lin) - radius + h, iH-1), 0);;

      output_mu[i] = input_mu[c*(iH*iW) + row*iW + col];
      output_var[i] = input_var[c*(iH*iW) + row*iW + col];
   }
}

static int stcunn_StochasticSpatialSampling_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input_mu = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *input_var = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  int radius = luaT_getfieldcheckint(L, 1, "radius");
  THCudaTensor *randi = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "randi", "torch.CudaTensor");
  THCudaTensor *mu = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "mu", "torch.CudaTensor");
  THCudaTensor *var = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "var", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, input_mu, input_var, mu, var));
  luaL_argcheck(L, input_mu->nDimension == 3 || input_mu->nDimension == 4, 2, "3D or 4D (batch) tensor expected");


  long inputHeight, inputWidth;
  if (input_mu->nDimension == 4) {
    inputHeight  = input_mu->size[2];
    inputWidth   = input_mu->size[3];
  } else if (input_mu->nDimension == 3) {
    inputHeight  = input_mu->size[1];
    inputWidth   = input_mu->size[2];
  }

  input_mu = THCudaTensor_newContiguous(state, input_mu);
  input_var = THCudaTensor_newContiguous(state, input_var);
  THCudaTensor_resizeAs(state, mu, input_mu);
  THCudaTensor_resizeAs(state, var, input_var);

  int num_kernels = THCudaTensor_nElement(state, input_mu);
  sampling_prob <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>> (
    num_kernels,
    THCudaTensor_data(state, input_mu),
    THCudaTensor_data(state, mu),
    THCudaTensor_data(state, input_var),
    THCudaTensor_data(state, var),
    THCudaTensor_data(state, randi),
    radius, inputHeight, inputWidth
  );

  // clean
  THCudaTensor_free(state, input_mu);
  THCudaTensor_free(state, input_var);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in StochasticSpatialSampling.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 2;
}

static const struct luaL_Reg stcunn_StochasticSpatialSampling__ [] = {
  {"StochasticSpatialSampling_updateOutput", stcunn_StochasticSpatialSampling_updateOutput},
  {NULL, NULL}
};

static void stcunn_StochasticSpatialSampling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, stcunn_StochasticSpatialSampling__, "nn");
  lua_pop(L,1);
}
