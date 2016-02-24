#include "utils.h"

#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"

typedef THCDeviceTensor<float, 4> DeviceTensor4;
typedef THCDeviceTensor<float, 1> DeviceTensor1;

template <int Dim>
static THCDeviceTensor<float, Dim> checktensor(lua_State* L, int index) {
  THCudaTensor *t = (THCudaTensor*)luaT_toudata(L, index, "torch.CudaTensor");
  if (!t) {
    return THCDeviceTensor<float, Dim>();
  }
  return toDeviceTensor<float, Dim>(getCutorchState(L), t);
}

__global__ void StochasticSpatialBatchNormalizationUpdateOutputInference_kernel(
    const DeviceTensor4 imu,
    const DeviceTensor4 ivar,
    DeviceTensor4 omu,
    DeviceTensor4 ovar,
    DeviceTensor1 runningMean,
    DeviceTensor1 runningVar,
    const DeviceTensor1 weight,
    const DeviceTensor1 bias,
    float epsilon) {

  int x = threadIdx.x;
  int plane = blockIdx.x;
  int batch = blockIdx.y;

  float invstd = 1.0f / sqrt(runningVar[plane].ldg() + epsilon);
  float mean = runningMean[plane].ldg();
  float gamma = weight.numElements() > 0 ? weight[plane].ldg() : 1.0f;
  float beta = bias.numElements() > 0 ? bias[plane].ldg() : 0.0f;

  for (int y = threadIdx.y; y < omu.getSize(2); y += blockDim.y) {
    float in_mu = imu[batch][plane][y][x].ldg();
    float in_var = ivar[batch][plane][y][x].ldg();
    omu[batch][plane][y][x] = gamma * (in_mu - mean) * invstd + beta;
    ovar[batch][plane][y][x] = gamma * gamma * in_var * invstd * invstd;
  }
}

static int stcunn_StochasticSpatialBatchNormalization_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);

  DeviceTensor4 imu = checktensor<4>(L, 1);
  DeviceTensor4 ivar = checktensor<4>(L, 2);
  DeviceTensor4 omu = checktensor<4>(L, 3);
  DeviceTensor4 ovar = checktensor<4>(L, 4);
  DeviceTensor1 weight = checktensor<1>(L, 5);
  DeviceTensor1 bias = checktensor<1>(L, 6);
  double eps = lua_tonumber(L, 7);
  DeviceTensor1 runningMean = checktensor<1>(L, 8);
  DeviceTensor1 runningVar = checktensor<1>(L, 9);

  cudaStream_t s = THCState_getCurrentStream(state);
  cudaDeviceProp *prop = THCState_getCurrentDeviceProperties(state);
  int maxThreadsPerBlock = prop->maxThreadsPerBlock;

  dim3 blocks(imu.getSize(1), imu.getSize(0));
  dim3 threads(imu.getSize(3), min(imu.getSize(2), maxThreadsPerBlock / imu.getSize(3)));

  StochasticSpatialBatchNormalizationUpdateOutputInference_kernel
    <<<blocks, threads, 0, s>>>
    (imu, ivar, omu, ovar, runningMean, runningVar, weight, bias, eps);

  return 0;
}

static const struct luaL_Reg stcunn_StochasticSpatialBatchNormalization__ [] = {
  {"StochasticSpatialBatchNormalization_updateOutput", stcunn_StochasticSpatialBatchNormalization_updateOutput},
  {NULL, NULL}
};

void stcunn_StochasticSpatialBatchNormalization_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, stcunn_StochasticSpatialBatchNormalization__, "nn");
  lua_pop(L,1);
}
