#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "utils.cu"

#include "StochasticSpatialConvolution.cu"
#include "StochasticSpatialBatchNormalization.cu"
#include "StochasticSpatialMaxPooling.cu"
#include "StochasticSpatialAveragePooling.cu"
#include "StochasticThreshold.cu"
#include "StochasticSpatialSampling.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libstcunn(lua_State *L);

int luaopen_libstcunn(lua_State *L)
{
  lua_newtable(L);

  stcunn_StochasticSpatialConvolution_init(L);
  stcunn_StochasticSpatialBatchNormalization_init(L);
  stcunn_StochasticSpatialMaxPooling_init(L);
  stcunn_StochasticSpatialAveragePooling_init(L);
  stcunn_StochasticThreshold_init(L);
  stcunn_StochasticSpatialSampling_init(L);

  return 1;
}
