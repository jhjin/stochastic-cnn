#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define stnn_(NAME) TH_CONCAT_3(stnn_, Real, NAME)

#include "generic/StochasticSpatialConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/StochasticSpatialBatchNormalization.c"
#include "THGenerateFloatTypes.h"

#include "generic/StochasticSpatialMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/StochasticSpatialAveragePooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/StochasticThreshold.c"
#include "THGenerateFloatTypes.h"

#include "generic/StochasticSpatialSampling.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libstnn(lua_State *L);

int luaopen_libstnn(lua_State *L)
{
  lua_newtable(L);

  stnn_FloatStochasticSpatialConvolution_init(L);
  stnn_FloatStochasticSpatialBatchNormalization_init(L);
  stnn_FloatStochasticSpatialMaxPooling_init(L);
  stnn_FloatStochasticSpatialAveragePooling_init(L);
  stnn_FloatStochasticThreshold_init(L);
  stnn_FloatStochasticSpatialSampling_init(L);

  stnn_DoubleStochasticSpatialConvolution_init(L);
  stnn_DoubleStochasticSpatialBatchNormalization_init(L);
  stnn_DoubleStochasticSpatialMaxPooling_init(L);
  stnn_DoubleStochasticSpatialAveragePooling_init(L);
  stnn_DoubleStochasticThreshold_init(L);
  stnn_DoubleStochasticSpatialSampling_init(L);

  return 1;
}
