#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/StochasticSpatialBatchNormalization.c"
#else

static int stnn_(StochasticSpatialBatchNormalization_updateOutput)(lua_State *L)
{
  THTensor *imu = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *ivar = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *omu = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *ovar = luaT_checkudata(L, 4, torch_Tensor);
  THTensor *weight = luaT_toudata(L, 5, torch_Tensor);
  THTensor *bias = luaT_toudata(L, 6, torch_Tensor);
  double eps = lua_tonumber(L, 7);
  THTensor *running_mean = luaT_checkudata(L, 8, torch_Tensor);
  THTensor *running_var = luaT_checkudata(L, 9, torch_Tensor);

  int batch = 1;
  if (imu->nDimension == 3) {
    batch = 0;
    THTensor_(resize4d)(imu, 1, imu->size[0], imu->size[1], imu->size[2]);
    THTensor_(resize4d)(ivar, 1, ivar->size[0], ivar->size[1], ivar->size[2]);
    THTensor_(resize4d)(omu, 1, omu->size[0], omu->size[1], omu->size[2]);
    THTensor_(resize4d)(ovar, 1, ovar->size[0], ovar->size[1], ovar->size[2]);
  }

  long nFeature = THTensor_(size)(imu, 1);
  long f;

  #pragma parallel for
  for (f = 0; f < nFeature; ++f) {
    THTensor *in_mu = THTensor_(newSelect)(imu, 1, f);
    THTensor *out_mu = THTensor_(newSelect)(omu, 1, f);
    THTensor *in_var = THTensor_(newSelect)(ivar, 1, f);
    THTensor *out_var = THTensor_(newSelect)(ovar, 1, f);

    real mean = THTensor_(get1d)(running_mean, f);
    real invstd = 1 / sqrt(THTensor_(get1d)(running_var, f) + eps);

    // compute output
    real w = weight ? THTensor_(get1d)(weight, f) : 1;
    real b = bias ? THTensor_(get1d)(bias, f) : 0;

    TH_TENSOR_APPLY2(real, in_mu, real, out_mu,
      *out_mu_data = (real) (((*in_mu_data - mean) * invstd) * w + b););

    TH_TENSOR_APPLY2(real, in_var, real, out_var,
      *out_var_data = (real) ((*in_var_data) * invstd * invstd * w * w););

    THTensor_(free)(out_mu);
    THTensor_(free)(in_mu);
    THTensor_(free)(out_var);
    THTensor_(free)(in_var);
  }

  if (batch == 0) {
    THTensor_(resize3d)(imu, imu->size[1], imu->size[2], imu->size[3]);
    THTensor_(resize3d)(ivar, ivar->size[1], ivar->size[2], ivar->size[3]);
    THTensor_(resize3d)(omu, omu->size[1], omu->size[2], omu->size[3]);
    THTensor_(resize3d)(ovar, ovar->size[1], ovar->size[2], ovar->size[3]);
  }

  return 0;
}

static const struct luaL_Reg stnn_(StochasticSpatialBatchNormalization__) [] = {
  {"StochasticSpatialBatchNormalization_updateOutput", stnn_(StochasticSpatialBatchNormalization_updateOutput)},
  {NULL, NULL}
};

static void stnn_(StochasticSpatialBatchNormalization_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, stnn_(StochasticSpatialBatchNormalization__), "nn");
  lua_pop(L,1);
}

#endif
