#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/StochasticThreshold.c"
#else

#include <math.h>

static void stnn_(StochasticThreshold_updateOutput_frame)(real *input_mu, real *output_mu,
                                                  real *input_var, real *output_var,
                                                  real val, long length)
{
  long i;
  for (i = 0; i < length; i++) {
    real std = sqrt(input_var[i]);
    real alpha = (val - input_mu[i])/std;
    real pdf = exp(-1.*alpha*alpha/2.)/sqrt(2.*M_PI);
    real cdf = 0.5*(1+erf( alpha/sqrt(2.)));
    real cdf_c = fmax(1.-cdf, 1e-20);
    real lambda = pdf/cdf_c;
    real delta = lambda*(lambda - alpha);

    output_mu[i] = val*cdf + (input_mu[i] + std*lambda)*cdf_c;
    output_var[i] = input_var[i]*cdf_c*(1-delta + (alpha-lambda)*(alpha-lambda)*cdf);
  }
}

static int stnn_(StochasticThreshold_updateOutput)(lua_State *L)
{
  THTensor *input_mu = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *input_var = luaT_checkudata(L, 3, torch_Tensor);
  real val = luaT_getfieldchecknumber(L, 1, "val");
  real threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  int inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");
  THTensor *mu = luaT_getfieldcheckudata(L, 1, "mu", torch_Tensor);
  THTensor *var = luaT_getfieldcheckudata(L, 1, "var", torch_Tensor);

  input_mu = THTensor_(newContiguous)(input_mu);
  input_var = THTensor_(newContiguous)(input_var);

  THTensor_(clamp)(input_var, input_var, 1e-20, (real) FLT_MAX);

  if (inPlace) {
    // TBD
    THTensor_(set)(mu, input_mu);
    THTensor_(set)(var, input_var);
  } else {
    THTensor_(resizeAs)(mu, input_mu);
    THTensor_(resizeAs)(var, input_var);

    // #pragma omp parallel for private(p)
    stnn_(StochasticThreshold_updateOutput_frame)(THTensor_(data)(input_mu), THTensor_(data)(mu),
                                          THTensor_(data)(input_var), THTensor_(data)(var),
                                          val, THTensor_(nElement)(input_mu));

  }

  return 2;
}

static const struct luaL_Reg stnn_(StochasticThreshold__) [] = {
  {"StochasticThreshold_updateOutput", stnn_(StochasticThreshold_updateOutput)},
  {NULL, NULL}
};

static void stnn_(StochasticThreshold_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, stnn_(StochasticThreshold__), "nn");
  lua_pop(L,1);
}

#endif
