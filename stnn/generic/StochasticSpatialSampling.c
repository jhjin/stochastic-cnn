#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/StochasticSpatialSampling.c"
#else

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)

static void stnn_(StochasticSpatialSampling_updateOutput_frame)(real *input_mu, real *output_mu,
                                                          real *input_var, real *output_var,
                                                          real *randi, real radius, long iH, long iW)
{
  int number, col, row;
  int px_lin = (2*radius + 1);
  int px_box = px_lin*px_lin;

  long i, j;
  for (i = 0; i < iH; i++) {
    for (j = 0; j < iW; j++) {
      number = randi[i*iW+j];
      col = MAX( MIN( (number % px_lin)-radius+j, iW-1), 0);
      row = MAX( MIN( (number / px_lin)-radius+i, iH-1), 0);;

      output_mu[i*iW+j] = input_mu[row*iW+col];
      output_var[i*iW+j] = input_var[row*iW+col];
    }
  }
}

static int stnn_(StochasticSpatialSampling_updateOutput)(lua_State *L)
{
  THTensor *input_mu = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *input_var = luaT_checkudata(L, 3, torch_Tensor);
  real radius = luaT_getfieldchecknumber(L, 1, "radius");
  THTensor *randi = luaT_getfieldcheckudata(L, 1, "randi", torch_Tensor);
  THTensor *mu = luaT_getfieldcheckudata(L, 1, "mu", torch_Tensor);
  THTensor *var = luaT_getfieldcheckudata(L, 1, "var", torch_Tensor);

  int batch = 0;
  long batchSize, nInputPlane, nOutputPlane, inputWidth, inputHeight;

  if (input_mu->nDimension == 4) {
    batch = 1;
    batchSize    = input_mu->size[0];
    nInputPlane  = input_mu->size[1];
    inputHeight  = input_mu->size[2];
    inputWidth   = input_mu->size[3];

    THTensor_(resize3d)(input_mu, batchSize*nInputPlane, inputHeight, inputWidth);
    THTensor_(resize3d)(input_var, batchSize*nInputPlane, inputHeight, inputWidth);
    THTensor_(resize3d)(randi, batchSize*nInputPlane, inputHeight, inputWidth);

  } else if (input_mu->nDimension == 3) {
    batchSize    = 1;
    nInputPlane  = input_mu->size[0];
    inputHeight  = input_mu->size[1];
    inputWidth   = input_mu->size[2];
  }

  input_mu = THTensor_(newContiguous)(input_mu);
  input_var = THTensor_(newContiguous)(input_var);
  THTensor_(resize3d)(mu, batchSize*nInputPlane, inputHeight, inputWidth);
  THTensor_(resize3d)(var, batchSize*nInputPlane, inputHeight, inputWidth);

  THTensor *input_mu_n   = THTensor_(new)();
  THTensor *input_var_n  = THTensor_(new)();
  THTensor *output_mu_n  = THTensor_(new)();
  THTensor *output_var_n = THTensor_(new)();
  THTensor *randi_n      = THTensor_(new)();

  int elt;
  for (elt = 0; elt < batchSize*nInputPlane; elt ++) {

    THTensor_(select)(input_mu_n, input_mu, 0, elt);
    THTensor_(select)(input_var_n, input_var, 0, elt);
    THTensor_(select)(output_mu_n, mu, 0, elt);
    THTensor_(select)(output_var_n, var, 0, elt);
    THTensor_(select)(randi_n, randi, 0, elt);

    stnn_(StochasticSpatialSampling_updateOutput_frame)(THTensor_(data)(input_mu_n),
                                                  THTensor_(data)(output_mu_n),
                                                  THTensor_(data)(input_var_n),
                                                  THTensor_(data)(output_var_n),
                                                  THTensor_(data)(randi_n),
                                                  radius, inputHeight, inputWidth);
  }

  THTensor_(free)(input_mu_n);
  THTensor_(free)(input_var_n);
  THTensor_(free)(output_mu_n);
  THTensor_(free)(output_var_n);
  THTensor_(free)(randi_n);

  if (batch == 1) {
    THTensor_(resize4d)(mu, batchSize, nInputPlane, inputHeight, inputWidth);
    THTensor_(resize4d)(var, batchSize, nInputPlane, inputHeight, inputWidth);
    THTensor_(resize4d)(input_mu, batchSize, nInputPlane, inputHeight, inputWidth);
    THTensor_(resize4d)(input_var, batchSize, nInputPlane, inputHeight, inputWidth);
    THTensor_(resize4d)(randi, batchSize, nInputPlane, inputHeight, inputWidth);
  }

  return 2;
}

static const struct luaL_Reg stnn_(StochasticSpatialSampling__) [] = {
  {"StochasticSpatialSampling_updateOutput", stnn_(StochasticSpatialSampling_updateOutput)},
  {NULL, NULL}
};

static void stnn_(StochasticSpatialSampling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, stnn_(StochasticSpatialSampling__), "nn");
  lua_pop(L,1);
}

#endif
