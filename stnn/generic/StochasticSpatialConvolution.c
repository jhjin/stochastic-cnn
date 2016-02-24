#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/StochasticSpatialConvolution.c"
#else

static void stnn_(unfolded_copy)(THTensor *finput, THTensor *input,
                               int kW, int kH,
                               int dW, int dH,
                               int padding,
                               int nInputPlane,
                               int inputWidth, int inputHeight,
                               int outputWidth, int outputHeight)
{
  long k;
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane*kH*kW; k++) {
    int nip = k / (kH*kW);
    int rest = k % (kH*kW);
    int kh = rest / kW;
    int kw = rest % kW;
    int x,y,ix,iy;
    real *dst = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
    real *src = input_data + nip*(inputHeight*inputWidth);
    if (padding > 0) {
      int lpad,rpad;
      for(y = 0; y < outputHeight; y++) {
        iy = y*dH - padding + kh;
        if (iy < 0 || iy >= inputHeight) {
          memset(dst+y*outputWidth, 0, sizeof(real)*outputWidth);
        } else {
          if (dW==1){
             ix = 0 - padding + kw;
             lpad = fmaxf(0,padding-kw);
             rpad = fmaxf(0,padding-(kW-kw-1));
             if (outputWidth-rpad-lpad <= 0) {
                memset(dst+y*outputWidth, 0, sizeof(real)*outputWidth);
             } else {
                if (lpad > 0) memset(dst+y*outputWidth, 0, sizeof(real)*lpad);
                memcpy(dst+y*outputWidth+lpad, src+iy*inputWidth+ix+lpad, sizeof(real)*(outputWidth-rpad-lpad));
                if (rpad > 0) memset(dst+y*outputWidth + outputWidth - rpad, 0, sizeof(real)*rpad);
             }
          }
          else{
            for (x=0; x<outputWidth; x++){
               ix = x*dW - padding + kw;
               if (ix < 0 || ix >= inputWidth)
                 memset(dst+y*outputWidth+x, 0, sizeof(real)*1);
               else
                 memcpy(dst+y*outputWidth+x, src+iy*inputWidth+ix, sizeof(real)*(1));
            }
          }
        }
      }
    } else {
      for(y = 0; y < outputHeight; y++) {
        iy = y*dH + kh;
        ix = 0 + kw;
        if (dW == 1)
           memcpy(dst+y*outputWidth, src+iy*inputWidth+ix, sizeof(real)*outputWidth);
        else{
          for (x=0; x<outputWidth; x++)
             memcpy(dst+y*outputWidth+x, src+iy*inputWidth+ix+x*dW, sizeof(real)*(1));
         }
      }
    }
  }
}

static void stnn_(StochasticSpatialConvolution_updateOutput_frame)(THTensor *input_mu, THTensor *output_mu,
                                                         THTensor *input_var, THTensor *output_var,
                                                         THTensor *weight, THTensor *weight2,
                                                         THTensor *bias, THTensor *finput,
                                                         int kW, int kH, int dW, int dH, int padding,
                                                         long nInputPlane, long inputWidth, long inputHeight,
                                                         long nOutputPlane, long outputWidth, long outputHeight)
{
  long i;
  THTensor *output2d_mu, *output2d_var;


  stnn_(unfolded_copy)(finput, input_var, kW, kH, dW, dH, padding,
                     nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);

  output2d_var = THTensor_(newWithStorage2d)(output_var->storage, output_var->storageOffset,
                                             nOutputPlane, -1, outputHeight*outputWidth, -1);

  THTensor_(addmm)(output2d_var, 0, output2d_var, 1, weight2, finput);


  stnn_(unfolded_copy)(finput, input_mu, kW, kH, dW, dH, padding,
                     nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);

  output2d_mu = THTensor_(newWithStorage2d)(output_mu->storage, output_mu->storageOffset,
                                            nOutputPlane, -1, outputHeight*outputWidth, -1);

  for(i = 0; i < nOutputPlane; i++)
    THVector_(fill)(output_mu->storage->data+output_mu->storageOffset+output_mu->stride[0]*i,
                    THTensor_(get1d)(bias, i), outputHeight*outputWidth);

  THTensor_(addmm)(output2d_mu, 1, output2d_mu, 1, weight, finput);


  THTensor_(free)(output2d_mu);
  THTensor_(free)(output2d_var);
}

static int stnn_(StochasticSpatialConvolution_updateOutput)(lua_State *L)
{
  THTensor *input_mu = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *input_var = luaT_checkudata(L, 3, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THTensor *finput = luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *weight2 = luaT_getfieldcheckudata(L, 1, "weight2", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *mu = luaT_getfieldcheckudata(L, 1, "mu", torch_Tensor);
  THTensor *var = luaT_getfieldcheckudata(L, 1, "var", torch_Tensor);

  int dimf = 0;
  int dimw = 2;
  int dimh = 1;

  long nInputPlane;
  long inputWidth;
  long inputHeight;
  long nOutputPlane;
  long outputWidth;
  long outputHeight;

  luaL_argcheck(L, input_mu->nDimension == 3 || input_mu->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");


  if (input_mu->nDimension == 4) {
    dimf++;
    dimw++;
    dimh++;
  }

  nInputPlane = input_mu->size[dimf];
  inputWidth   = input_mu->size[dimw];
  inputHeight  = input_mu->size[dimh];
  nOutputPlane = weight->size[0];
  outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  outputHeight = (inputHeight + 2*padding - kH) / dH + 1;

  if(input_mu->nDimension == 3)
  {
    THTensor_(resize2d)(finput, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize3d)(mu, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(var, nOutputPlane, outputHeight, outputWidth);

    stnn_(StochasticSpatialConvolution_updateOutput_frame)(input_mu, mu,
                                                 input_var, var,
                                                 weight, weight2, bias, finput,
                                                 kW, kH, dW, dH, padding,
                                                 nInputPlane, inputWidth, inputHeight,
                                                 nOutputPlane, outputWidth, outputHeight);
  }
  else
  {
    long T = input_mu->size[0];
    long t;

    THTensor_(resize3d)(finput, T, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize4d)(mu, T, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize4d)(var, T, nOutputPlane, outputHeight, outputWidth);

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *input_mu_t = THTensor_(newSelect)(input_mu, 0, t);
      THTensor *input_var_t = THTensor_(newSelect)(input_var, 0, t);
      THTensor *output_mu_t = THTensor_(newSelect)(mu, 0, t);
      THTensor *output_var_t = THTensor_(newSelect)(var, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      stnn_(StochasticSpatialConvolution_updateOutput_frame)(input_mu_t, output_mu_t,
                                                   input_var_t, output_var_t,
                                                   weight, weight2, bias, finput_t,
                                                   kW, kH, dW, dH, padding,
                                                   nInputPlane, inputWidth, inputHeight,
                                                   nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(input_mu_t);
      THTensor_(free)(input_var_t);
      THTensor_(free)(output_mu_t);
      THTensor_(free)(output_var_t);
      THTensor_(free)(finput_t);
    }
  }

  return 2;
}

static const struct luaL_Reg stnn_(StochasticSpatialConvolution__) [] = {
  {"StochasticSpatialConvolution_updateOutput", stnn_(StochasticSpatialConvolution_updateOutput)},
  {NULL, NULL}
};

static void stnn_(StochasticSpatialConvolution_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, stnn_(StochasticSpatialConvolution__), "nn");
  lua_pop(L,1);
}

#endif
