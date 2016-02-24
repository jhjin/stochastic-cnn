#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/StochasticSpatialAveragePooling.c"
#else

static int stnn_(StochasticSpatialAveragePooling_updateOutput)(lua_State *L)
{
  THTensor *input_mu = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *input_var = luaT_checkudata(L, 3, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  int ceil_mode = luaT_getfieldcheckboolean(L, 1, "ceil_mode");
  int count_include_pad = luaT_getfieldcheckboolean(L, 1, "count_include_pad");
  THTensor *output_mu = luaT_getfieldcheckudata(L, 1, "mu", torch_Tensor);
  THTensor *output_var = luaT_getfieldcheckudata(L, 1, "var", torch_Tensor);
  real *input_mu_data;
  real *input_var_data;
  real *output_mu_data;
  real *output_var_data;

  int dimw = 2;
  int dimh = 1;
  int dimc = 0;
  long nbatch = 1;

  long inputWidth;
  long inputHeight;
  long outputWidth;
  long outputHeight;
  long nInputPlane; // number of channels (or colors)

  long k;

  THArgCheck(input_mu->nDimension == 3 || input_mu->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");
  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");

  if (input_mu->nDimension == 4) {
    nbatch = input_mu->size[0];
    dimw++;
    dimh++;
    dimc++;
  }

  inputWidth = input_mu->size[dimw];
  inputHeight = input_mu->size[dimh];
  nInputPlane = input_mu->size[dimc];

  if(ceil_mode)
  {
    outputWidth  = (long)(ceil((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
    outputHeight = (long)(ceil((float)(inputHeight - kH + 2*padH) / dH)) + 1;
  }
  else
  {
    outputWidth  = (long)(floor((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
    outputHeight = (long)(floor((float)(inputHeight - kH + 2*padH) / dH)) + 1;
  }
  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((outputHeight - 1)*dH >= inputHeight + padH)
      --outputHeight;
    if ((outputWidth  - 1)*dW >= inputWidth  + padW)
      --outputWidth;
  }

  THArgCheck(inputWidth >= kW - 2 * padW && inputHeight >= kH - 2 * padH, 2, "input image smaller than kernel size");

  if (input_mu->nDimension == 3) {
    THTensor_(resize3d)(output_mu, nInputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(output_var, nInputPlane, outputHeight, outputWidth);
  } else {
    THTensor_(resize4d)(output_mu, input_mu->size[0], nInputPlane, outputHeight, outputWidth);
    THTensor_(resize4d)(output_var, input_mu->size[0], nInputPlane, outputHeight, outputWidth);
  }
  
  input_mu = THTensor_(newContiguous)(input_mu);
  input_var = THTensor_(newContiguous)(input_var);
  THArgCheck(THTensor_(isContiguous)(output_mu), 3, "output must be contiguous");
  THArgCheck(THTensor_(isContiguous)(output_var), 3, "output must be contiguous");
  input_mu_data = THTensor_(data)(input_mu);
  input_var_data = THTensor_(data)(input_var);
  output_mu_data = THTensor_(data)(output_mu);
  output_var_data = THTensor_(data)(output_var);
  
#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    long p;
    for(p = 0; p < nbatch; p++)
    {
      long xx, yy;
      /* For all output pixels... */
      real *omu = output_mu_data + p*nInputPlane*outputWidth*outputHeight + k*outputWidth*outputHeight;
      real *ovar = output_var_data + p*nInputPlane*outputWidth*outputHeight + k*outputWidth*outputHeight;
      real *imu = input_mu_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;
      real *ivar = input_var_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;
      long i;
      for(i = 0; i < outputWidth*outputHeight; i++) {
        omu[i] = 0;
        ovar[i] = 0;
      }
      
      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          /* Compute the mean of the input image... */
          long hstart = yy * dH - padH;
          long wstart = xx * dW - padW;
          long hend = fminf(hstart + kH, inputHeight + padH);
          long wend = fminf(wstart + kW, inputWidth + padW);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = fmaxf(hstart, 0);
          wstart = fmaxf(wstart, 0);
          hend = fminf(hend, inputHeight);
          wend = fminf(wend, inputWidth);

          real sum_mu = 0;
          real sum_var = 0;

          int divide_factor;
          if(count_include_pad)
            divide_factor = pool_size;
          else
            divide_factor = (hend - hstart) * (wend - wstart);

          long kx, ky;

          for(ky = hstart; ky < hend; ky++)
          {
            for(kx = wstart; kx < wend; kx++)
            {
              sum_mu += imu[ky*inputWidth + kx];
              sum_var += ivar[ky*inputWidth + kx];
            }
          }
          /* Update output */
          *omu++ += sum_mu/divide_factor;
          *ovar++ += sum_var/(divide_factor*divide_factor);
        }
      }
    }
  }
  THTensor_(free)(input_mu);
  THTensor_(free)(input_var);
  return 2;
}

static const struct luaL_Reg stnn_(StochasticSpatialAveragePooling__) [] = {
  {"StochasticSpatialAveragePooling_updateOutput", stnn_(StochasticSpatialAveragePooling_updateOutput)},
  {NULL, NULL}
};

static void stnn_(StochasticSpatialAveragePooling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, stnn_(StochasticSpatialAveragePooling__), "nn");
  lua_pop(L,1);
}


#endif
