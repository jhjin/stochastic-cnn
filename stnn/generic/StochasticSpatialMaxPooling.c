#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/StochasticSpatialMaxPooling.c"
#else

#include <math.h>

static void stnn_(quick_sort)(real *a, real *b, int n) {
  int i, j;
  real p, t, s;
  if (n < 2)
    return;
  p = a[(int)(n / 2)];
  for (i = 0, j = n - 1;; i++, j--) {
    while (a[i] < p)
      i++;
    while (p < a[j])
      j--;
    if (i >= j)
      break;
    t = a[i];
    a[i] = a[j];
    a[j] = t;
    s = b[i];
    b[i] = b[j];
    b[j] = s;
  }
  stnn_(quick_sort)(a, b, i);
  stnn_(quick_sort)(a + i, b + i, n - i);
}

static void stnn_(StochasticSpatialMaxPooling_updateOutput_frame)(real *input_mu, real *output_mu,
                                                                  real *input_var, real *output_var,
                                                                  long nslices,
                                                                  long iwidth, long iheight,
                                                                  long owidth, long oheight,
                                                                  int kW, int kH, int dW, int dH,
                                                                  int padW, int padH, int sort)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    real *mu_sorted = (real *) malloc(kW*kH*sizeof(real));
    real *var_sorted = (real *) malloc(kW*kH*sizeof(real));

    /* loop over output */
    long i, j;
    real *imu = input_mu  + k*iwidth*iheight;
    real *ivar = input_var + k*iwidth*iheight;
    for(i = 0; i < oheight; i++)
    {
      for(j = 0; j < owidth; j++)
      {
        long hstart = i * dH - padH;
        long wstart = j * dW - padW;
        long hend = fminf(hstart + kH, iheight);
        long wend = fminf(wstart + kW, iwidth);
        hstart = fmaxf(hstart, 0);
        wstart = fmaxf(wstart, 0);

        long x, y, cnt = 0;
        for(y = hstart; y < hend; y++)
        {
          for(x = wstart; x < wend; x++)
          {
            mu_sorted[cnt] = *(imu + y*iwidth + x);
            var_sorted[cnt] = *(ivar + y*iwidth + x);
            cnt++;
          }
        }

        /* ascending order */
        if (sort)
          stnn_(quick_sort)(mu_sorted, var_sorted, cnt);

        /* local pointers */
        real *omu  = output_mu + k*owidth*oheight + i*owidth + j;
        real *ovar = output_var + k*owidth*oheight + i*owidth + j;

        /* compute local max: */
        real mu1 = mu_sorted[0];
        real var1 = var_sorted[0];
        for(x = 1; x < cnt; x++)
        {
          real mu2 = mu_sorted[x];
          real var2 = var_sorted[x];

          real theta = sqrt(var1 + var2);
          real alpha12 = (mu1 - mu2)/theta;

          real cdf12 = 0.5*(1+erf( alpha12/sqrt(2.)));
          real cdf21 = 0.5*(1+erf(-alpha12/sqrt(2.)));
          real pdf12 = exp(-1.*alpha12*alpha12/2.)/sqrt(2.*M_PI);

          real t_mu = mu1*cdf12 + mu2*cdf21 + theta*pdf12;
          real t_var = (var1+mu1*mu1)*cdf12 + (var2+mu2*mu2)*cdf21 + (mu1+mu2)*theta*pdf12 - t_mu*t_mu;

          mu1 = t_mu;
          var1 = t_var;
        }
        *omu = mu1;
        *ovar = var1;
      }
    }
  }
}

static int stnn_(StochasticSpatialMaxPooling_updateOutput)(lua_State *L)
{
  THTensor *input_mu = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *input_var = luaT_checkudata(L, 3, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  int sort = luaT_getfieldcheckboolean(L, 1, "sort");
  int ceil_mode = luaT_getfieldcheckboolean(L, 1, "ceil_mode");
  THTensor *mu = luaT_getfieldcheckudata(L, 1, "mu", torch_Tensor);
  THTensor *var = luaT_getfieldcheckudata(L, 1, "var", torch_Tensor);
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  long nslices;
  long iheight;
  long iwidth;
  long oheight;
  long owidth;
  real *input_mu_data;
  real *input_var_data;
  real *output_mu_data;
  real *output_var_data;


  luaL_argcheck(L, input_mu->nDimension == 3 || input_mu->nDimension == 4 , 2, "3D or 4D (batch mode) tensor expected");

  if (input_mu->nDimension == 4)
  {
    nbatch = input_mu->size[0];
    dimw++;
    dimh++;
  }
  luaL_argcheck(L, input_mu->size[dimw] >= kW && input_mu->size[dimh] >= kH, 2, "input image smaller than kernel size");

  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");

  /* sizes */
  nslices = input_mu->size[dimh-1];
  iheight = input_mu->size[dimh];
  iwidth = input_mu->size[dimw];
  if (ceil_mode)
  {
    oheight = (long)(ceil((float)(iheight - kH + 2*padH) / dH)) + 1;
    owidth  = (long)(ceil((float)(iwidth  - kW + 2*padW) / dW)) + 1;
  }
  else
  {
    oheight = (long)(floor((float)(iheight - kH + 2*padH) / dH)) + 1;
    owidth  = (long)(floor((float)(iwidth  - kW + 2*padW) / dW)) + 1;
  }

  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    if ((oheight - 1)*dH >= iheight + padH)
      --oheight;
    if ((owidth  - 1)*dW >= iwidth  + padW)
      --owidth;
  }

  /* get contiguous input */
  input_mu = THTensor_(newContiguous)(input_mu);
  input_var = THTensor_(newContiguous)(input_var);

  /* avoid zero division */
  THTensor_(clamp)(input_var, input_var, 1e-20, (real) FLT_MAX);

  /* resize output */
  if (input_mu->nDimension == 3)
  {
    THTensor_(resize3d)(mu, nslices, oheight, owidth);
    THTensor_(resize3d)(var, nslices, oheight, owidth);

    input_mu_data = THTensor_(data)(input_mu);
    input_var_data = THTensor_(data)(input_var);
    output_mu_data = THTensor_(data)(mu);
    output_var_data = THTensor_(data)(var);

    stnn_(StochasticSpatialMaxPooling_updateOutput_frame)(input_mu_data, output_mu_data,
                                                          input_var_data, output_var_data,
                                                          nslices,
                                                          iwidth, iheight, owidth, oheight,
                                                          kW, kH, dW, dH,
                                                          padW, padH, sort);
  }
  else
  {
    long p;

    THTensor_(resize4d)(mu, nbatch, nslices, oheight, owidth);
    THTensor_(resize4d)(var, nbatch, nslices, oheight, owidth);

    input_mu_data = THTensor_(data)(input_mu);
    input_var_data = THTensor_(data)(input_var);
    output_mu_data = THTensor_(data)(mu);
    output_var_data = THTensor_(data)(var);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      stnn_(StochasticSpatialMaxPooling_updateOutput_frame)(input_mu_data+p*nslices*iwidth*iheight,
                                                            output_mu_data+p*nslices*owidth*oheight,
                                                            input_var_data+p*nslices*iwidth*iheight,
                                                            output_var_data+p*nslices*owidth*oheight,
                                                            nslices,
                                                            iwidth, iheight,
                                                            owidth, oheight,
                                                            kW, kH, dW, dH,
                                                            padW, padH, sort);
    }
  }

  /* cleanup */
  THTensor_(free)(input_mu);
  THTensor_(free)(input_var);
  return 2;
}

static const struct luaL_Reg stnn_(StochasticSpatialMaxPooling__) [] = {
  {"StochasticSpatialMaxPooling_updateOutput", stnn_(StochasticSpatialMaxPooling_updateOutput)},
  {NULL, NULL}
};

static void stnn_(StochasticSpatialMaxPooling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, stnn_(StochasticSpatialMaxPooling__), "nn");
  lua_pop(L,1);
}

#endif
