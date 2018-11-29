#include <THC/THC.h>
#include <stdio.h>

#include "sample_weighting_kernel.h"

extern THCState *state;

extern "C" {

int sample_weighting_forward(
    THCudaTensor *data,
    THCudaTensor *coords,
    THCudaTensor *params,
    THCudaTensor *kernels,
    THCudaTensor *output,
    THCudaTensor *output_w,
    const long nsize) {

  /* TODO: checks */
  THArgCheck(THCudaTensor_nDimension(state, data) == 5, 0,  "data should be 5D");
  THArgCheck(THCudaTensor_nDimension(state, coords) == 5, 1,  "coords should be 5D");
  THArgCheck(THCudaTensor_nDimension(state, params) == 4, 2, "params should be 4D");
  THArgCheck(THCudaTensor_nDimension(state, kernels) == 4, 3, "kernels should be 4D");
  THArgCheck(nsize % 2 == 1, 4, "nsize should be odd");
  long kernel_ci = THCudaTensor_size(state, kernels, 0);
  // long kernel_co = THCudaTensor_size(state, kernels, 1);
  long data_ci = THCudaTensor_size(state, data, 1);
  long ncoords = THCudaTensor_size(state, coords, 1);
  long nparams = THCudaTensor_size(state, params, 1);
  THArgCheck(ncoords >= 2, 1, "should at least have 2 coordinates (x, y)");
  THArgCheck(kernel_ci == data_ci, 3,
      "kernels input channels (%d) and data (%d) should match",
      kernel_ci, data_ci);
  THArgCheck(
      2*(ncoords) == nparams, 3,
      "nparams (%d) != (ncoords)*2 (%d)", nparams, (ncoords)*2);

  long bs = THCudaTensor_size(state, data, 0);
  long bs_coords = THCudaTensor_size(state, coords, 0);
  long bs_params = THCudaTensor_size(state, params, 0);
  THArgCheck(bs_coords == bs && bs_params == bs, 0,  "batch size should match: data (%d), coords (%d), params (%d)", bs, bs_coords, bs_params);
  long h = THCudaTensor_size(state, data, 2);
  long h_coords = THCudaTensor_size(state, coords, 2);
  long h_params = THCudaTensor_size(state, params, 2);
  THArgCheck(h_coords == h && h_params == h, 0,  "height should match: data (%d), coords (%d), params (%d)", h, h_coords, h_params);
  long w = THCudaTensor_size(state, data, 3);
  long w_coords = THCudaTensor_size(state, coords, 3);
  long w_params = THCudaTensor_size(state, params, 3);
  THArgCheck(w_coords == w && w_params == w, 0,  "width should match: data (%d), coords (%d), params (%d)", w, w_coords, w_params);


  data = THCudaTensor_newContiguous(state, data);
  coords = THCudaTensor_newContiguous(state, coords);
  params = THCudaTensor_newContiguous(state, params);
  kernels = THCudaTensor_newContiguous(state, kernels);

  sample_weighting_forward_cuda(
      data, coords, params, kernels, output, output_w, nsize);

  THCudaTensor_free(state, data);
  THCudaTensor_free(state, coords);
  THCudaTensor_free(state, params);
  THCudaTensor_free(state, kernels);

  return 0;
}

int sample_weighting_backward(
    THCudaTensor *data,
    THCudaTensor *coords,
    THCudaTensor *params,
    THCudaTensor *kernels,
    THCudaTensor *grad_output,
    THCudaTensor *grad_output_w,
    THCudaTensor *grad_data,
    THCudaTensor *grad_params,
    THCudaTensor *grad_kernels,
    const long nsize) {

  /* TODO: checks */

  data = THCudaTensor_newContiguous(state, data);
  coords = THCudaTensor_newContiguous(state, coords);
  params = THCudaTensor_newContiguous(state, params);
  kernels = THCudaTensor_newContiguous(state, kernels);
  grad_output = THCudaTensor_newContiguous(state, grad_output);
  grad_output_w = THCudaTensor_newContiguous(state, grad_output_w);

  sample_weighting_backward_cuda(
      data, coords, params, kernels, grad_output, grad_output_w, 
      grad_data, grad_params, grad_kernels, nsize);

  THCudaTensor_free(state, data);
  THCudaTensor_free(state, coords);
  THCudaTensor_free(state, params);
  THCudaTensor_free(state, kernels);
  THCudaTensor_free(state, grad_output);
  THCudaTensor_free(state, grad_output_w);

  return 0;
}

} // extern C
