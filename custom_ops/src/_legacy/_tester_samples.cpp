#include <Halide.h>
#include <HalideBuffer.h>
#include <HalideRuntimeCuda.h>
#include <iostream>

// #include "sample_integrator_forward.h"
#include "sample_integrator_forward_cuda.h"

// #include "sample_integrator_backward.h"
#include "sample_integrator_backward_cuda.h"

using Halide::Runtime::Buffer;

int main(int argc, char *argv[])
{
  int spp = 4;
  int w = 64;
  int h = 64;
  int ci = 3;
  int n = 4;
  int ncoords = 3;
  int nprojections = 16;
  int nsize = 3;

  Buffer<float> samples(spp, w, h, ci, n);
  Buffer<float> coords(spp, w, h, ncoords, n);
  Buffer<float> projections(ncoords, nprojections);
  Buffer<float> biases(nprojections);
  Buffer<float> output(w, h, ci*nprojections, n);

  Buffer<float> d_output(w, h, ci*nprojections, n);
  Buffer<float> d_samples(spp, w, h, ci, n);
  Buffer<float> d_projections(ncoords, nprojections);
  Buffer<float> d_biases(nprojections);

  int ret;
  void * __user_context;
  ret = sample_integrator_forward_cuda(__user_context,
      samples, coords, projections, biases, 5, output);

  ret = sample_integrator_backward_cuda(__user_context,
      samples, coords, projections, biases, 5, d_output,
      d_samples, d_projections, d_biases);

  return 0;
}
