#include <Halide.h>
#include <HalideBuffer.h>
#include <HalideRuntimeCuda.h>
#include <iostream>

#include "sample_weighting_forward_cuda.h"
#include "sample_weighting_backward_cuda.h"

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
  Buffer<float> projections(w, h, ncoords, 2*nprojections, n);
  Buffer<float> output(w, h, ci*nprojections, n);

  Buffer<float> d_output(w, h, ci*nprojections, n);
  Buffer<float> d_samples(spp, w, h, ci, n);
  Buffer<float> d_projections(w, h, ncoords, 2*nprojections, n);
  Buffer<float> d_biases(nprojections);

  void * __user_context;
  int ret;
  for (int i = 0; i < 10; ++i) {
    ret = sample_weighting_forward_cuda(__user_context,
        samples, coords, projections, nsize, output);
    assert(ret == 0);

    ret = sample_weighting_backward_cuda(__user_context,
        samples, coords, projections, nsize, d_output,
        d_samples, d_projections);
    assert(ret == 0);
  }

  return 0;
}
