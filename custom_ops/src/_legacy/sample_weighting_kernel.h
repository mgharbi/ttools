#ifndef SAMPLE_WEIGHTING_KERNEL_H_XGEDSHY3
#define SAMPLE_WEIGHTING_KERNEL_H_XGEDSHY3

#include <THC/THC.h>

extern THCState *state;


#ifdef __cplusplus
extern "C" {
#endif

void sample_weighting_forward_cuda(
    THCudaTensor *data,
    THCudaTensor *coords,
    THCudaTensor *params,
    THCudaTensor *kernels,
    THCudaTensor *output,
    THCudaTensor *output_w,
    const long nsize);

void sample_weighting_backward_cuda(
    THCudaTensor *data,
    THCudaTensor *coords,
    THCudaTensor *params,
    THCudaTensor *kernels,
    THCudaTensor *grad_output,
    THCudaTensor *grad_output_w,
    THCudaTensor *grad_data,
    THCudaTensor *grad_params,
    THCudaTensor *grad_kernels,
    const long nsize);

#ifdef __cplusplus
}
#endif

#endif /* end of include guard: SAMPLE_WEIGHTING_KERNEL_H_XGEDSHY3 */
