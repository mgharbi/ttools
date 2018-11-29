int sample_weighting_forward(
    THCudaTensor *data,
    THCudaTensor *coords,
    THCudaTensor *params,
    THCudaTensor *kernels,
    THCudaTensor *output,
    THCudaTensor *output_w,
    const long nsize);

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
    const long nsize);

