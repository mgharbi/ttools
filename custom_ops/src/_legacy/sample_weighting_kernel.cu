#include <THC/THC.h>
#include "sample_weighting_kernel.h"

#define BLOCK_SIZE 256
#define BLOCK_COUNT(n) \
  (n + BLOCK_SIZE - 1) / BLOCK_SIZE

#define KERNEL_LOOP(index, count) \
  const int index = blockIdx.x*blockDim.x + threadIdx.x; \
  if (index < count)

#define GET_KERNEL_POINT(xx, yy, ci_, co_) \
  kernels[xx + kw*(yy + kh*(co_ + co*ci_))]

__global__ void sample_weighting_forward_kernel(
    const float* data, 
    const float* coords,
    const float* params, 
    const float* kernels, 
    float*  output, 
    float*  output_w, 
    const int bs,
    const int h,
    const int w,
    const int spp,
    const int ci,
    const int kh,
    const int kw,
    const int co, 
    const int ncoords,
    const int nsize) {

  const int sample_chan_stride = h*w*spp;
  const int pxs = h*w;
  const int nparams = 2*ncoords; // 2 output coordinates

  const int out_count = bs*co*h*w;
  const int coords_batch_stride = ncoords*h*w*spp;
  const int data_batch_stride = ci*h*w*spp;
  const int params_batch_stride = nparams*h*w;
  const int output_batch_stride = co*h*w;

  KERNEL_LOOP(index, out_count) {
    const int batch_id = index / output_batch_stride;
    const int index_in_batch = index % output_batch_stride;
    const int co_ = index_in_batch / pxs;
    const int pixel = index_in_batch % pxs;
    const int x = pixel % w;
    const int y = pixel / w;

    const float* p = params + batch_id*params_batch_stride + pixel;

    float v_output = 0.0f;
    float v_output_w = 0.0f;
    for(int ny = -nsize/2; ny < nsize/2 + 1; ++ ny )
    for(int nx = -nsize/2; nx < nsize/2 + 1; ++ nx ) {
      const bool neighbor_in_range = (x+nx >= 0) && (x+nx) < w &&
        (y+ny >= 0) && (y+ny) < h;
      if( !neighbor_in_range ) {
        // TODO: implement 'VALID' boundary condition
        continue;
      }

      const int neigh_pixel = x+nx + w*(y+ny);

      for(int s = 0; s < spp; ++s) {
        const int sample_id = (s + spp*(neigh_pixel));
        const float *cur_coords = 
          coords + (batch_id*coords_batch_stride + sample_id);

        // First two dims are spatial
        const float sx = nx + cur_coords[0*sample_chan_stride];
        const float sy = ny + cur_coords[1*sample_chan_stride];

        // center relative pixel coordinates in kernel
        float csx = (sx-0.5) / (0.5*nsize);
        float csy = (sy-0.5) / (0.5*nsize);

        float sx2 = p[0*pxs]*csx + p[1*pxs]*csy;
        float sy2 = p[(ncoords + 0)*pxs]*csx + p[(ncoords + 1)*pxs]*csy;

        for(int coord_ = 2; coord_ < ncoords; ++ coord_) {
          const float cval = cur_coords[coord_*sample_chan_stride];
          sx2 += p[(coord_)*pxs]*cval;
          sy2 += p[(ncoords + coord_)*pxs]*cval;
        }

        // Bring back to kernel space
        sx2 = 0.5*(sx2 + 1.0f)*kw;
        sy2 = 0.5*(sy2 + 1.0f)*kh;

        if (sx2-0.5 > static_cast<float>(-1.0) &&
            sy2-0.5 > static_cast<float>(-1.0) &&
            sx2-0.5 < static_cast<float>(kw) &&
            sy2-0.5 < static_cast<float>(kh)) {

          // Precompute floor (f) and ceil (c) values for x and y.
          const int fx = std::floor(static_cast<float>(sx2-0.5));
          const int fy = std::floor(static_cast<float>(sy2-0.5));
          const int cx = fx + 1;
          const int cy = fy + 1;
          const float dx = static_cast<float>(cx) - (sx2-0.5);
          const float dy = static_cast<float>(cy) - (sy2-0.5);

          for (int ci_ = 0; ci_ < ci; ++ci_) {
            float input_value = data[batch_id*data_batch_stride +
              ci_*sample_chan_stride + sample_id];

              const float k1 = (fx >= 0 && fy >= 0) ? 
                GET_KERNEL_POINT(fx, fy, ci_, co_) : 0.0f;
              const float k2 = (cx <= kw - 1 && cy <= kh - 1) ? 
                GET_KERNEL_POINT(cx, cy, ci_, co_): 0.0f;
              const float k3 = (fx >= 0 && cy <= kh - 1) ? 
                GET_KERNEL_POINT(fx, cy, ci_, co_) : 0.0f;
              const float k4 = (cx <= kw - 1 && fy >= 0) ? 
                GET_KERNEL_POINT(cx, fy, ci_, co_) : 0.0f;

              const float k_fxfy = dx*dy*k1;
              const float k_cxcy = (1.0f-dx)*(1.0f-dy)*k2;
              const float k_fxcy = dx*(1.0f-dy)*k3;
              const float k_cxfy = (1.0f-dx)*dy*k4;

              const float kweight = k_fxfy + k_cxcy + k_fxcy + k_cxfy;

              v_output += kweight*input_value;
              v_output_w += kweight;
          }
        } // else both (f) and (c) are outside the kernel
      } // spp
    } // nx, ny
    output[index] = v_output;
    output_w[index] = v_output_w;
  } // kernel loop
}


__device__ __forceinline__ float get_dsx2_dp(
    const float sx, const float sy, const float *coords,
    const int coord_, const int kw, const int sample_chan_stride) {
  switch(coord_) {
    case 0:
      return float(0.5)*kw*sx;
    case 1:
      return float(0.5)*kw*sy;
    default:
      return float(0.5)*kw*coords[(coord_)*sample_chan_stride];
  }
}


__device__ __forceinline__ float get_dsy2_dp(
    const float sx, const float sy, const float *coords,
    const int coord_, const int kh, const int sample_chan_stride) {
  switch(coord_) {
    case 0:
      return float(0.5)*kh*sx;
    case 1:
      return float(0.5)*kh*sy;
    default:
      return float(0.5)*kh*coords[(coord_)*sample_chan_stride];
  }
}


__global__  __launch_bounds__(1024,2) void sample_weighting_backward_kernel(
    const float* data,
    const float* coords,
    const float* params,
    const float* kernels,
    const float* grad_output,
    const float* grad_output_w,
    float* grad_data,
    float* grad_params,
    float* grad_kernels,
    const int bs,
    const int h,
    const int w,
    const int spp,
    const int ci,
    const int kh,
    const int kw,
    const int co,
    const int ncoords,
    const int nsize) {

  const int nparams = 2*(ncoords); // 2 output coordinates
  const int coords_batch_stride = ncoords*h*w*spp;
  const int data_batch_stride = ci*h*w*spp;
  const int params_batch_stride = nparams*h*w;
  const int output_batch_stride = co*h*w;

  const int sample_chan_stride = h*w*spp;
  const int pxs = h*w;
  const int npixels = bs*h*w;

  KERNEL_LOOP(index, npixels) {
    const int batch_id = index / pxs;
    const int pixel = index % pxs;
    const int x = pixel % w;
    const int y = pixel / w;

    for(int ny = -nsize/2; ny < nsize/2 + 1; ++ ny )
    for(int nx = -nsize/2; nx < nsize/2 + 1; ++ nx ) {
      const bool neighbor_in_range = (x+nx >= 0) && (x+nx) < w &&
        (y+ny >= 0) && (y+ny) < h;
      if( !neighbor_in_range ) {
        // TODO: implement 'VALID' boundary condition
        continue;
      }

      const int neigh_pixel = x+nx + w*(y+ny);

      const float* p = params + batch_id*params_batch_stride + neigh_pixel;

      for(int s = 0; s < spp; ++s) {
        const int sample_id = (s + spp*pixel);
        const float *cur_coords = 
          coords + (batch_id*coords_batch_stride + sample_id);

        // First two dims are spatial
        const float sx = cur_coords[0*sample_chan_stride];
        const float sy = cur_coords[1*sample_chan_stride];
        
        // center relative pixel coordinates in kernel
        float csx = (sx-(nx+0.5)) / (0.5*nsize);
        float csy = (sy-(ny+0.5)) / (0.5*nsize);

        float sx2 = p[0*pxs]*csx + p[1*pxs]*csy;
        float sy2 = p[(ncoords + 0)*pxs]*csx + p[(ncoords + 1)*pxs]*csy;

        for(int coord_ = 2; coord_ < ncoords; ++ coord_) {
          const float cval = cur_coords[coord_*sample_chan_stride];
          sx2 += p[(coord_)*pxs]*cval;
          sy2 += p[(ncoords + coord_)*pxs]*cval;
        }

        // Bring back to kernel space
        sx2 = 0.5*(sx2 + 1.0f)*kw;
        sy2 = 0.5*(sy2 + 1.0f)*kh;

        if (sx2-0.5 > static_cast<float>(-1.0) &&
            sy2-0.5 > static_cast<float>(-1.0) &&
            sx2-0.5 < static_cast<float>(kw) &&
            sy2-0.5 < static_cast<float>(kh)) {
          // Precompute floor (f) and ceil (c) values for x and y.
          const int fx = std::floor(static_cast<float>(sx2-0.5));
          const int fy = std::floor(static_cast<float>(sy2-0.5));
          const int cx = fx + 1;
          const int cy = fy + 1;
          const float dx = static_cast<float>(cx) - (sx2-0.5);
          const float dy = static_cast<float>(cy) - (sy2-0.5);

          for (int co_ = 0; co_ < co; ++co_) {
            float grad_output_value = 
              grad_output[batch_id*output_batch_stride +
              co_*pxs + neigh_pixel];
            float grad_output_w_value = 
              grad_output_w[batch_id*output_batch_stride +
              co_*pxs + neigh_pixel];
            for (int ci_ = 0; ci_ < ci; ++ci_) {
              float input_value = data[batch_id*data_batch_stride + 
                ci_*sample_chan_stride + sample_id];

              const float k1 = (fx >= 0 && fy >= 0) ? 
                GET_KERNEL_POINT(fx, fy, ci_, co_) : 0.0f;
              const float k2 = (cx <= kw - 1 && cy <= kh - 1) ? 
                GET_KERNEL_POINT(cx, cy, ci_, co_): 0.0f;
              const float k3 = (fx >= 0 && cy <= kh - 1) ? 
                GET_KERNEL_POINT(fx, cy, ci_, co_) : 0.0f;
              const float k4 = (cx <= kw - 1 && fy >= 0) ? 
                GET_KERNEL_POINT(cx, fy, ci_, co_) : 0.0f;

              const float k_fxfy = dx*dy*k1;
              const float k_cxcy = (1.0f-dx)*(1.0f-dy)*k2;
              const float k_fxcy = dx*(1.0f-dy)*k3;
              const float k_cxfy = (1.0f-dx)*dy*k4;

              const float kweight = k_fxfy + k_cxcy + k_fxcy + k_cxfy;

              grad_data[batch_id*data_batch_stride + 
                ci_*sample_chan_stride + sample_id]
                += kweight*grad_output_value;

              // Update partial gradients wrt kernel
              if (fx >= 0 && fy >= 0) {
                atomicAdd(&grad_kernels[fx + kw*(fy + kh*(co_ + co*ci_))], 
                  (grad_output_w_value + 
                   input_value*grad_output_value) * dx * dy);
              }
              if (cx <= kw - 1 && cy <= kh - 1) {
                atomicAdd(&grad_kernels[cx + kw*(cy + kh*(co_ + co*ci_))], 
                  (grad_output_w_value + 
                   input_value*grad_output_value) * (1.0f-dx) * (1.0f-dy));
              }
              if (fx >= 0 && cy <= kh - 1) {
                atomicAdd(&grad_kernels[fx + kw*(cy + kh*(co_ + co*ci_))], 
                  (grad_output_w_value + 
                   input_value*grad_output_value) * dx * (1.0f-dy));
              }
              if (cx <= kw - 1 && fy >= 0) {
                atomicAdd(&grad_kernels[cx + kw*(fy + kh*(co_ + co*ci_))], 
                  (grad_output_w_value + 
                   input_value*grad_output_value) * (1.0f-dx) * dy);
              }

              for (int coord_ = 0; coord_ < ncoords; ++coord_) {
                // params 0...ncoords affect sx2 only
                const float dsx2_dp = get_dsx2_dp(
                    csx, csy, cur_coords, coord_, kw, sample_chan_stride);
                float d_weight_dp1 = 
                  (-dy*k1 + (1.0f-dy)*k2 - 1.0f*(1.0f-dy)*k3 + dy*k4)*dsx2_dp;
                atomicAdd(&grad_params[batch_id*params_batch_stride + 
                    coord_*pxs + neigh_pixel],
                    (grad_output_w_value + 
                     input_value*grad_output_value)*d_weight_dp1);

                // params ncoords+1...2*ncoords affect sy2 only
                const float dsy2_dp = get_dsy2_dp(
                    csx, csy, cur_coords, coord_, kh, sample_chan_stride);
                float d_weight_dp2 = 
                  (-dx*k1 + (1.0f-dx)*k2 + dx*k3 - (1.0f-dx)*k4)*dsy2_dp;
                atomicAdd(&grad_params[batch_id*params_batch_stride + 
                    (coord_ + ncoords)*pxs + neigh_pixel],
                    (grad_output_w_value + 
                     input_value*grad_output_value)*d_weight_dp2);
              } // coord_
            } // ci_
          } // co_
        } // else both (f) and (c) are outside the kernel
      } // s loop
    } // ny, nx (neighborhood pixel)
  } // kernel
} // backward kernel


void sample_weighting_forward_cuda(
    THCudaTensor *data,
    THCudaTensor *coords,
    THCudaTensor *params,
    THCudaTensor *kernels,
    THCudaTensor *output,
    THCudaTensor *output_w,
    const long nsize) {

  int bs = THCudaTensor_size(state, data, 0);
  int ci = THCudaTensor_size(state, data, 1);
  int h = THCudaTensor_size(state, data, 2);
  int w = THCudaTensor_size(state, data, 3);
  int spp = THCudaTensor_size(state, data, 4);

  int ncoords = THCudaTensor_size(state, coords, 1);

  int kh = THCudaTensor_size(state, kernels, 2);
  int kw = THCudaTensor_size(state, kernels, 3);
  int co = THCudaTensor_size(state, kernels, 1);

  THCudaTensor_resize4d(state, output, bs, co, h, w);
  THCudaTensor_resize4d(state, output_w, bs, co, h, w);

  float* pData = THCudaTensor_data(state, data);
  float* pCoords = THCudaTensor_data(state, coords);
  float* pParams = THCudaTensor_data(state, params);
  float* pKernels = THCudaTensor_data(state, kernels);
  float* pOutput = THCudaTensor_data(state, output);
  float* pOutput_w = THCudaTensor_data(state, output_w);

  const int64_t count = bs*h*w*co;
  const int64_t blocks = BLOCK_COUNT(count);  

  sample_weighting_forward_kernel
    <<<blocks, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>> (
    pData, pCoords, pParams, pKernels, 
    pOutput, pOutput_w, 
    bs, h, w, spp, ci, kh, kw, co, ncoords, nsize);
  THCudaCheck(cudaPeekAtLastError());
}


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
    const long nsize) {
  int bs = THCudaTensor_size(state, data, 0);
  int ci = THCudaTensor_size(state, data, 1);
  int h = THCudaTensor_size(state, data, 2);
  int w = THCudaTensor_size(state, data, 3);
  int spp = THCudaTensor_size(state, data, 4);

  int ncoords = THCudaTensor_size(state, coords, 1);

  int co = THCudaTensor_size(state, kernels, 1);
  int kh = THCudaTensor_size(state, kernels, 2);
  int kw = THCudaTensor_size(state, kernels, 3);

  int nparams = THCudaTensor_size(state, params, 1);

  THCudaTensor_resize5d(state, grad_data, bs, ci, h, w, spp);
  THCudaTensor_resize4d(state, grad_params, bs, nparams, h, w);
  THCudaTensor_resize4d(state, grad_kernels, ci, co, kh, kw);
  THCudaTensor_zero(state, grad_data);
  THCudaTensor_zero(state, grad_params);
  THCudaTensor_zero(state, grad_kernels);

  float* pData = THCudaTensor_data(state, data);
  float* pCoords = THCudaTensor_data(state, coords);
  float* pParams = THCudaTensor_data(state, params);
  float* pKernels = THCudaTensor_data(state, kernels);
  float* pGradOutput = THCudaTensor_data(state, grad_output);
  float* pGradOutput_w = THCudaTensor_data(state, grad_output_w);

  float* pGradData = THCudaTensor_data(state, grad_data);
  float* pGradParams = THCudaTensor_data(state, grad_params);
  float* pGradKernels = THCudaTensor_data(state, grad_kernels);

  const int64_t blocks = BLOCK_COUNT(bs*h*w);  

  sample_weighting_backward_kernel
    <<<blocks, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>> (
    pData, pCoords, pParams, pKernels, pGradOutput, pGradOutput_w,
    pGradData, pGradParams, pGradKernels, 
    bs, h, w, spp, ci, kh, kw, co, ncoords, nsize);
  THCudaCheck(cudaPeekAtLastError());
}
