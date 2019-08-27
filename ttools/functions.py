"""Autograd functional extensions."""
import torch as th
import custom_ops as ops


def _is_cuda(*args):
  """Returns True is any of the argument is on a CUDA device, False otherwise."""
  for a in args:
    if a.is_cuda:
      return True
  return False


class Scatter2Gather(th.autograd.Function):
  """Converts (transpose) scatter kernels into gather kernels.

  Args:
    data(th.Tensor)[bs, k_h, k_w, h, w]: scatter kernel weights.

  Returns:
    data(th.Tensor)[bs, k_h, k_w, h, w]: gather kernel weights.
  """
  @staticmethod
  def forward(ctx, data):
    output = data.new()
    output.resize_as_(data)
    assert len(data.shape) == 5, "data should be 5d"
    if _is_cuda(data):
      ops.scatter2gather_forward_cuda(data, output)
    else:
      ops.scatter2gather_forward(data, output)
    return output

  @staticmethod
  def backward(ctx, d_output):
    d_data = d_output.new()
    d_data.resize_as_(d_output)
    _, kh, kw, _, _ = d_data.shape
    if _is_cuda(d_output):
      ops.scatter2gather_forward_cuda(d_output, d_data)
    else:
      ops.scatter2gather_forward(d_output, d_data)
    return d_data


class KernelWeighting(th.autograd.Function):
  """Locally-weighted average of the input values using kernel weights.

  Args:
    data(th.Tensor)[bs, c, h, w]: input values to be locally averaged
    weights(th.Tensor)[bs, k_h, k_w, h, w]: averaging weights for each k_h x k_w
    neighborhood. Channels are filtered independently.

  Returns:
    output(th.Tensor)[bs, c, h, w]: weighted average of data using weights.
    sum_w(th.Tensor)[bs, h, w]: sum of weights per pixel
  """
  @staticmethod
  def forward(ctx, data, weights):
    bs, c, h, w = data.shape
    output = data.new()
    sum_w = data.new()
    output.resize_as_(data)
    sum_w.resize_(bs, h, w)
    if _is_cuda(data, weights):
      ops.kernel_weighting_forward_cuda(data, weights, output, sum_w)
    else:
      ops.kernel_weighting_forward(data, weights, output, sum_w)
    ctx.save_for_backward(data, weights, sum_w)
    return output, sum_w

  @staticmethod
  def backward(ctx, d_output, d_sum_w):
    data, weights, sum_w = ctx.saved_tensors
    d_data = data.new()
    d_weights = weights.new()
    d_data.resize_as_(data)
    d_weights.resize_as_(weights)
    if _is_cuda(d_output, d_sum_w):
      ops.kernel_weighting_backward_cuda(
        data, weights, sum_w, d_output, d_sum_w, d_data, d_weights)
    else:
      ops.kernel_weighting_backward(
        data, weights, sum_w, d_output, d_sum_w, d_data, d_weights)
    return d_data, d_weights


class KernelLookup(th.autograd.Function):
  """Locally-weighted average of the input values using a bank of kernels
     indexed by kernel_idx.

  Args:
    data(th.Tensor)[bs, c, h, w]: input values to be locally averaged
    kernel_idx(th.IntTensor)[bs, c_out, h, w]: index to select a kernel for
      each output pixel
    weights(th.Tensor)[nkernels, c, k_h, k_w]: averaging weights for each k_h x
      k_w x c neighborhood in data.

  Returns:
    output(th.Tensor)[bs, c_out, h, w]: weighted average of data using weights.
  """
  @staticmethod
  def forward(ctx, data, kernel_idx, weights):
    bs, ci, h, w = data.shape
    co = kernel_idx.shape[1]
    output = data.new()
    output.resize_(bs, co, h, w)
    if _is_cuda(data, kernel_idx, weights):
      ops.kernel_lookup_forward_cuda(data, kernel_idx, weights, output)
    else:
      ops.kernel_lookup_forward(data, kernel_idx, weights, output)
    ctx.save_for_backward(data, kernel_idx, weights)
    return output

  @staticmethod
  def backward(ctx, d_output):
    data, kernel_idx, weights = ctx.saved_tensors
    d_data = data.new()
    d_weights = weights.new()
    d_data.resize_as_(data)
    d_weights.resize_as_(weights)
    if _is_cuda(d_output):
      ops.kernel_lookup_backward_cuda(
        data, kernel_idx, weights, d_output.contiguous(), d_data, d_weights)
    else:
      ops.kernel_lookup_backward(
        data, kernel_idx, weights, d_output.contiguous(), d_data, d_weights)
    return d_data, None, d_weights