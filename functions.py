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
  """"""
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
    ops.scatter2gather_forward(d_output, d_data)
    return d_data


class KernelWeighting(th.autograd.Function):
  """"""
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
