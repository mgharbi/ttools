"""Autograd extensions."""
import torch as th
import custom_ops as ops


class Scatter2Gather(th.autograd.Function):
  """"""
  @staticmethod
  def forward(ctx, data):
    output = data.new()
    output.resize_as_(data)
    ops.scatter2gather_forward(data, output)
    return output

  @staticmethod
  def backward(ctx, d_output):
    d_data = d_output.new()
    d_data.resize_as_(d_output)
    ops.scatter2gather_forward(d_output, d_data)
    return d_data


class KernelWeighting(th.autograd.Function):
  """"""
  @staticmethod
  def forward(ctx, data, weights):
    bs, c, h, w = data.shape
    output = data.new()
    output.resize_as_(data)

    sum_w = data.new()
    sum_w.resize_(bs, h, w)

    ops.kernel_weighting_forward(data, weights, output, sum_w)

    ctx.save_for_backward(data, weights, sum_w)

    return output, sum_w

  @staticmethod
  def backward(ctx, d_output, d_sum_w):
    data, weights, sum_w = ctx.saved_variables
    d_data = data.new()
    d_weights = weights.new()

    d_data.resize_as_(data)
    d_weights.resize_as_(weights)

    ops.kernel_weighting_backward(
      data, weights, sum_w, d_output, d_sum_w, d_data, d_weights)

    return d_data, d_weights
