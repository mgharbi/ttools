import unittest
import os

import torch as th
from torch.autograd import gradcheck, profiler

from .. import functions as funcs

test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, "data")
out_dir = os.path.join(test_dir, "output")

class TestKernelWeighting(unittest.TestCase):
  def test_kernel_weighting(self, gpu=th.cuda.is_available()):
    bs = 4
    c = 3
    h = 128
    w = 128

    ksize = 21

    data = 2*th.ones(bs, c, h, w, requires_grad=True)
    weights = th.ones(bs, ksize, ksize, h, w, requires_grad=True)

    if gpu:
      data = data.cuda()
      weights = weights.cuda()

    print("profiling")
    for i in range(5):
      o, s = funcs.KernelWeighting.apply(data, weights)

    with profiler.profile(use_cuda=gpu) as prof:
      for i in range(1):
        o, s = funcs.KernelWeighting.apply(data, weights)
        loss = o.mean()
        loss.backward()
    print(prof)

  def test_kernel_weighting_odd_size(self, gpu=th.cuda.is_available()):
    bs = 8
    c = 3
    h = 122
    w = 122

    ksize = 21

    data = 2*th.ones(bs, c, h, w, requires_grad=True)
    weights = th.ones(bs, ksize, ksize, h, w, requires_grad=True)

    if gpu:
      data = data.cuda()
      weights = weights.cuda()

    print("profiling")
    for i in range(5):
      o, s = funcs.KernelWeighting.apply(data, weights)

    with profiler.profile(use_cuda=gpu) as prof:
      for i in range(1):
        o, s = funcs.KernelWeighting.apply(data, weights)
        loss = o.mean()
        loss.backward()
    print(prof)

  def test_kernel_weighting_grad(self, gpu=th.cuda.is_available()):
    bs = 2
    c = 2
    h = 32
    w = 32

    ksize = 3

    print("testing data gradient")
    data = 2*th.randn(bs, c, h, w, requires_grad=True)
    weights = th.randn(bs, ksize, ksize, h, w, requires_grad=False)
    if gpu:
      data = data.cuda()
      weights = weights.cuda()

    o, s = funcs.KernelWeighting.apply(data, weights)

    gradcheck(
        funcs.KernelWeighting.apply,
        (data, weights),
        eps=1e-4, atol=5e-2, rtol=5e-4,
         )

    print("testing weight gradient")
    data = 2*th.randn(bs, c, h, w, requires_grad=False)
    weights = th.randn(bs, ksize, ksize, h, w, requires_grad=True)
    if gpu:
      data = data.cuda()
      weights = weights.cuda()

    gradcheck(
        funcs.KernelWeighting.apply,
        (data, weights),
        eps=1e-4, atol=5e-2, rtol=5e-4,
         )


class TestScatter2Gather(unittest.TestCase):
  def test_scatter2gather(self, gpu=th.cuda.is_available()):
    bs = 4
    c = 3
    h = 128
    w = 128
    ksize = 1

    weights = th.ones(bs, ksize, ksize, h, w, requires_grad=True)

    if gpu:
      weights = weights.cuda()

    print("profiling")
    for i in range(5):
      w2 = funcs.Scatter2Gather.apply(weights)

    with profiler.profile(use_cuda=gpu) as prof:
      for i in range(1):
        w2 = funcs.Scatter2Gather.apply(weights)
        loss = w2.mean()
        loss.backward()
    print(prof)

  def test_scatter2gather_grad(self, gpu=th.cuda.is_available()):
    bs = 2
    c = 2
    h = 32
    w = 32

    ksize = 3

    print("testing gradient")
    weights = th.randn(bs, ksize, ksize, h, w, requires_grad=True)
    if gpu:
      weights = weights.cuda()

    gradcheck(
      funcs.Scatter2Gather.apply,
      (weights, ),
      eps=1e-4, atol=5e-2, rtol=5e-4,
    )


class TestKernelLookup(unittest.TestCase):
  def setUp(self):
    bs = 4
    ci = 1
    co = 3
    h = 16
    w = 16
    self.ksize = 5
    self.nkernels = 64

    self.data = th.zeros(bs, ci, h, w, requires_grad=False)
    self.kernel_idx = th.ones(bs, co, h, w, requires_grad=False).int()
    self.weights = th.ones(self.nkernels, ci, self.ksize, self.ksize, requires_grad=False)

    if th.cuda.is_available():
      self.data = self.data.cuda()
      self.kernel_idx = self.kernel_idx.cuda()
      self.weights = self.weights.cuda()
  #
  # def test_kernel_lookup(self):
  #   for i in range(5):
  #     o = funcs.KernelLookup.apply(self.data, self.kernel_idx, self.weights)
  #
  #   with profiler.profile(use_cuda=th.cuda.is_available()) as prof:
  #     for i in range(1):
  #       o = funcs.KernelLookup.apply(self.data, self.kernel_idx, self.weights)
  #   print(prof)
  #   print(o.mean().item())

  def test_choose_right_kernel(self):
    # Each output channel uses a different kernel
    self.kernel_idx[0, 0, 5, 3] = 1
    self.kernel_idx[0, 1, 5, 4] = 2  # next channel, pixel to the right
    self.kernel_idx[0, 2, 6, 3] = 3  # next chanel, pixel below

    # Set those kernels
    self.weights.fill_(0.0)
    self.weights[1, 0, 2, 2] = 1  # centered
    self.weights[2, 0, 2, 1] = 1  # pixel to the left
    self.weights[3, 0, 1, 2] = 1  # pixel above

    # A single dirac in the input data
    self.data.fill_(0.0)
    self.data[0, :, 5, 3] = 1.0

    # Filter
    o = funcs.KernelLookup.apply(self.data, self.kernel_idx, self.weights)

    # Check the kernel assignments are correct
    self.assertLess((o[0, 0] - self.data[0, 0]).abs().sum(), 1e-8)
    self.assertAlmostEqual(o[0, 1, 5, 4].item(), 1.0)
    self.assertAlmostEqual(o[0, 2, 6, 3].item(), 1.0)

