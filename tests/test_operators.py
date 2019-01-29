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
  def test_kernel_lookup(self, gpu=th.cuda.is_available()):
    bs = 4
    ci = 1
    co = 3
    h = 16
    w = 16

    ksize = 3

    nkernels = 8

    data = th.ones(bs, ci, h, w, requires_grad=False)
    kernel_idx = th.ones(bs, co, h, w, requires_grad=False).int()
    weights = th.ones(nkernels, ci, ksize, ksize, requires_grad=False)

    if gpu:
      data = data.cuda()
      kernel_idx = kernel_idx.cuda()
      weights = weights.cuda()

    # print("profiling")
    # for i in range(5):
    o = funcs.KernelLookup.apply(data, kernel_idx, weights)

    print(o)

    # with profiler.profile(use_cuda=gpu) as prof:
    #   for i in range(1):
    #     o, s = funcs.KernelWeighting.apply(data, weights)
    #     loss = o.mean()
    #     loss.backward()
    # print(prof)
