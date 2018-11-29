import os

import torch as th
from torch.autograd import gradcheck, profiler

from .. import functions as funcs

test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, "data")
out_dir = os.path.join(test_dir, "output")

def test_kernel_weighting(gpu=th.cuda.is_available()):
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

def test_kernel_weighting_odd_size(gpu=th.cuda.is_available()):
  bs = 4
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

def test_kernel_weighting_grad(gpu=th.cuda.is_available()):
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

def test_scatter2gather(gpu=th.cuda.is_available()):
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

def test_scatter2gather_grad(gpu=th.cuda.is_available()):
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

