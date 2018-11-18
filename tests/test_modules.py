import unittest

import torch as th
import torch.nn as nn

from ..modules import networks

class TestConvModule(unittest.TestCase):
  def setUp(self):
    self.bs = 1
    self.c = 3
    self.c_out = 4
    self.h = 16
    self.w = 16
    self.in_ = th.ones(self.bs, self.c, self.h, self.w)

  def test_basic_conv(self):
    cv = networks._ConvModule(self.c, self.c_out, 3)
    out_ = cv(self.in_)
    self.assertListEqual(list(out_.shape), [self.bs, self.c_out, self.h, self.w])

    self.assertIsNotNone(cv.conv)
    self.assertIsNotNone(cv.activation)
    self.assertIsNotNone(cv.conv.weight)
    self.assertIsNotNone(cv.conv.bias)

  def test_norm(self):
    cv = networks._ConvModule(self.c, self.c_out, 3, norm_layer="instance")
    self.assertIsNone(cv.conv.bias)
    self.assertIsNotNone(cv.norm)

  def test_no_pad(self):
    k = 3
    cv = networks._ConvModule(self.c, self.c_out, k, pad=False)
    out_ = cv(self.in_)
    self.assertListEqual(list(out_.shape), [self.bs, self.c_out, self.h-k+1, self.w-k+1])

  def test_strided(self):
    k = 3
    cv = networks._ConvModule(self.c, self.c_out, k, stride=2)
    out_ = cv(self.in_)
    self.assertListEqual(list(out_.shape), [self.bs, self.c_out, self.h/2, self.w/2])


class TestConvChain(unittest.TestCase):
  def setUp(self):
    self.bs = 1
    self.c = 3
    self.c_out = 4
    self.h = 16
    self.w = 16
    self.in_data = th.ones(self.bs, self.c, self.h, self.w)

  def test_default(self):
    w = 32
    k = 3
    cv = networks.ConvChain(self.c, self.c_out, depth=5, width=w, ksize=k)

    self.assertListEqual(list(cv.conv0.conv.weight.shape), [w, self.c, k, k])
    self.assertListEqual(list(cv.conv1.conv.weight.shape), [w, w, k, k])
    self.assertListEqual(list(cv.conv2.conv.weight.shape), [w, w, k, k])
    self.assertListEqual(list(cv.conv3.conv.weight.shape), [w, w, k, k])
    self.assertListEqual(list(cv.conv4.conv.weight.shape), [self.c_out, w, k, k])
    self.assertRaises(AttributeError, getattr, cv, "conv5")
    self.assertEqual(len(list(cv.children())), 5)

    self.assertEqual(len(list(cv.conv0.children())), 2)
    self.assertEqual(len(list(cv.conv1.children())), 2)
    self.assertEqual(len(list(cv.conv2.children())), 2)
    self.assertEqual(len(list(cv.conv3.children())), 2)
    self.assertEqual(len(list(cv.conv4.children())), 1)

    out_ = cv(self.in_data)
    self.assertEqual(out_.shape[1], self.c_out)
    
  def test_output_activation(self):
    w = 32
    k = 3
    cv = networks.ConvChain(self.c, self.c_out, depth=5, width=w, ksize=k, out_activation="relu")
    self.assertEqual(len(list(cv.conv4.children())), 2)

  def test_normalization(self):
    w = 32
    k = 3
    cv = networks.ConvChain(self.c, self.c_out, depth=4, width=w, ksize=k, norm_layer="batch")
    self.assertEqual(len(list(cv.conv0.children())), 2)  # no norm
    self.assertEqual(len(list(cv.conv1.children())), 3)
    self.assertEqual(len(list(cv.conv2.children())), 3)
    self.assertEqual(len(list(cv.conv3.children())), 1) # no norm, no activation

    self.assertRaises(AttributeError, getattr, cv.conv0, "norm")
    self.assertIsNotNone(cv.conv1.norm)
    self.assertIsNotNone(cv.conv2.norm)
    self.assertRaises(AttributeError, getattr, cv.conv3, "norm")

  def test_even_padding(self):
    w = 32
    k = 4
    cv = networks.ConvChain(self.c, self.c_out, depth=3, width=w, ksize=k)
    out_ = cv(self.in_data)
    self.assertEqual(out_.shape[1], self.c_out)
    
  def test_even_no_padding(self):
    w = 32
    k = 4
    depth = 2
    cv = networks.ConvChain(self.c, self.c_out, depth=depth, width=w, ksize=k, pad=False)
    out_ = cv(self.in_data)
    self.assertEqual(out_.shape[2], self.h - depth*(k-1))
    self.assertEqual(out_.shape[3], self.w - depth*(k-1))

  def test_variable_width(self):
    # Width should have 3-1 = 2 values
    self.assertRaises(AssertionError, networks.ConvChain, self.c, self.c_out, depth=3, width=[12])

    k = 3
    cv = networks.ConvChain(self.c, self.c_out, ksize=k, depth=3, width=[12, 24])
    
    # Check sizes and children length is correct
    self.assertListEqual(list(cv.conv0.conv.weight.shape), [12, self.c, k, k])
    self.assertListEqual(list(cv.conv1.conv.weight.shape), [24, 12, k, k])
    self.assertListEqual(list(cv.conv2.conv.weight.shape), [self.c_out, 24, k, k])
    self.assertRaises(AttributeError, getattr, cv, "conv3" )
    self.assertEqual(len(list(cv.children())), 3)

  def test_variable_kernel_size(self):
    width = 32
    cv = networks.ConvChain(self.c, self.c_out, ksize=[3, 5, 3], depth=3, width=width)
    
    # Check sizes and children length is correct
    self.assertListEqual(list(cv.conv0.conv.weight.shape), [width, self.c, 3, 3])
    self.assertListEqual(list(cv.conv1.conv.weight.shape), [width, width, 5, 5])
    self.assertListEqual(list(cv.conv2.conv.weight.shape), [self.c_out, width, 3, 3])

  def test_strided(self):
    k = 3
    # 3 strides should be passed
    self.assertRaises(AssertionError, networks.ConvChain, self.c, self.c_out, ksize=k, depth=3, width=32, strides=[1, 2])

    cv = networks.ConvChain(self.c, self.c_out, ksize=k, depth=3, width=32, strides=[1, 2, 2])
    
    # Check sizes and children length is correct
    self.assertEqual(cv.conv0.conv.stride[0],  1)
    self.assertEqual(cv.conv1.conv.stride[0],  2)
    self.assertEqual(cv.conv2.conv.stride[0],  2)


class TestFCModule(unittest.TestCase):
  def test_basic_fc(self):
    bs = 1
    c = 16
    c_out = 32

    fc = networks._FCModule(c, c_out, dropout=0.5)
