import unittest

import torch as th
import torch.nn as nn

from ..modules import networks
from ..modules import ops


class TestConvModule(unittest.TestCase):
    def setUp(self):
        self.bs = 1
        self.c = 3
        self.c_out = 4
        self.h = 16
        self.w = 16
        self.in_ = th.ones(self.bs, self.c, self.h, self.w)

    def test_basic_conv(self):
        cv = networks.ConvModule(self.c, self.c_out, 3, activation="relu")
        out_ = cv(self.in_)

        self.assertListEqual(list(out_.shape), [
                             self.bs, self.c_out, self.h, self.w])
        self.assertIsNotNone(cv.conv)
        self.assertIsNotNone(cv.activation)
        self.assertIsNotNone(cv.conv.weight)
        self.assertIsNotNone(cv.conv.bias)

        # Default layer should be linear
        cv = networks.ConvModule(self.c, self.c_out, 3)
        self.assertRaises(AttributeError, getattr, cv, "activation")

    def test_norm(self):
        cv = networks.ConvModule(self.c, self.c_out, 3, norm_layer="instance")
        self.assertIsNone(cv.conv.bias)
        self.assertIsNotNone(cv.norm)

    def test_no_pad(self):
        k = 3
        cv = networks.ConvModule(self.c, self.c_out, k, pad=False)
        out_ = cv(self.in_)
        self.assertListEqual(list(out_.shape), [
                             self.bs, self.c_out, self.h-k+1, self.w-k+1])

    def test_strided(self):
        k = 3
        cv = networks.ConvModule(self.c, self.c_out, k, stride=2)
        out_ = cv(self.in_)
        self.assertListEqual(list(out_.shape), [
                             self.bs, self.c_out, self.h/2, self.w/2])


class TestConvChain(unittest.TestCase):
    def setUp(self):
        self.bs = 1
        self.c = 3
        self.h = 16
        self.w = 16
        self.in_data = th.ones(self.bs, self.c, self.h, self.w)

    def test_default(self):
        w = 32
        k = 3
        cv = networks.ConvChain(self.c, depth=5, width=w, ksize=k)

        self.assertListEqual(
            list(cv.conv0.conv.weight.shape), [w, self.c, k, k])
        self.assertListEqual(list(cv.conv1.conv.weight.shape), [w, w, k, k])
        self.assertListEqual(list(cv.conv2.conv.weight.shape), [w, w, k, k])
        self.assertListEqual(list(cv.conv3.conv.weight.shape), [w, w, k, k])
        self.assertListEqual(list(cv.conv4.conv.weight.shape), [
                             w, w, k, k])
        self.assertRaises(AttributeError, getattr, cv, "conv5")
        self.assertEqual(len(list(cv.children())), 5)

        self.assertEqual(len(list(cv.conv0.children())), 2)
        self.assertEqual(len(list(cv.conv1.children())), 2)
        self.assertEqual(len(list(cv.conv2.children())), 2)
        self.assertEqual(len(list(cv.conv3.children())), 2)
        self.assertEqual(len(list(cv.conv4.children())), 2)

        out_ = cv(self.in_data)
        self.assertListEqual(
            list(out_.shape), [self.bs, w, self.h, self.w])

    def test_output_activation(self):
        w = 32
        k = 3
        cv = networks.ConvChain(
            self.c, depth=5, width=w, ksize=k)
        self.assertEqual(len(list(cv.conv4.children())), 2)

    def test_normalization(self):
        w = 32
        k = 3
        cv = networks.ConvChain(
            self.c, depth=4, width=w, ksize=k, norm_layer="batch")
        self.assertEqual(len(list(cv.conv0.children())), 3)
        self.assertEqual(len(list(cv.conv1.children())), 3)
        self.assertEqual(len(list(cv.conv2.children())), 3)
        self.assertEqual(len(list(cv.conv3.children())), 3)

        self.assertIsNotNone(cv.conv0.norm)
        self.assertIsNotNone(cv.conv1.norm)
        self.assertIsNotNone(cv.conv2.norm)
        self.assertIsNotNone(cv.conv3.norm)

    def test_even_padding(self):
        w = 32
        k = 4
        cv = networks.ConvChain(self.c, depth=3, width=w, ksize=k)
        out_ = cv(self.in_data)
        self.assertEqual(out_.shape[1], w)

    def test_even_no_padding(self):
        w = 32
        k = 4
        depth = 2
        cv = networks.ConvChain(
            self.c, depth=depth, width=w, ksize=k, pad=False)
        out_ = cv(self.in_data)
        self.assertEqual(out_.shape[2], self.h - depth*(k-1))
        self.assertEqual(out_.shape[3], self.w - depth*(k-1))

    def test_variable_width(self):
        # Width should have 3-1 = 2 values
        self.assertRaises(AssertionError, networks.ConvChain,
                          self.c, depth=3, width=[12])

        k = 3
        cv = networks.ConvChain(
            self.c, ksize=k, depth=3, width=[12, 24, 21])

        # Check sizes and children length is correct
        self.assertListEqual(
            list(cv.conv0.conv.weight.shape), [12, self.c, k, k])
        self.assertListEqual(list(cv.conv1.conv.weight.shape), [24, 12, k, k])
        self.assertListEqual(list(cv.conv2.conv.weight.shape), [
                             21, 24, k, k])
        self.assertRaises(AttributeError, getattr, cv, "conv3")
        self.assertEqual(len(list(cv.children())), 3)

    def test_variable_kernel_size(self):
        width = 32
        cv = networks.ConvChain(self.c, ksize=[
                                3, 5, 3], depth=3, width=width)

        # Check sizes and children length is correct
        self.assertListEqual(list(cv.conv0.conv.weight.shape), [
                             width, self.c, 3, 3])
        self.assertListEqual(list(cv.conv1.conv.weight.shape), [
                             width, width, 5, 5])
        self.assertListEqual(list(cv.conv2.conv.weight.shape), [
                             width, width, 3, 3])

    def test_strided(self):
        k = 3
        # 3 strides should be passed
        self.assertRaises(AssertionError, networks.ConvChain, self.c,
                          ksize=k, depth=3, width=32, strides=[1, 2])

        cv = networks.ConvChain(
            self.c, ksize=k, depth=3, width=32, strides=[1, 2, 2])

        # Check sizes and children length is correct
        self.assertEqual(cv.conv0.conv.stride[0],  1)
        self.assertEqual(cv.conv1.conv.stride[0],  2)
        self.assertEqual(cv.conv2.conv.stride[0],  2)


class TestFCModule(unittest.TestCase):
    def setUp(self):
        self.bs = 1
        self.c = 16
        self.c_out = 32
        self._in = th.ones(self.bs, self.c)

    def test_basic_fc(self):
        fc = networks.FCModule(self.c, self.c_out, dropout=0.5, activation="relu")
        out_ = fc(self._in)

        self.assertListEqual(list(out_.shape), [self.bs, self.c_out])
        self.assertIsNotNone(fc.fc)
        self.assertIsNotNone(fc.activation)
        self.assertIsNotNone(fc.dropout)
        self.assertIsNotNone(fc.fc.weight)
        self.assertIsNotNone(fc.fc.bias)

        # Default layer should be linear
        fc = networks.FCModule(self.c, self.c_out)
        self.assertRaises(AttributeError, getattr, fc, "activation")
        self.assertRaises(AttributeError, getattr, fc, "dropout")

    def test_no_dropout(self):
        fc = networks.FCModule(self.c, self.c_out)
        self.assertRaises(AttributeError, getattr, fc, "dropout")

    def test_no_activation(self):
        fc = networks.FCModule(self.c, self.c_out, activation=None)
        self.assertRaises(AttributeError, getattr, fc, "activation")
        self.assertRaises(AttributeError, getattr, fc, "dropout")


class TestFCChain(unittest.TestCase):
    def setUp(self):
        self.bs = 1
        self.c = 16
        self.c_out = 32
        self.in_data = th.ones(self.bs, self.c)

    def test_default(self):
        w = 32
        k = 3
        fc = networks.FCChain(self.c, depth=5, width=w)

        self.assertListEqual(list(fc.fc0.fc.weight.shape), [w, self.c])
        self.assertListEqual(list(fc.fc1.fc.weight.shape), [w, w])
        self.assertListEqual(list(fc.fc2.fc.weight.shape), [w, w])
        self.assertListEqual(list(fc.fc3.fc.weight.shape), [w, w])
        self.assertListEqual(list(fc.fc4.fc.weight.shape), [w, w])
        self.assertRaises(AttributeError, getattr, fc, "fc5")
        self.assertEqual(len(list(fc.children())), 5)

        self.assertEqual(len(list(fc.fc0.children())), 2)
        self.assertEqual(len(list(fc.fc1.children())), 2)
        self.assertEqual(len(list(fc.fc2.children())), 2)
        self.assertEqual(len(list(fc.fc3.children())), 2)
        self.assertEqual(len(list(fc.fc4.children())), 2)

        out_ = fc(self.in_data)
        self.assertListEqual(list(out_.shape), [self.bs, w])

    def test_output_activation(self):
        w = 32
        fc = networks.FCChain(self.c, depth=5, width=w)
        self.assertEqual(len(list(fc.fc4.children())), 2)

    def test_dropout(self):
        w = 32
        fc = networks.FCChain(self.c,
                              depth=5, width=w, dropout=0.2)
        self.assertEqual(len(list(fc.fc0.children())), 3)
        self.assertEqual(len(list(fc.fc1.children())), 3)
        self.assertEqual(len(list(fc.fc2.children())), 3)
        self.assertEqual(len(list(fc.fc3.children())), 3)
        self.assertEqual(len(list(fc.fc4.children())), 3)

        # self.assertRaises(AttributeError, getattr, fc.fc0, "dropout")
        self.assertIsNotNone(fc.fc0.dropout)
        self.assertIsNotNone(fc.fc1.dropout)
        self.assertIsNotNone(fc.fc2.dropout)
        self.assertIsNotNone(fc.fc3.dropout)
        self.assertIsNotNone(fc.fc4.dropout)
        # self.assertRaises(AttributeError, getattr, fc.fc4, "dropout")

    def test_variable_width(self):
        # Width should have 3-1 = 2 values
        self.assertRaises(AssertionError, networks.FCChain,
                          self.c, depth=3, width=[12])
        w = 32
        fc = networks.FCChain(self.c, depth=3, width=[12, 24, 32])
        # Check sizes and children length is correct
        self.assertListEqual(list(fc.fc0.fc.weight.shape), [12, self.c])
        self.assertListEqual(list(fc.fc1.fc.weight.shape), [24, 12])
        self.assertListEqual(list(fc.fc2.fc.weight.shape), [32, 24])
        self.assertRaises(AttributeError, getattr, fc, "fc3")
        self.assertEqual(len(list(fc.children())), 3)

    def test_variable_dropout(self):
        # dropout should have3 entries
        self.assertRaises(AssertionError, networks.FCChain,
                          self.c, depth=3, dropout=[0.2, 0.1])
        w = 32
        fc = networks.FCChain(self.c, depth=3, dropout=[0.2, 0.1, 0.05])
        print(fc)
        self.assertEqual(fc.fc0.dropout.p, 0.2)
        self.assertEqual(fc.fc1.dropout.p, 0.1)
        self.assertEqual(fc.fc2.dropout.p, 0.05)
        self.assertEqual(len(list(fc.children())), 3)


class TestUnet(unittest.TestCase):
    def setUp(self):
        self.bs = 1
        self.c = 3
        self.c_out = 4
        self.h = 128
        self.w = 128
        self.in_data = th.ones(self.bs, self.c, self.h, self.w)

    def test_default(self):
        unet = networks.UNet(self.c, self.c_out)
        print(unet)
        out = unet(self.in_data)


class TestKernelLookup(unittest.TestCase):
    def setUp(self):
        self.bs = 1
        self.c = 3
        self.c_out = 4
        self.h = 128
        self.w = 128
        self.in_data = th.ones(self.bs, self.c, self.h, self.w)
        self.kidx = th.ones(self.bs, self.c_out, self.h, self.w).int()

    def test_default(self):
        lk = ops.KernelLookup(self.c, 3, 64)
        print(lk)
        self.in_data.requires_grad=True
        self.in_data = self.in_data.cuda()
        self.kidx = self.kidx.cuda()
        lk.cuda()
        out = lk(self.in_data, self.kidx)
        loss = out.mean()
        loss.backward()
        print(lk.weights.grad.max())

# class TestDownConvChain(unittest.TestCase):
#     def setUp(self):
#         self.bs = 1
#         self.c = 3
#         self.c_out = 4
#         self.h = 32
#         self.w = 32
#         self.in_data = th.ones(self.bs, self.c, self.h, self.w)
#
#     def test_default(self):
#         w = 32
#         k = 3
#         cv = networks.DownConvChain(self.c, self.c_out, base_width=32, num_levels=3,
#                                     convs_per_level=2)
#
#         print(cv)
#
#         # self.assertListEqual(list(cv.conv0.conv.weight.shape), [w, self.c, k, k])
#         # self.assertListEqual(list(cv.conv1.conv.weight.shape), [w, w, k, k])
#         # self.assertListEqual(list(cv.conv2.conv.weight.shape), [w, w, k, k])
#         # self.assertListEqual(list(cv.conv3.conv.weight.shape), [w, w, k, k])
#         # self.assertListEqual(list(cv.conv4.conv.weight.shape), [self.c_out, w, k, k])
#         # self.assertRaises(AttributeError, getattr, cv, "conv5")
#         # self.assertEqual(len(list(cv.children())), 5)
#         #
#         # self.assertEqual(len(list(cv.conv0.children())), 2)
#         # self.assertEqual(len(list(cv.conv1.children())), 2)
#         # self.assertEqual(len(list(cv.conv2.children())), 2)
#         # self.assertEqual(len(list(cv.conv3.children())), 2)
#         # self.assertEqual(len(list(cv.conv4.children())), 1)
#         #
#         # out_ = cv(self.in_data)
#         # self.assertListEqual(list(out_.shape), [self.bs, self.c_out, self.h, self.w])
#
#     def test_variable_increase(self):
#         pass
#
#     def test_variable_num_convs(self):
#         pass
