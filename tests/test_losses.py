import unittest

import torch as th
import torch.nn as nn

from ttools.modules import losses


class TestLPIPS(unittest.TestCase):
    def test_lpips(self):
        bs, c, h, w = 1, 3, 128, 128
        th.manual_seed(0)
        pred = th.rand(bs, c, h, w)
        tgt = pred.clone()

        loss_fn = losses.LPIPS()

        if th.cuda.is_available():
            loss_fn.cuda()
            pred = pred.cuda()
            tgt = tgt.cuda()

        loss = loss_fn(pred, tgt)

        self.assertAlmostEqual(loss.item(), 0.0)

        # Make sure loss grows with distance
        prev = 0.0
        eps = 0.01
        for i in range(1, 10):
            tgt = tgt + i*eps
            loss = loss_fn(pred, tgt).item()
            self.assertGreaterEqual(loss, prev)
            prev = loss

    def test_optimize(self):
        bs, c, h, w = 1, 3, 128, 128
        th.manual_seed(0)
        device = "cpu"
        if th.cuda.is_available():
            device = "cuda"

        pred = th.rand(bs, c, h, w, device=device)
        noise = th.rand(bs, c, h, w, requires_grad=True, device=device)

        loss_fn = losses.LPIPS()

        if device == "cuda":
            loss_fn.cuda()

        opt = th.optim.Adam([noise], lr=1e-2)

        for step in range(1000):
            tgt = pred + 0.1*noise

            loss = loss_fn(pred, tgt)
            mse = th.nn.functional.mse_loss(pred, tgt).item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            print("step", step, "loss", loss.item(), "mse", mse)

        self.assertAlmostEqual(loss.item(), 0.0, 2)
        self.assertAlmostEqual(mse, 0.0, 3)


class TestELPIPS(unittest.TestCase):
    def test_elpips(self):
        bs, c, h, w = 1, 3, 128, 128
        th.manual_seed(0)
        pred = th.rand(bs, c, h, w)
        tgt = pred.clone()

        loss_fn = losses.ELPIPS()

        if th.cuda.is_available():
            loss_fn.cuda()
            pred = pred.cuda()
            tgt = tgt.cuda()

        loss = loss_fn(pred, tgt)
        print("loss", loss.item())

        self.assertAlmostEqual(loss.item(), 0.0)
