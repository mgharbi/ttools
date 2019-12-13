"""Collections of loss functions."""
import torch as th
import numpy as np

import logging

from torchvision import models

LOG = logging.getLogger(__name__)


class PSNR(th.nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        self.mse = th.nn.MSELoss()
    def forward(self, out, ref):
        mse = self.mse(out, ref)
        return -10*th.log10(mse+1e-12)


class LedigPerceptualLoss(th.nn.Module):
    """Perceptual loss as defined by [Ledig2017] in 'Photo-Realistic Single
    Image Super-Resolution Using a Generative Adversarial Network'
    """
    def __init__(self, features="54"):
        super(LedigPerceptualLoss, self).__init__()
        LOG.warning("LedigPerceptualLoss is untested")
        if features not in ["54", "22"]:
            raise ValueError("Valid features extracted for Ledig's VGG loss are '54' and '22', got %s", features)
        self.feature_extractor = LedigPerceptualLoss._FeatureExtractor(features)
        self.mse = th.nn.MSELoss()

    def forward(self, out, ref):
        out_f = self.feature_extractor(out)
        ref_f = self.feature_extractor(ref)

        scores = []
        for idx, (o_f, r_f) in enumerate(zip(out_f, ref_f)):
            scores.append(self.mse(o_f, r_f))
        return sum(scores)

    class _FeatureExtractor(th.nn.Module):
        def __init__(self, features):
            super(LedigPerceptualLoss._FeatureExtractor, self).__init__()
            vgg_pretrained = models.vgg19(pretrained=True).features
            if features == "54":
                breakpoints = [0, 35]  # 4th conv before pool5, pre-activation
            elif features == "22":
                breakpoints = [0, 8]  # 2nd conv before pool2, pre activation
            else:
                raise ValueError("Incorrect features to exactra '%s', should be '54' or '22'.")

            for i, b in enumerate(breakpoints[:-1]):
                ops = th.nn.Sequential()
                for idx in range(b, breakpoints[i+1]):
                    op = vgg_pretrained[idx]
                    ops.add_module(str(idx), op)
                self.add_module("group{}".format(i), ops)

            for p in self.parameters():
                p.requires_grad = False

            self.register_buffer("shift", th.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("scale", th.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


        def forward(self, x):
            feats = []
            x = (x-self.shift) / self.scale
            for m in self.children():
                x = m(x)
                feats.append(x)
            return feats


class TotalVariation(th.nn.Module):
    def __init__(self):
        LOG.warning("Total Variation loss is untested")
        super(TotalVariation, self).__init__()

    def _dx(self, tensor):
        return tensor[..., 1:] - tensor[..., :-1]

    def _dy(self, tensor):
        return tensor[..., 1:, :] - tensor[..., :-1, :]

    def forward(self, tensor):
        return th.abs(self._dx(tensor)).mean() + th.abs(self._dy(tensor)).mean()


class LPIPS(th.nn.Module):
    def __init__(self, pretrained=True, normalize=True, pre_relu=True): 
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(LPIPS, self).__init__()
         # VGG using perceptually-learned weights (LPIPS metric)
        self.normalize = normalize
        self.pretrained = pretrained
        LOG.warning("LPIPS is untested")

        self.feature_extractor = LPIPS._FeatureExtractor(pretrained, pre_relu)

    def _l2_normalize_features(self, x, eps=1e-10):
        nrm = th.sqrt(th.sum(x*x, dim=1, keepdim=True))
        return x / (nrm + eps)

    def forward(self, pred, target):
        """Compare VGG features of two inputs."""

        # Get VGG features
        pred = self.feature_extractor(pred)
        target = self.feature_extractor(target)

        # L2 normalize features
        if self.normalize:
            pred = [self._l2_normalize_features(f) for f in pred]
            target = [self._l2_normalize_features(f) for f in target]

        # TODO(mgharbi) Apply Richard's linear weights?

        if self.normalize:
            diffs = [th.sum((p-t)**2, 1) for (p, t) in zip(pred, target)]
        else:
            # mean instead of sum to avoid super high range
            diffs = [th.mean((p-t)**2, 1) for (p, t) in zip(pred, target)]

        # Spatial average
        diffs = [diff.mean([1, 2]) for diff in diffs]

        return sum(diffs).mean(0)

    class _FeatureExtractor(th.nn.Module):
        def __init__(self, pretrained, pre_relu):
            super(LPIPS._FeatureExtractor, self).__init__()
            vgg_pretrained = models.vgg16(pretrained=pretrained).features

            self.breakpoints = [0, 4, 9, 16, 23, 30]
            if pre_relu:
                for i, _ in enumerate(self.breakpoints[1:]):
                    self.breakpoints[i+1] -= 1

            # Split at the maxpools
            for i, b in enumerate(self.breakpoints[:-1]):
                ops = th.nn.Sequential()
                for idx in range(b, self.breakpoints[i+1]):
                    op = vgg_pretrained[idx]
                    ops.add_module(str(idx), op)
                # print(ops)
                self.add_module("group{}".format(i), ops)

            # No gradients
            for p in self.parameters():
                p.requires_grad = False

            # Torchvision's normalization: <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>
            self.register_buffer("shift", th.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("scale", th.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            feats = []
            x = (x - self.shift) / self.scale
            for idx in range(len(self.breakpoints)-1):
                m = getattr(self, "group{}".format(idx))
                x = m(x)
                feats.append(x)
            return feats


class ELPIPS(th.nn.Module):
    def __init__(self, pretrained=True, normalize=True, pre_relu=True, max_shift=16,
                 nsamples=1):
        """Ensemble of LPIPS."""
        super(ELPIPS, self).__init__()
        self.max_shift = max_shift
        self.ploss = LPIPS(pretrained=pretrained, normalize=normalize, pre_relu=pre_relu)
        self.nsamples = nsamples
        LOG.warning("E-LPIPS is untested")

    def sample_xform(self):
        shift = np.random.randint(0, self.max_shift, size=(2,))
        color_scale = th.rand(size=(3,))
        # TODO(mgharbi): Limit anistropic scaling
        # TODO(mgharbi): issue with lowpass in 
        scale = th.pow(2.0, th.rand(size=(2,))*2.0 - 2.0)
        scale[1] = scale[0]
        channel_perm = np.random.permutation(3)
        transpose = np.random.choice([True, False])
        fliplr = np.random.choice([True, False])
        flipud = np.random.choice([True, False])
        return dict(shift=shift, color_scale=color_scale, scale=scale,
                    channel_perm=channel_perm, transpose=transpose,
                    fliplr=fliplr, flipud=flipud)

    def xform(self, im, params):
        scale = params["scale"]
        im = th.nn.functional.interpolate(im, scale_factor=scale,
                                          mode="bilinear")

        shift = params["shift"]
        im = im[..., shift[0]:, shift[1]:]

        # flip
        if params["fliplr"]:
            im = th.flip(im, (3,))

        if params["flipud"]:
            im = th.flip(im, (2,))

        # transpose
        if params["transpose"]:
            im = im.permute(0, 1, 3, 2)

        # color permutation
        channel_perm = params["channel_perm"]
        im = im[:, channel_perm]

        color_scale = params["color_scale"].view(1, 3, 1, 1).to(im.device)
        im = im * color_scale

        return im

    def forward(self, a, b):
        losses = []
        for smp in range(self.nsamples):
            p = self.sample_xform()
            xa = self.xform(a, p)
            xb = self.xform(b, p)
            losses.append(self.ploss(xa, xb))
        losses = th.stack(losses)
        return losses.mean()
