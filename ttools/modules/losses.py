"""Collections of loss functions."""
import torch as th

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
    def __init__(self): 
        super(LPIPS, self).__init__()
         # VGG using perceptually-learned weights (LPIPS metric)

        self.feature_extractor = LPIPS._FeatureExtractor()

    def _normalize(self, x):
        # Network assumes inputs are in [-1, 1]
        return 2.0 * x  - 1.0

    def forward(self, pred, target):
        """ """

        # Get VGG features
        pred = self.feature_extractor(self._normalize(pred))
        target = self.feature_extractor(self._normalize(target))

        # TODO: L2 normalize features?

        # TODO(mgharbi) Apply Richard's linear weights?

        diffs = [th.nn.functional.mse_loss(p, t) for (p, t) in zip(pred, target)]

        return sum(diffs)

    class _FeatureExtractor(th.nn.Module):
        def __init__(self):
            super(LPIPS._FeatureExtractor, self).__init__()
            vgg_pretrained = models.vgg16(pretrained=True).features

            breakpoints = [0, 4, 9, 16, 23, 30]

            # Split at the maxpools
            for i, b in enumerate(breakpoints[:-1]):
                ops = th.nn.Sequential()
                for idx in range(b, breakpoints[i+1]):
                    op = vgg_pretrained[idx]
                    ops.add_module(str(idx), op)
                self.add_module("group{}".format(i), ops)

            # No gradients
            for p in self.parameters():
                p.requires_grad = False

            self.register_buffer("shift", th.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("scale", th.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Richard's
        # self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        # self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

        def forward(self, x):
            feats = []
            x = (x-self.shift) / self.scale
            for m in self.children():
                x = m(x)
                feats.append(x)
            return feats


# class vgg16(torch.nn.Module):
#     def __init__(self, requires_grad=False, pretrained=True):
#         super(vgg16, self).__init__()
#         vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         self.N_slices = 5
#         for x in range(4):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(4, 9):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(9, 16):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(16, 23):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(23, 30):
#             self.slice5.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False
#
#     def forward(self, X):
#         h = self.slice1(X)
#         h_relu1_2 = h
#         h = self.slice2(h)
#         h_relu2_2 = h
#         h = self.slice3(h)
#         h_relu3_3 = h
#         h = self.slice4(h)
#         h_relu4_3 = h
#         h = self.slice5(h)
#         h_relu5_3 = h
#         vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
#         out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
#
#         return out
