"""Common image operators."""
import numpy as np
import torch as th


def crop_like(src, tgt):
    """Crop a source image to match the spatial dimensions of a target.

    Assumes sizes are even.

    Args:
        src (th.Tensor or np.ndarray): image to be cropped
        tgt (th.Tensor or np.ndarray): reference image
    """
    src_sz = np.array(src.shape)
    tgt_sz = np.array(tgt.shape)

    # Assumes the spatial dimensions are the last two
    delta = (src_sz[2:4]-tgt_sz[2:4])
    # delta = (src_sz[-2:]-tgt_sz[-2:])
    crop = np.maximum(delta // 2, 0)  # no negative crop
    crop2 = delta - crop

    if (crop > 0).any() or (crop2 > 0).any():
        # NOTE: convert to ints to enable static slicing in ONNX conversion
        src_sz = [int(x) for x in src_sz]
        crop = [int(x) for x in crop]
        crop2 = [int(x) for x in crop2]
        return src[..., crop[0]:src_sz[-2]-crop2[0],
                   crop[1]:src_sz[-1]-crop2[1]]
    else:
        return src


class RGB2YCbCr(th.nn.Module):
    def __init__(self):
        super(RGB2YCbCr, self).__init__()

        # ACR numbers
        kr = 0.299
        kg = 0.587
        kb = 0.114

        kInvSum = 1.0 / (kr + kg + kb)

        kr *= kInvSum
        kg *= kInvSum
        kb *= kInvSum

        krScale = 0.5 / (kr - 1.0)
        kbScale = 0.5 / (kb - 1.0)

        rgb2ycc = th.Tensor(3, 3)
        rgb2ycc[0, 0] = kr
        rgb2ycc[0, 1] = kg
        rgb2ycc[0, 2] = kb

        rgb2ycc[1, 0] = (kr - 1.0) * krScale
        rgb2ycc[1, 1] = (kg	  ) * krScale
        rgb2ycc[1, 2] = (kb	  ) * krScale

        rgb2ycc[2, 0] = (kr	  ) * kbScale
        rgb2ycc[2, 1] = (kg	  ) * kbScale
        rgb2ycc[2, 2] = (kb - 1.0) * kbScale

        self.register_buffer("rgb2ycc", rgb2ycc)

    def forward(self, x):
        y = th.tensordot(x, self.rgb2ycc.to(x.device), ([-3,], [1,])).permute(0, 3, 1, 2)
        return y


class ImageGradients(th.nn.Module):
  def __init__(self, c_in):
    super(ImageGradients, self).__init__()
    self.dx = th.nn.Conv2d(c_in, c_in, [3, 3], padding=1, bias=False, groups=c_in)
    self.dy = th.nn.Conv2d(c_in, c_in, [3, 3], padding=1, bias=False, groups=c_in)

    self.dx.weight.requires_grad = False
    self.dy.weight.requires_grad = False

    # Sobel filters

    self.dx.weight.data.zero_()
    self.dx.weight.data[:, :, 0, 0]  = -1
    self.dx.weight.data[:, :, 0, 2]  = 1
    self.dx.weight.data[:, :, 1, 0]  = -2
    self.dx.weight.data[:, :, 1, 2]  = 2
    self.dx.weight.data[:, :, 2, 0]  = -1
    self.dx.weight.data[:, :, 2, 2]  = 1

    self.dy.weight.data.zero_()
    self.dy.weight.data[:, :, 0, 0]  = -1
    self.dy.weight.data[:, :, 2, 0]  = 1
    self.dy.weight.data[:, :, 0, 1]  = -2
    self.dy.weight.data[:, :, 2, 1]  = 2
    self.dy.weight.data[:, :, 0, 2]  = -1
    self.dy.weight.data[:, :, 2, 2]  = 1

  def forward(self, im):
    return th.cat([self.dx(im), self.dy(im)], 1)


class GaussianBlur(th.nn.Module):
    def __init__(self, sigma, channels=3):
        super(GaussianBlur, self).__init__()
        ksize = int(np.ceil(4*sigma))
        self.ksize = ksize

        self.channels = channels

        kernel = th.pow(th.arange(-ksize, ksize+1).float(), 2) / (2*sigma*sigma)
        kernel = th.exp(-kernel)
        kernel /= kernel.sum()
        self.register_buffer("kernel", kernel.view(1, 2*ksize+1).repeat(channels, 1))

    def forward(self, x):
        c, h, w = x.shape[-3:]
        assert c == self.channels
        ksize = self.ksize
        # Gaussian blur
        lp = th.nn.functional.pad(x, (ksize, ksize, ksize, ksize), mode="reflect")

        # blur y
        lp = th.nn.functional.conv2d(lp, self.kernel.view(self.channels, 1, 2*ksize+1, 1), groups=self.channels)
        # blur x
        lp = th.nn.functional.conv2d(lp, self.kernel.view(self.channels, 1, 1, 2*ksize+1), groups=self.channels)
        return lp

# class MedianFilter(nn.Module):
#   def __init__(self, ksize=3):
#     super(MedianFilter, self).__init__()
#     self.ksize = ksize
#
#   def forward(self, x):
#     k = self.ksize
#     assert(len(x.shape) == 4)
#     x = F.pad(x, [k//2, k//2, k//2, k//2])
#     x = x.unfold(2, k, 1).unfold(3, k, 1)
#     x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
#     return x



class BilinearUpsampler(th.nn.Module):
  def __init__(self, scale=2, channels=1):
    super(BilinearUpsampler, self).__init__()

    ksize = 2*scale
    total_pad = ksize-scale//2
    if scale % 2 == 1:
      ksize += 1

    self.pad = th.nn.ReplicationPad2d((1, 1, 1, 1))
    self.us_x = th.nn.ConvTranspose2d(channels, channels, (1, ksize),
                                      stride=(1, scale),
                                      padding=(0, total_pad), groups=channels, bias=False)
    self.us_y = th.nn.ConvTranspose2d(channels, channels, (ksize, 1),
                                      stride=(scale, 1),
                                      padding=(total_pad, 0), groups=channels, bias=False)

    k_idx = th.arange(0, ksize) + 0.5
    k_coord = k_idx / scale - ksize*0.5 / scale

    # linear kernel
    k_weight = th.clamp(1.0-th.abs(k_coord), min=0)

    for c in range(channels):
      self.us_x.weight.data[c, 0, 0, :].copy_(k_weight)
      self.us_y.weight.data[c, 0, :, 0].copy_(k_weight)

    for p in self.parameters():
      p.requires_grad = False

  def forward(self, x):
    x = self.pad(x)
    x = self.us_x(x)
    x = self.us_y(x)
    return x


class BicubicUpsampler(th.nn.Module):
  def __init__(self, scale=2, channels=1):
    super(BicubicUpsampler, self).__init__()

    ksize = 2*scale * 2
    total_pad = ksize - scale//2
    if scale % 2 == 1:
      ksize += 1

    self.pad = th.nn.ReplicationPad2d((2, 2, 2, 2))
    self.us_x = th.nn.ConvTranspose2d(channels, channels, (1, ksize),
                                      stride=(1, scale),
                                      padding=(0, total_pad),
                                      groups=channels, bias=False)
    self.us_y = th.nn.ConvTranspose2d(channels, channels, (ksize, 1),
                                      stride=(scale, 1),
                                      padding=(total_pad, 0),
                                      groups=channels, bias=False)

    k_idx = th.arange(0, ksize) + 0.5
    k_coord = k_idx / scale - ksize*0.5 / scale

    # cubic kernel
    absx = th.abs(k_coord)
    absx2 = absx.pow(2)
    absx3 = absx.pow(3)
    k_weight = th.zeros(ksize)
    k_weight += (-0.5*absx3 + 2.5*absx2 - 4*absx + 2.0) * ((absx > 1.0) & (absx < 2.0))
    k_weight += (1.5*absx3 - 2.5*absx2 + 1.0) * (absx <= 1.0)

    for c in range(channels):
      self.us_x.weight.data[c, 0, 0, :].copy_(k_weight)
      self.us_y.weight.data[c, 0, :, 0].copy_(k_weight)

    for p in self.parameters():
      p.requires_grad = False

  def forward(self, x):
    x = self.pad(x)
    x = self.us_x(x)
    x = self.us_y(x)
    return x
