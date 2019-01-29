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
    crop = (src_sz[-2:]-tgt_sz[-2:])
    assert (np.mod(crop, 2) == 0).all(), "crop like sizes should be even"
    crop //= 2
    if (crop > 0).any():
        return src[..., crop[0]:src_sz[-2]-crop[0], crop[1]:src_sz[-1]-crop[1]]
    else:
        return src


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
#
#
# class ImageGradients(nn.Module):
#   def __init__(self, c_in):
#     super(ImageGradients, self).__init__()
#     self.dx = nn.Conv2d(c_in, c_in, [3, 3], padding=1, bias=False, groups=c_in)
#     self.dy = nn.Conv2d(c_in, c_in, [3, 3], padding=1, bias=False, groups=c_in)
#
#     self.dx.weight.requires_grad = False
#     self.dy.weight.requires_grad = False
#
#     self.dx.weight.data.zero_()
#     self.dx.weight.data[:, :, 0, 0]  = -1
#     self.dx.weight.data[:, :, 0, 2]  = 1
#     self.dx.weight.data[:, :, 1, 0]  = -2
#     self.dx.weight.data[:, :, 1, 2]  = 2
#     self.dx.weight.data[:, :, 2, 0]  = -1
#     self.dx.weight.data[:, :, 2, 2]  = 1
#
#     self.dy.weight.data.zero_()
#     self.dy.weight.data[:, :, 0, 0]  = -1
#     self.dy.weight.data[:, :, 2, 0]  = 1
#     self.dy.weight.data[:, :, 0, 1]  = -2
#     self.dy.weight.data[:, :, 2, 1]  = 2
#     self.dy.weight.data[:, :, 0, 2]  = -1
#     self.dy.weight.data[:, :, 2, 2]  = 1
#
#   def forward(self, im):
#     return th.cat([self.dx(im), self.dy(im)], 1)
