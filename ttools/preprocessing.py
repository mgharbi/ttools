"""Utilities to preprocess data."""
from . import get_logger

import numpy as np
import torch as th

__all__ = ["extract_tiles"]

LOG = get_logger(__name__)


def extract_tiles(im, tile_size=128, tile_stride=None,
                  drop_last=True, align=None):
    """Generator that extracts tiles from an image.

    Args:
        im(np.array with size [h, w, ...] or th.Tensor with
            size [..., h, w]: the image.
        tile_size(int): size of the square tiles in pixel.
        tile_stride(int or None): if None, the tiles are
            non-overlapping, otherwise stride between tiles.
        drop_last(bool): if True, drop last tile if tile_size
            does not divide the image dimension. Otherwise, shift
            the tile to make sure it is in bounds
            (creates an overlap for the last tile)
        align(int or None): align to a multiple of this value.
    Yields:
        tile(np.arrray): the subimage extracted
        coord(2-tuple): the (y, x) coordinate of the tile's top left corner.
    """
    if tile_stride is None:
        tile_stride = tile_size

    extractor = None
    if isinstance(im, np.ndarray):
        extractor = lambda x, y: im[y:y+tile_size, x:x+tile_size]
        h, w = im.shape[:2]
    elif isinstance(im, th.Tensor):
        extractor = lambda x, y: im[..., y:y+tile_size, x:x+tile_size]
        h, w = im.shape[-2:]
    else:
        LOG.error("Unknown input type %s", im.__class__.__name__)
        raise RuntimeError("Unknown input type %s" % im.__class__.__name__)

    if len(im.shape) < 3:
        LOG.error("Incorrect image shape, expected at least 3 dimensions.")
        raise ValueError("Incorrect image shape")

    for y in range(0, h, tile_stride):
        if y+tile_size > h:
            if drop_last:
                break
            y = h - tile_size
            if align is not None:
                y -= y % align
            if y < 0:  # Image is smaller than a tile
                break

        for x in range(0, w, tile_stride):
            if x+tile_size > w:
                if drop_last:
                    break
                x = w - tile_size
                if align is not None:
                    x -= x % align
                if x < 0:  # Image is smaller than a tile
                    break
            tile = extractor(x, y)
            yield tile, y, x
