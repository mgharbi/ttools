"""Extra custom callbacks used at train time."""
import os

import torch as th
import numpy as np

import ttools.callbacks as cb

__all__ = ["BasicCallback"]


class BasicCallback(cb.ImageDisplayCallback):
    """Simple callback that visualize images."""
    def visualized_image(self, batch, fwd_result):
        out = fwd_result.cpu()
        in_ = batch[0].cpu()
        ref = batch[1].cpu()
        diff = 2*(out-ref).abs()

        # tensor to visualize, concatenate images
        viz = th.clamp(th.cat([in_, out, ref, diff], 2), 0, 1)
        return viz

    def caption(self, batch, fwd_result):
        # write some informative caption into the visdom window
        return "input, output, ref, diff"
