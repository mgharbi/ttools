"""Defines the dataset I/O."""
import os

import torch as th
import numpy as np

import ttools


LOG = ttools.get_logger(__name__)


__all__ = ["Dataset"]


class Dataset(th.utils.data.Dataset):
    """A dummy dataset"""
    def __init__(self, data_path):
        pass

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        # return a gray and a white image
        in_ = np.ones((3, 16, 16)).astype(np.float32)*0.5
        target_ = np.ones((3, 16, 16)).astype(np.float32)
        return in_, target_
