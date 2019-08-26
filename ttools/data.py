"""Utilities to manage datasets."""

import logging
import os
import shutil

import torch as th

LOG = logging.getLogger(__name__)


class CachedDataset(th.utils.data.Dataset):
    """A dataset baseclass that downloads and cache data locally as they are requested from a remote path.

    Args:
        cache_dir(str): the root path where the data will be saved
        root(str): the root path of the remote data. This prefix will be removed to re-root the data at cache_dir 
    """

    def __init__(self, cache_dir, root):
        self.cache_dir = os.path.normpath(cache_dir)
        self.root = os.path.normpath(root)

    def path(self, path):
        """Converts a path from the 'remote' to the 'local' path, copying the file
        if it not available locally."""

        splits = path.split(self.root+os.path.sep)
        if len(splits) !=2:
            raise ValueError("Wrong path provided in CachedDataset")

        suffix = splits[-1]
        newpath = os.path.join(self.cache_dir, suffix)

        if not os.path.exists(newpath):
            # Copy file to local path
            # TODO(beware of multithreaded dataloader, maybe we need to have a locking mechanism on the file)
            dirname = os.path.dirname(newpath)
            os.makedirs(dirname)
            LOG.debug("caching locally %s -> %s", path, newpath)
            shutil.copy(path, newpath)

        # Return local path
        return newpath
