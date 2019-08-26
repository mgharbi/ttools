"""Test the Trainer module."""

import os
import unittest
import tempfile
import time

import numpy as np
import torch.utils.data

from ttools.data import CachedDataset


class TestCachedDataset(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.cache = os.path.join(tmp, "cache")
        self.root = os.path.join(tmp, "root")
        self.path = os.path.join(self.root, "subdir", "file.txt")
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, 'w') as fid:
            fid.write("sometext\n")

    def tearDown(self):
        pass

    def test_path(self):
        # cache = tmpfile.mkd
        dset = CachedDataset(self.cache, self.root)
        path = dset.path(self.path)
        self.assertEqual(path, os.path.join(self.cache,  "subdir", "file.txt"))

    def test_wrong_prefix(self):
        dset = CachedDataset(self.cache, self.root)

        with self.assertRaises(ValueError):
            path = dset.path(os.path.join(self.root+"typo", "subdir", "file.txt"))

    def test_copies(self):
        dset = CachedDataset(self.cache, self.root)

        newpath = dset.path(self.path)
        self.assertTrue(os.path.exists(self.path))
        self.assertTrue(os.path.exists(newpath))

    def test_copies_only_once(self):
        dset = CachedDataset(self.cache, self.root)

        newpath = dset.path(self.path)
        self.assertTrue(os.path.exists(self.path))
        self.assertTrue(os.path.exists(newpath))
        timea = os.path.getmtime(newpath)

        time.sleep(1)

        newpath = dset.path(self.path)
        self.assertTrue(os.path.exists(newpath))
        timeb = os.path.getmtime(newpath)

        self.assertEqual(timea, timeb)
