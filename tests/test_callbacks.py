"""Test the built-in callbacks."""

import os
import io
import shutil
import tempfile
import random
import string
import time
import unittest
import logging

import torch as th

from ttools.callbacks import *


class TestLoggingCallback(unittest.TestCase):
    def setUp(self):
        self.loggername = ''.join(
            [random.choice(string.ascii_letters + string.digits) for n in
             range(32)])
        capture = io.StringIO()
        ch = logging.StreamHandler(capture)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s, %(message)s')
        ch.setFormatter(formatter)
        logger = logging.getLogger(self.loggername)
        logger.addHandler(ch)
        self.capture = capture

    def tearDown(self):
        self.capture.close()

    def testBatchLogging(self):
        cb = LoggingCallback(self.loggername, keys=["loss", "acc"], frequency=1)
        bwd = dict(loss=0.01, acc=0.99)
        cb.batch_end(None, None, bwd)
        log_contents = self.capture.getvalue()
        self.assertEqual(log_contents, "INFO, Step 1.1 | loss = 0.01 | acc = 0.99\n")

    def testNoneLogging(self):
        cb = LoggingCallback(self.loggername, keys=["loss", "acc"], frequency=1)
        bwd = dict(loss=None, acc=0.99)
        cb.batch_end(None, None, bwd)
        log_contents = self.capture.getvalue()
        self.assertEqual(log_contents, "INFO, Step 1.1 | acc = 0.99\n")
