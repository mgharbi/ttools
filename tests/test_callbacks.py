"""Test the built-in callbacks."""

import os
import io
import shutil
import tempfile
import unittest
import time
import logging

import torch as th

from ttools.callbacks import *

class TestLoggingCallback(unittest.TestCase):
    def _get_log_capture(self, name):
        capture = io.StringIO()
        ch = logging.StreamHandler(capture)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s, %(message)s')
        ch.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.addHandler(ch)
        return capture

    def testBatchLogging(self):
        loggername = "someFancyName"
        cb = LoggingCallback(loggername, keys=["loss", "acc"], frequency=1)

        capture = self._get_log_capture(loggername)

        bwd = dict(loss=0.01, acc=0.99)
        cb.batch_end(None, None, bwd)


        log_contents = capture.getvalue()
        capture.close()

        print(log_contents)
        self.assertEqual(log_contents, "INFO, Step 1.1 | loss = 0.01 | acc = 0.99\n")
