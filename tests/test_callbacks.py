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
import pandas as pd

import torch as th

from ttools import callbacks
from ttools import database

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
        cb = callbacks.LoggingCallback(self.loggername, keys=["loss", "acc"], frequency=1)
        bwd = dict(loss=0.01, acc=0.99)
        cb.batch_end(None, bwd)
        log_contents = self.capture.getvalue()
        self.assertEqual(log_contents, "INFO, Step 1.1 | loss = 0.01 | acc = 0.99\n")

    def testNoneLogging(self):
        cb = callbacks.LoggingCallback(self.loggername, keys=["loss", "acc"], frequency=1)
        bwd = dict(loss=None, acc=0.99)
        cb.batch_end(None, bwd)
        log_contents = self.capture.getvalue()
        self.assertEqual(log_contents, "INFO, Step 1.1 | acc = 0.99\n")


class TestSQLLoggingCallback(unittest.TestCase):
    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        # self.assertRaises

    # TODO:
    # test wrong extension
    # test full train loop

    def testLogging(self):
        for session in range(2):  # simulate a stop and resume
            keys = ["loss", "accuracy"]
            val_keys = ["accuracy"]
            # new key added in second session
            # if session == 1:
            #     keys.append("some_new_key")
            cb = callbacks.SQLLoggingCallback(self.outdir, keys=keys,
                                              val_keys=val_keys, frequency=1)
            cb.training_start([])
            for epoch in range(2):
                cb.epoch_start(epoch)
                for i in range(3):
                    batch_data = None
                    train_step_data = {
                        "loss": i,
                        "accuracy": random.random(),
                        "some_new_key": 5,
                    }
                    cb.batch_start(i, batch_data)
                    cb.batch_end(batch_data, train_step_data)
                    # time.sleep(.1)
                cb.epoch_end()
                val_data = {
                    "accuracy": random.random(),
                }
                cb.validation_end(val_data)
            cb.training_end()

            if session == 0:
                del cb

        db = database.SQLiteDatabase(os.path.join(self.outdir, "logs.sqlite"))
        print(self.outdir)
        import ipdb; ipdb.set_trace()
        # print()
        # print(db.read_table("events"))
        # print(db.read_table("logs"))
        # print(db.read_table("val_logs"))
