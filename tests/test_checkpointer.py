import os
import shutil
import tempfile
import unittest
import time

import torch as th

from ttools import Checkpointer
from ttools.callbacks import CheckpointingCallback


class TestCheckpointer(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_save_and_load_basic(self):
        chkpt = Checkpointer(self.root)
        chkpt.save("first")
        assert os.path.exists(os.path.join(
            self.root, "first" + Checkpointer.EXTENSION))

        res = chkpt.load("first")
        assert len(res) == 2
        assert res[0] is None
        assert res[1] is None

    def test_load_no_folder(self):
        chkpt = Checkpointer("a_folder_that_does_not_exists")
        res = chkpt.load_latest()
        assert res[0] is None
        with self.assertRaises(FileNotFoundError) as ctx:
            res = chkpt.load("first")

    def test_save_and_load_meta(self):
        meta = {"somekey": [1, 2, 3]}
        chkpt = Checkpointer(self.root, meta=meta)
        chkpt.save("file")
        res = chkpt.load("file")

        assert "somekey" in res[1]
        for i, d in enumerate(res[1]["somekey"]):
            assert d == meta["somekey"][i]

        meta = Checkpointer.load_meta(self.root)
        assert "somekey" in meta
        for i, d in enumerate(meta["somekey"]):
            assert d == meta["somekey"][i]

    def test_save_and_load_model(self):
        model = th.nn.Conv2d(1, 1, 1)
        chkpt = Checkpointer(self.root, model=model)

        # Create a different model
        model2 = th.nn.Conv2d(1, 1, 1)
        model2.weight.data = model.weight.data*2
        model2.bias.data = model.bias.data*2
        chkpt2 = Checkpointer(self.root, model=model2)

        # Save model 1 and load its params with model2
        chkpt.save("first")
        res = chkpt2.load("first")

        assert model.weight.data == model2.weight.data
        assert model.bias.data == model2.bias.data

    def test_save_and_load_optimizer(self):
        model = th.nn.Conv2d(1, 1, 1)
        opt = th.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
        chkpt = Checkpointer(self.root, model=model, optimizers=[opt])

        # Create a different model with its own optimizer
        model2 = th.nn.Conv2d(1, 1, 1)
        model2.weight.data = model.weight.data*2
        model2.bias.data = model.bias.data*2
        opt2 = th.optim.Adam(model2.parameters(), lr=1e-3, eps=1e-8)
        chkpt2 = Checkpointer(self.root, model=model2, optimizers=[opt2])

        # Save model 1 and load its params with model2
        chkpt.save("first")
        chkpt2.load("first")

        assert model.weight.data == model2.weight.data
        assert model.bias.data == model2.bias.data

        assert opt.state_dict()["param_groups"][0]["lr"] == opt2.state_dict()[
            "param_groups"][0]["lr"]
        assert opt.state_dict()["param_groups"][0]["eps"] == opt2.state_dict()[
            "param_groups"][0]["eps"]

    def test_save_gpu_and_load_cpu(self):
        # TODO(mgharbi)
        pass


class TestCheckpointingCallback(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.root2 = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)
        shutil.rmtree(self.root2)

    def testPeriodicCheckpoints(self):
        chkpt = Checkpointer(self.root)
        interval = 0.5  # checkpoint every x seconds
        cb = CheckpointingCallback(chkpt, max_files=10, max_epochs=None,
                                   interval=interval)
        chkpts = chkpt.sorted_checkpoints()
        self.assertFalse(chkpts)  # check is empty

        # Two batches in the interval
        time.sleep(2*interval)
        cb.batch_end(None, None, None)
        time.sleep(2*interval)
        cb.batch_end(None, None, None)

        chkpts = chkpt.sorted_checkpoints()
        self.assertEqual(len(chkpts), 2)

    def testDisablePeriodicCheckpoints(self):
        chkpt = Checkpointer(self.root)
        cb = CheckpointingCallback(chkpt, max_files=10, max_epochs=None,
                                   interval=None)
        chkpts = chkpt.sorted_checkpoints()
        self.assertFalse(chkpts)  # check is empty

        # Two batches in the interval
        time.sleep(1)
        cb.batch_end(None, None, None)
        time.sleep(1)
        cb.batch_end(None, None, None)

        # There should be no periodic checkpoints
        chkpts = chkpt.sorted_checkpoints()
        self.assertFalse(chkpts)

    def testMaxCheckpoints(self):
        chkpt = Checkpointer(self.root)
        chkpt2 = Checkpointer(self.root2)
        interval = 0.5  # checkpoint every x seconds
        cb = CheckpointingCallback(chkpt, max_files=10, max_epochs=None,
                                   interval=interval)
        cb2 = CheckpointingCallback(chkpt2, max_files=1, max_epochs=None,
                                   interval=interval)
        chkpts = chkpt.sorted_checkpoints()
        self.assertFalse(chkpts)  # check is empty

        # Three batches in the interval
        time.sleep(2*interval)
        cb.batch_end(None, None, None)
        cb2.batch_end(None, None, None)
        time.sleep(2*interval)
        cb.batch_end(None, None, None)
        cb2.batch_end(None, None, None)
        time.sleep(2*interval)
        cb.batch_end(None, None, None)
        cb2.batch_end(None, None, None)

        # Make sure we have the right count in the comparison chkpt
        chkpts = chkpt.sorted_checkpoints()
        self.assertEqual(len(chkpts), 3)

        # Make sure 2 is properly capped
        chkpts = chkpt2.sorted_checkpoints()
        self.assertEqual(len(chkpts), 1)


    def testEpochCheckpoints(self):
        chkpt = Checkpointer(self.root)
        interval = 0.5  # checkpoint every x seconds
        cb = CheckpointingCallback(chkpt, max_files=1, max_epochs=None,
                                   interval=interval)
        chkpts = chkpt.sorted_checkpoints()
        self.assertFalse(chkpts)  # check is empty

        # A few epoch have passed
        for epoch in range(4):
            cb.epoch_start(epoch)
            cb.epoch_end()

        time.sleep(1)

        # Check we have the right checkpoint count
        chkpts = chkpt.sorted_checkpoints()
        self.assertEqual(len(chkpts), 4)


    def testCappedEpochCheckpoints(self):
        chkpt = Checkpointer(self.root)
        chkpt2 = Checkpointer(self.root2)
        interval = 0.5  # checkpoint every x seconds
        cb = CheckpointingCallback(chkpt, max_files=1, max_epochs=None,
                                   interval=interval)
        interval = 0.5  # checkpoint every x seconds
        cb2 = CheckpointingCallback(chkpt2, max_files=1, max_epochs=3,
                                   interval=interval)

        chkpts = chkpt.sorted_checkpoints()
        self.assertFalse(chkpts)  # check is empty

        chkpts = chkpt2.sorted_checkpoints()
        self.assertFalse(chkpts)  # check is empty

        # A few epoch have passed
        for epoch in range(8):
            cb.epoch_start(epoch)
            cb.epoch_end()
            cb2.epoch_start(epoch)
            cb2.epoch_end()

        time.sleep(1)

        # Check we have the right checkpoint count
        chkpts = chkpt.sorted_checkpoints()
        self.assertEqual(len(chkpts), 8)

        chkpts = chkpt2.sorted_checkpoints()
        self.assertEqual(len(chkpts), 3)
