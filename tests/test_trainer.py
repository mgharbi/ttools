"""Test the Trainer module."""

import unittest

import numpy as np
import torch.utils.data

from ttools import ModelInterface
from ttools import Trainer
from ttools import Callback


class DummyInterface(ModelInterface):
    """Simple model interface used for testing."""
    def __init__(self):
        super(DummyInterface, self).__init__()
        self.fwd = 0
        self.bwd = 0

        self.init_val = 0
        self.update_val = 0

    def training_step(self, batch):
        self.fwd += 1
        self.bwd += 1

    def init_validation(self):
        self.init_val += 1

    def validation_step(self, batch, running):
        self.update_val += 1


class DummyCallback(Callback):
    """Simple dummy callback used for testing."""

    def __init__(self):
        super(DummyCallback, self).__init__()
        self.batch_ends = 0
        self.batch_starts = 0
        self.epoch_ends = 0
        self.epoch_starts = 0
        self.val_starts = 0
        self.val_ends = 0
        self.val_steps = 0

    def batch_start(self, *args):
        super(DummyCallback, self).batch_start(*args)
        self.batch_starts += 1

    def batch_end(self, *args):
        super(DummyCallback, self).batch_end(*args)
        self.batch_ends += 1

    def epoch_start(self, *args):
        super(DummyCallback, self).epoch_start(*args)
        self.epoch_starts += 1

    def epoch_end(self, *args):
        super(DummyCallback, self).epoch_end(*args)
        self.epoch_ends += 1

    def validation_start(self, *args):
        super(DummyCallback, self).validation_start(*args)
        self.val_starts += 1

    def val_batch_end(self, *args):
        super(DummyCallback, self).val_batch_end(*args)
        self.val_steps += 1

    def validation_end(self, *args):
        super(DummyCallback, self).validation_end(*args)
        self.val_ends += 1


class DummyData(torch.utils.data.Dataset):
    """Dummy data for testing."""
    def __getitem__(self, idx):
        x = np.random.randn()
        y = np.random.randn()
        return {"x": x, "y": y}

    def __len__(self):
        return 100


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.data = DummyData()
        self.loader = torch.utils.data.DataLoader(self.data, batch_size=1)
        self.interface = DummyInterface()
        self.callback = DummyCallback()
        self.trainer = Trainer(self.interface)

    def tearDown(self):
        pass

    def test_add_callback(self):
        self.trainer.add_callback(self.callback)
        self.trainer.add_callback(self.callback)
        self.assertEqual(len(self.trainer.callbacks), 2)

    def test_calls_callbacks(self):
        epochs = 3
        self.trainer.add_callback(self.callback)
        self.trainer.train(self.loader, num_epochs=epochs)

        self.assertEqual(self.callback.batch_starts, len(self.data)*epochs)
        self.assertEqual(self.callback.batch_ends, len(self.data)*epochs)
        self.assertEqual(self.callback.epoch_starts, epochs)
        self.assertEqual(self.callback.epoch_ends, epochs)
        self.assertEqual(self.callback.epoch, epochs-1)
        self.assertEqual(self.callback.batch, len(self.data)-1)

        # With no validation data there should be no validation calls
        self.assertEqual(self.callback.val_starts, 0)
        self.assertEqual(self.callback.val_steps, 0)
        self.assertEqual(self.callback.val_ends, 0)

    def test_runs_validation(self):
        epochs = 3
        self.trainer.add_callback(self.callback)
        self.trainer.train(self.loader, num_epochs=epochs,
                           val_dataloader=self.loader)

        # With a dataloader, validation should be called
        self.assertEqual(self.callback.val_starts, epochs)
        self.assertEqual(self.callback.val_steps, epochs*len(self.data))
        self.assertEqual(self.callback.val_ends, epochs)

    def test_calls_interface(self):
        epochs = 2
        self.trainer.train(self.loader, num_epochs=epochs)

        self.assertEqual(self.interface.fwd, epochs*len(self.data))
        self.assertEqual(self.interface.bwd, epochs*len(self.data))
        self.assertEqual(self.interface.init_val, 0)
        self.assertEqual(self.interface.update_val, 0)

        self.trainer.train(self.loader, num_epochs=epochs,
                           val_dataloader=self.loader)

        self.assertEqual(self.interface.fwd, 2*epochs*len(self.data))
        self.assertEqual(self.interface.bwd, 2*epochs*len(self.data))
        self.assertEqual(self.interface.init_val, epochs)
        self.assertEqual(self.interface.update_val, epochs*len(self.data))
