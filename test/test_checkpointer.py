import logging
import os
import shutil
import tempfile
import unittest

import torch as th

from .. import Checkpointer

class TestCheckpointer(unittest.TestCase):

  """Test case docstring."""

  def setUp(self):
    self.root = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.root)

  def test_save_and_load_basic(self):
    chkpt = Checkpointer(self.root)
    chkpt.save("first")
    assert os.path.exists(os.path.join(self.root, "first" + Checkpointer.EXTENSION))

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
    chkpt = Checkpointer(self.root, None, None, meta=meta)
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
    opt = th.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
    chkpt = Checkpointer(self.root, model, opt)

    # Create a different model with its own optimizer
    model2 = th.nn.Conv2d(1, 1, 1)
    model2.weight.data = model.weight.data*2
    model2.bias.data = model.bias.data*2
    opt2 = th.optim.Adam(model2.parameters(), lr=1e-5, eps=1e-2)
    chkpt2 = Checkpointer(self.root, model2, opt2)

    # Save model 1 and load its params with model2
    chkpt.save("first")
    res = chkpt2.load("first")

    assert model.weight.data == model2.weight.data
    assert model.bias.data == model2.bias.data

    assert opt.state_dict()["param_groups"][0]["lr"] == opt2.state_dict()["param_groups"][0]["lr"]
    assert opt.state_dict()["param_groups"][0]["eps"] == opt2.state_dict()["param_groups"][0]["eps"]

  def test_save_gpu_and_load_cpu(self):
    # TODO(mgharbi)
    pass
