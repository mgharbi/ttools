from abc import ABCMeta, abstractmethod
import logging
import os
import re
import signal

import torch as th


LOG = logging.getLogger(__name__)


class ModelInterface(metaclass=ABCMeta):
  """An adapter to run or train a model."""

  def __init__(self):
    pass

  @abstractmethod
  def forward(self, batch):
    """Runs the model on a batch of data.

    Args:
      batch (dict): batch of data provided by a data pipeline.

    Returns:
      forward_data (dict): a dictionary of outputs
    """

    forward_data = {}

    return forward_data

  @abstractmethod
  def backward(self, batch, forward_data):
    """Computes gradients, take an optimizer step and update the model.

    Args:
      batch (dict): batch of data provided by a data pipeline.
      forward_data (dict): outputs from the forward pass

    Returns:
      backward_data (dict): a dictionary of outputs
    """

    backward_data = {}

    return backward_data

  @abstractmethod
  def init_validation(self):
    """Initializes the quantities to be reported during validation.

    Returns:
      data (dict): initialized values
    """

    data = {}
    return data

  @abstractmethod
  def update_validation(self, batch, fwd, running_data):
    """Updates the running val data using the current batch's forward output.

    Args:
      batch (dict): batch of data provided by a data pipeline.
      fwd (dict): data from one forward step in validation mode
      running_data (dict): current aggregates of the validation loop.

    Returns:
      updated_data (dict): initialized values
    """

    updated_data = {}
    return updated_data

  @abstractmethod
  def finalize_validation(self, running_data):
    """Computes the final validation aggregates from the running data.

    Args:
      running_data (dict): current aggregates of the validation loop.

    Returns:
      validation_data (dict): initialized values
    """

    validation_data = {}
    return validation_data

  def __repr__(self):
    return self.__class__.__name__


class Trainer(object):
  """Implements a simple training loop with hooks for callbacks.

  Args:
    interface (ModelInterface): adapter to run forward and backward
      pass on the model being trained.

  Attributes:
    callbacks (list of Callbacks): hooks that will be called while training
      progresses.
  """

  class TrainerInterrupt(Exception):
    pass

  def __init__(self, interface):
    super(Trainer, self).__init__()
    self.callbacks = []
    self.interface = interface
    LOG.debug("Creating {}".format(self))

    signal.signal(signal.SIGINT, self.interrupt_handler)

    self._keep_running = True

  def interrupt_handler(self, signo, frame):
    LOG.debug("interrupting run")
    self._keep_running = False

  def _stop(self):
    # Reset the run flag
    self._keep_running = True
    self.__training_end()

  def add_callback(self, callback):
    """Adds a callback to the list of training hooks."""
    LOG.debug("Adding callback {}".format(callback))
    self.callbacks.append(callback)

  def train(self, dataloader, num_epochs=None, val_dataloader=None):
    """Main training loop. This starts the training procedure.

    Args:
      dataloader (DataLoader): loader that yields training batches.
      num_epochs (int, optional): max number of epochs to run.
      val_dataloader (DataLoader, optional): loader that yields validation
        batches
    """
    self.__training_start(dataloader)
    epoch = 0
    while num_epochs is None or epoch < num_epochs:
      self.__epoch_start(epoch)
      for batch_idx, batch in enumerate(dataloader):
        if not self._keep_running:
          self._stop()
          return
        self.__batch_start(batch_idx, batch)
        fwd_result = self.__forward_step(batch)
        bwd_result = self.__backward_step(batch, fwd_result)
        self.__batch_end(batch, fwd_result, bwd_result)
      self.__epoch_end()
        
      # Validate
      if val_dataloader:
        with th.no_grad():
          val_data = self.__validation_start(val_dataloader)  # data interface adapter
          for batch_idx, batch in enumerate(dataloader):
            fwd_result = self.__forward_step(batch)
            val_data = self.__validation_update(batch, fwd_result, val_data)
          self.__validation_end(val_data)

      epoch += 1

      if not self._keep_running:
        self._stop()
        return

    self._stop()

  def __repr__(self):
    return "Trainer({}, {} callbacks)".format(
      self.interface, len(self.callbacks))

  def __forward_step(self, batch):
    return self.interface.forward(batch)

  def __backward_step(self, batch, forward_data):
    return self.interface.backward(batch, forward_data)

  def __training_start(self, dataloader):
    for cb in self.callbacks:
      cb.training_start(dataloader)

  def __training_end(self):
    for cb in self.callbacks:
      cb.training_end()

  def __epoch_start(self, epoch_idx):
    for cb in self.callbacks:
      cb.epoch_start(epoch_idx)

  def __epoch_end(self):
    for cb in self.callbacks:
      cb.epoch_end()

  def __validation_start(self, dataloader):
    for cb in self.callbacks:
      cb.validation_start(dataloader)
    return self.interface.init_validation()

  def __validation_update(self, batch, fwd_data, val_data):
    for cb in self.callbacks:
      cb.validation_step(fwd_data, val_data)
    return self.interface.update_validation(batch, fwd_data, val_data)

  def __validation_end(self, val_data):
    val_data = self.interface.finalize_validation(val_data)
    for cb in self.callbacks:
      cb.validation_end(val_data)

  def __batch_start(self, batch_idx, batch):
    for cb in self.callbacks:
      cb.batch_start(batch_idx, batch)

  def __batch_end(self, batch, fwd_result, bwd_result):
    for cb in self.callbacks:
      cb.batch_end(batch, fwd_result, bwd_result)


class Checkpointer(object):
  """Save and restore model and optimizer variables.

  Args:
    root (string): path to the root directory where the files are stored.
    model (torch.nn.Module):
    optimizer (torch.optimizer):
    meta (dict):
  """

  EXTENSION = ".pth"

  def __init__(self, root, model=None, optimizer=None, meta=None):
    self.root = root
    self.model = model
    self.meta = meta
    self.optimizer = optimizer

    LOG.debug(self)

  def __repr__(self):
    return "Checkpointer with root at \"{}\"".format(self.root)

  def __path(self, path):
    return os.path.join(self.root, os.path.splitext(path)[0] + ".pth")

  def save(self, path, extras=None):
    """Save model, optimizer, metaparams and extras to relative path.

    Args:
      path (string): relative path to the file being saved (without extension).
      extras (dict): extra user-provided information to be saved with the model.
    """
    if self.optimizer is None:
      optimizer_state = None
    else:
      LOG.debug("Saving optimizer state dict")
      optimizer_state = self.optimizer.state_dict()

    if self.model is None:
      model_state = None
    else:
      LOG.debug("Saving model state dict")
      model_state = self.model.state_dict()

    filename = self.__path(path)
    os.makedirs(self.root, exist_ok=True)
    th.save({'model': model_state,
             'optimizer' : optimizer_state,
             'meta': self.meta,
             'extras': extras,
             }, filename)
    LOG.debug("Checkpoint saved to \"{}\"".format(filename))

  def load(self, path):
    """Loads a checkpoint, updates the model and optimizer and returns extra data.

    Args:
      path (string): path to the checkpoint file, relative to the root dir.

    Returns:
      extras (dict): extra information passed by the user at save time.
      meta (dict): metaparameters of the model passed at save time.
    """

    # TODO: handle case where we don't want to load the optimizer params (e.g. changing optimizer)

    filename = self.__path(path)
    chkpt = th.load(filename)

    if self.model is not None and chkpt["model"] is not None:
      LOG.debug("Loading model state dict")
      self.model.load_state_dict(chkpt["model"])

    if self.optimizer is not None and chkpt["optimizer"] is not None:
      LOG.debug("Loading optimizer state dict")
      self.optimizer.load_state_dict(chkpt["optimizer"])

    LOG.debug("Loaded checkpoint \"{}\"".format(filename))
    return tuple(chkpt[k] for k in ["extras", "meta"])

  def load_latest(self):
    """Try to load the most recent checkpoint, skip failing files.

    Returns:
      extras (dict): extra information passed by the user at save time.
      meta (dict): metaparameters of the model passed at save time.
    """
    all_checkpoints = self.sorted_checkpoints()

    extras = None
    meta = None
    for f in all_checkpoints:
      try:
        extras, meta = self.load(f)
        return extras, meta
      except Exception as e:
        LOG.debug("Could not load checkpoint \"{}\", moving on.".format(f))
    LOG.debug("No checkpoint found to load.")
    return extras, meta

  def sorted_checkpoints(self):
    """Get list of all checkpoints in root directory, sorted by creation date."""
    reg = re.compile(r".*\{}".format(Checkpointer.EXTENSION))
    if not os.path.exists(self.root):
      all_checkpoints = []
    else:
      all_checkpoints = [f for f in os.listdir(self.root) if reg.match(f)]
    mtimes = []
    for f in all_checkpoints:
      mtimes.append(os.path.getmtime(os.path.join(self.root, f)))

    mf = sorted(zip(mtimes, all_checkpoints))
    chkpts = [m[1] for m in reversed(mf)]
    LOG.debug("Sorted checkpoints {}".format(chkpts))
    return chkpts

  def delete(self, path):
    if path in self.sorted_checkpoints():
      os.remove(os.path.join(self.root, path))
    else:
      LOG.warn("Trying to delete a checkpoint that does not exists.")

  @staticmethod
  def load_meta(root):
    """Fetch model metadata without touching the saved parameters."""
    chkptr = Checkpointer(root, None, None)
    all_checkpoints = chkptr.sorted_checkpoints()

    extras, meta = chkptr.load_latest()
    return meta

#   # Load init weights from a source checkpoint
#   def override_params(self, filename, ignore=None):
#     ov_chkpt = th.load(filename)
#     tgt = self.model.state_dict()
#     src = ov_chkpt["state_dict"]
#     names = []
#     if ignore is not None:
#       ignore = re.compile(ignore)
#
#     for name, param in src.items():
#       if name in tgt and tgt[name].shape == param.shape:
#         if ignore is not None and ignore.match(name):
#           continue
#         s = "{:10.10s}".format(name)
#         s += " {:.2f} ({:.2f})".format(tgt[name].cpu().mean(), tgt[name].cpu().std())
#         tgt[name].copy_(param)
#         s += " -> {:.2f} ({:.2f})".format(param.cpu().mean(), param.cpu().std())
#         names.append(s)
#     return names
