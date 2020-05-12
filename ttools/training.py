"""Utilities to train a model."""

from abc import ABCMeta, abstractmethod
import argparse
import os
import re
import signal

import torch as th

from .utils import get_logger


LOG = get_logger(__name__)


__all__ = ["ModelInterface", "Trainer", "Checkpointer", "BasicArgumentParser"]


class BasicArgumentParser(argparse.ArgumentParser):
    """A basic argument parser with commonly used training options."""

    def __init__(self, *args, **kwargs):
        super(BasicArgumentParser, self).__init__(*args, **kwargs)

        self.add_argument("--data", required=True, help="path to the training data.")
        self.add_argument("--val_data", help="path to the validation data.")
        self.add_argument("--config", help="path to a config file.")
        self.add_argument("--checkpoint_dir", required=True,
                          help="Output directory where checkpoints are saved")
        self.add_argument("--init_from", help="path to a checkpoint from which to try and initialize the weights.")

        self.add_argument("--lr", type=float, default=1e-4,
                          help="Learning rate for the optimizer")
        self.add_argument("--bs", type=int, default=4, help="Batch size")
        self.add_argument("--num_epochs", type=int,
                          help="Number of epochs to train for")
        self.add_argument("--num_worker_threads", default=4, type=int,
                          help="Number of threads that load data")

        # self.add_argument("--experiment_log",
        #                   help="csv file in which we log our experiments")

        self.add_argument("--cuda", action="store_true",
                          dest="cuda", help="Force GPU")
        self.add_argument("--no-cuda", action="store_false",
                          dest="cuda", help="Force CPU")

        self.add_argument("--server", help="Visdom server url")
        self.add_argument("--base_url", default="/", help="Visdom base url")
        self.add_argument("--env", default="main", help="Visdom environment")
        self.add_argument("--port", default=8097, type=int,
                          help="Visdom server port")

        self.add_argument('--debug', dest="debug", action="store_true")

        self.set_defaults(cuda=th.cuda.is_available(), debug=False)


class ModelInterface(metaclass=ABCMeta):
    """An adapter to run or train a model."""

    def __init__(self):
        pass

    @abstractmethod
    def training_step(self, batch):
        """Training step given a batch of data.

        This should implement a forward pass of the model, compute gradients,
        take an optimizer step and return useful metrics and tensors for
        visualization and training callbacks. 

        Args:
          batch (dict): batch of data provided by a data pipeline.

        Returns:
          train_step_data (dict): a dictionary of outputs.
        """
        return {}

    def init_validation(self):
        """Initializes the quantities to be reported during validation.

        The default implementation is a no-op

        Returns:
          data (dict): initialized values
        """
        LOG.warning("Running a ModelInterface validation initialization that was not overriden: this is a no-op.")
        data = {}
        return data

    def validation_step(self, batch, running_val_data):
        """Updates the running validataion with the current batch's results.

        The default implementation is a no-op

        Args:
          batch (dict): batch of data provided by a data pipeline.
          running_val_data (dict): current aggregates of the validation loop.

        Returns:
          updated_data (dict): new updated value for the running_val_data.
        """
        LOG.warning("Running a ModelInterface validation step that was not overriden: this is a no-op.")
        return {}

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

    def __init__(self, interface):
        super(Trainer, self).__init__()
        self.callbacks = []
        self.interface = interface
        LOG.debug("Creating {}".format(self))

        signal.signal(signal.SIGINT, self.interrupt_handler)

        self._keep_running = True

    def interrupt_handler(self, signo, frame):
        """Stop the training process upon receiving a SIGINT (Ctrl+C)."""
        LOG.debug("interrupting run")
        self._keep_running = False

    def _stop(self):
        # Reset the run flag
        self._keep_running = True
        self.__training_end()

    def add_callback(self, callback):
        """Adds a callback to the list of training hooks.

        Args:
            callback(ttools.Callback): callback to add.
        """
        LOG.debug("Adding callback {}".format(callback))
        # pass an interface reference to the callback
        callback.model_interface = self.interface
        self.callbacks.append(callback)

    def train(self, dataloader, starting_epoch=None, num_epochs=None,
              val_dataloader=None):
        """Main training loop. This starts the training procedure.

        Args:
          dataloader (DataLoader): loader that yields training batches.
          starting_epoch (int, optional): index of the epoch we are starting from.
          num_epochs (int, optional): max number of epochs to run.
          val_dataloader (DataLoader, optional): loader that yields validation
            batches
        """
        self.__training_start(dataloader)
        if starting_epoch is None:
            starting_epoch = 0

        LOG.info("Starting taining from epoch %d", starting_epoch)

        epoch = starting_epoch

        while num_epochs is None or epoch < starting_epoch + num_epochs:
            self.__epoch_start(epoch)
            for batch_idx, batch in enumerate(dataloader):
                if not self._keep_running:
                    self._stop()
                    return
                self.__batch_start(batch_idx, batch)
                train_step_data = self.__training_step(batch)
                self.__batch_end(batch, train_step_data)
            self.__epoch_end()

            # TODO: allow validation at intermediate steps during one epoch

            # Validate
            if val_dataloader:
                with th.no_grad():
                    running_val_data = self.__validation_start(val_dataloader)
                    for batch_idx, batch in enumerate(val_dataloader):
                        if not self._keep_running:
                            self._stop()
                            return
                        self.__val_batch_start(batch_idx, batch)
                        running_val_data = self.__validation_step(batch, running_val_data)
                        self.__val_batch_end(batch, running_val_data)
                    self.__validation_end(running_val_data)

            epoch += 1

            if not self._keep_running:
                self._stop()
                return

        self._stop()

    def __repr__(self):
        return "Trainer({}, {} callbacks)".format(
            self.interface, len(self.callbacks))

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

    def __batch_start(self, batch_idx, batch):
        for cb in self.callbacks:
            cb.batch_start(batch_idx, batch)

    def __batch_end(self, batch, train_step_data):
        for cb in self.callbacks:
            cb.batch_end(batch, train_step_data)

    def __val_batch_start(self, batch_idx, batch):
        for cb in self.callbacks:
            cb.val_batch_start(batch_idx, batch)

    def __val_batch_end(self, batch, running_val_data):
        for cb in self.callbacks:
            cb.val_batch_end(batch, running_val_data)

    def __validation_start(self, dataloader):
        for cb in self.callbacks:
            cb.validation_start(dataloader)
        return self.interface.init_validation()

    def __validation_end(self, running_val_data):
        for cb in self.callbacks:
            cb.validation_end(running_val_data)

    def __training_step(self, batch):
        return self.interface.training_step(batch)

    def __validation_step(self, batch, running_val_data):
        return self.interface.validation_step(batch, running_val_data)


class Checkpointer(object):
    """Save and restore model and optimizer variables.

    Args:
      root (string): path to the root directory where the files are stored.
      model (torch.nn.Module):
      meta (dict): a dictionary of training or configuration parameters useful
          to initialize the model upon loading the checkpoint again.
      optimizers (single or list of torch.optimizer): optimizers whose parameters will
        be checkpointed together with the model.
      schedulers (single or list of
      torch.optim.lr_scheduler): schedulers whose
          parameters will be checkpointed together with
          the model.
      prefix (str): unique prefix name in case several models are stored in the
        same folder.
    """

    EXTENSION = ".pth"

    def __init__(self, root, model=None, meta=None, optimizers=None,
                 schedulers=None, prefix=None):
        self.root = root
        self.model = model
        self.meta = meta

        # TODO(mgharbi): verify the prefixes are unique.

        if optimizers is None:
            LOG.warning("No optimizer state will be stored in the checkpointer")
        else:
            # if we have only one optimizer, make it a list
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
        self.optimizers = optimizers
        if schedulers is not None:
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
        self.schedulers = schedulers

        LOG.debug(self)

        self.prefix = ""
        if prefix is not None:
            self.prefix = prefix

    def __repr__(self):
        return "Checkpointer with root at \"{}\"".format(self.root)

    def __path(self, path, prefix=None):
        if prefix is None:
            prefix = ""
        return os.path.join(self.root, prefix+os.path.splitext(path)[0] + ".pth")

    def save(self, path, extras=None):
        """Save model, metaparams and extras to relative path.

        Args:
          path (string): relative path to the file being saved (without extension).
          extras (dict): extra user-provided information to be saved with the model.
        """

        if self.model is None:
            model_state = None
        else:
            LOG.debug("Saving model state dict")
            model_state = self.model.state_dict()

        opt_dicts = []
        if self.optimizers is not None:
            for opt in self.optimizers:
                opt_dicts.append(opt.state_dict())

        sched_dicts = []
        if self.schedulers is not None:
            for s in self.schedulers:
                sched_dicts.append(s.state_dict())

        filename = self.__path(path, prefix=self.prefix)
        os.makedirs(self.root, exist_ok=True)
        th.save({'model': model_state,
                 'meta': self.meta,
                 'extras': extras,
                 'optimizers': opt_dicts,
                 'schedulers': sched_dicts,
                 }, filename)
        LOG.debug("Checkpoint saved to \"{}\"".format(filename))

    def try_and_init_from(self, path):
        """Try to initialize the models's weights from an external checkpoint.

        Args:
            path(str): full path to the checkpoints to load model parameters
                from.
        """
        LOG.info("Loading weights from foreign checkpoint {}".format(path))
        if not os.path.exists(path):
            raise ValueError("Checkpoint {} does not exist".format(path))

        chkpt = th.load(path, map_location=th.device("cpu"))
        if "model" not in chkpt.keys() or chkpt["model"] is None:
            raise ValueError("{} has no model saved".format(path))

        mdl = chkpt["model"]
        for n, p in self.model.named_parameters():
            if n in mdl:
                p2 = mdl[n]
                if p2.shape != p.shape:
                    LOG.warning("Parameter {} ignored, checkpoint size does not match: {}, should be {}".format(n, p2.shape, p.shape))
                    continue
                LOG.debug("Parameter {} copied".format(n))
                p.data.copy_(p2)
            else:
                LOG.warning("Parameter {} ignored, not found in source checkpoint.".format(n))

        LOG.info("Weights loaded from foreign checkpoint {}".format(path))

    def load(self, path):
        """Loads a checkpoint, updates the model and returns extra data.

        Args:
          path (string): path to the checkpoint file, relative to the root dir.

        Returns:
          extras (dict): extra information passed by the user at save time.
          meta (dict): metaparameters of the model passed at save time.
        """

        filename = self.__path(path, prefix=None)
        chkpt = th.load(filename, map_location="cpu")  # TODO: check behavior

        if self.model is not None and chkpt["model"] is not None:
            LOG.debug("Loading model state dict")
            self.model.load_state_dict(chkpt["model"])

        if "optimizers" in chkpt.keys():
            if self.optimizers is not None and chkpt["optimizers"] is not None:
                try:
                    for opt, state in zip(self.optimizers,
                                          chkpt["optimizers"]):
                        LOG.debug("Loading optimizers state dict for %s", opt)
                        opt.load_state_dict(state)
                except:
                    # We do not raise an error here, e.g. in case the user simply
                    # changes optimizer
                    LOG.warning("Could not load optimizer state dicts, "
                                "starting from scratch")

        if "schedulers" in chkpt.keys():
            if self.schedulers is not None and chkpt["schedulers"] is not None:
                try:
                    for s, state in zip(self.schedulers,
                                          chkpt["schedulers"]):
                        LOG.debug("Loading scheduler state dict for %s", s)
                        s.load_state_dict(state)
                except:
                    LOG.warning("Could not load scheduler state dicts, "
                                "starting from scratch")

        LOG.debug("Loaded checkpoint \"{}\"".format(filename))
        return tuple(chkpt[k] for k in ["extras", "meta"])

    def load_latest(self):
        """Try to load the most recent checkpoint, skip failing files.

        Returns:
          extras (dict): extra user-defined information that was saved in the
              checkpoint.
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
                LOG.debug(
                    "Could not load checkpoint \"{}\", moving on ({}).".format(f, e))
        LOG.debug("No checkpoint found to load.")
        return extras, meta

    def sorted_checkpoints(self):
        """Get list of all checkpoints in root directory, sorted by creation date.

        Returns:
            chkpts (list of str): sorted checkpoints in the root folder.
        """
        reg = re.compile(r"{}.*\{}".format(self.prefix, Checkpointer.EXTENSION))
        if not os.path.exists(self.root):
            all_checkpoints = []
        else:
            all_checkpoints = [f for f in os.listdir(
                self.root) if reg.match(f)]
        mtimes = []
        for f in all_checkpoints:
            mtimes.append(os.path.getmtime(os.path.join(self.root, f)))

        mf = sorted(zip(mtimes, all_checkpoints))
        chkpts = [m[1] for m in reversed(mf)]
        LOG.debug("Sorted checkpoints {}".format(chkpts))
        return chkpts

    def delete(self, path):
        """Delete checkpoint at path.

        Args:
            path(str): full path to the checkpoint to delete.
        """
        if path in self.sorted_checkpoints():
            os.remove(os.path.join(self.root, path))
        else:
            LOG.warning("Trying to delete a checkpoint that does not exists.")

    @staticmethod
    def load_meta(root, prefix=None):
        """Fetch model metadata without touching the saved parameters.

        This loads the metadata from the most recent checkpoint in the root
        directory.

        Args:
            root(str): path to the root directory containing the checkpoints
            prefix(str): unique prefix for the checkpoint to be loaded (e.g. if
                multiple models are saved in the same location)
        """
        chkptr = Checkpointer(root, model=None, meta=None, prefix=prefix)
        LOG.debug("checkpoints: %s", chkptr.sorted_checkpoints())
        _, meta = chkptr.load_latest()
        return meta
