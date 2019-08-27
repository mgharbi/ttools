"""A simple training interface using ttools."""
import argparse
import os
import logging

import numpy as np
import torch as th
from torchvision.datasets import MNIST
import torchvision.transforms as xforms
from torch.utils.data import DataLoader

import ttools


LOG = ttools.get_logger(__name__)


th.manual_seed(123)
np.random.seed(123)
th.backends.cudnn.deterministic = True


class MNISTClassifier(th.nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()

        self.convnet = th.nn.Sequential(
            th.nn.Conv2d(1, 32, 3, padding=1, stride=2),  # size 16x16
            th.nn.ReLU(inplace=True),
            th.nn.Conv2d(32, 64, 3, padding=1, stride=2),  # size 8x8
            th.nn.ReLU(inplace=True),
            th.nn.Conv2d(64, 128, 3, padding=1, stride=2),  # size 4x4
            th.nn.ReLU(inplace=True),
            th.nn.Conv2d(128, 256, 3, padding=1, stride=2),  # size 2x2
            th.nn.ReLU(inplace=True),
            th.nn.Conv2d(256, 512, 3, padding=1, stride=2),  # size 1x1
            th.nn.ReLU(inplace=True),
        )

        self.fc = th.nn.Sequential(
            th.nn.Linear(512, 2048),
            th.nn.ReLU(inplace=True),
            th.nn.Linear(2048, 10),
        )

    def forward(self, x):
        """Evaluate the classifier."""
        bs = x.shape[0]
        return self.fc(self.convnet(x).view(bs, -1))


class MNISTInterface(ttools.ModelInterface):
    """An adapter to run or train a model."""

    def __init__(self, model, device, lr=1e-4):
        self.device = device
        self.model = model.to(device)

        # loss critertion
        self._ce = th.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

        # optimizer
        self.opt = th.optim.Adam(self.model.parameters(), lr=lr)


    def forward(self, batch):
        im = batch[0]
        return self.model(im.to(self.device))

    def _accuracy(self, pred, label):
        pred_label = pred.max(1)[1]  # argmax, not need to take the softmax
        acc = ((pred_label == label).float()).mean()
        return acc

    def backward(self, batch, forward_data):
        label = batch[1].to(self.device)
        prediction = forward_data
        loss = self._ce(prediction, label)

        # Compute gradients
        self.opt.zero_grad()  # make sure we do not accumulate over previous gradients...
        loss.backward()

        # Take a gradient step
        self.opt.step()

        with th.no_grad():
            acc = self._accuracy(prediction, label)

        return {"loss": loss.item(), "accuracy": acc.item()}

    def _update_mean(self, val, prev_count, current_sum, num_added):
        new_val = val + 1.0/(prev_count+num_added)*(current_sum - num_added*val)
        return new_val

    def init_validation(self):
        return {"accuracy": 0, "loss": 0, "count": 0}

    def update_validation(self, batch, fwd, running_data):
        label = batch[1].to(self.device)
        prediction = fwd

        # Previous data and number of new elements
        acc = running_data["accuracy"]
        loss = running_data["loss"]
        count = running_data["count"]
        bs = label.shape[0]

        with th.no_grad():
            i_loss = self._ce(prediction, label).item()
            i_acc = self._accuracy(prediction, label).item()

        return {
            "accuracy": self._update_mean(acc, count, i_acc*bs, bs),
            "loss": self._update_mean(loss, count, i_loss*bs, bs),
            "count": count + bs,
        }

    def finalize_validation(self, running_data):
        return running_data  # we compute a running mean, no need to normalize


def train(args):
    """Train a MNIST classifier."""

    # Setup train and val data
    _xform = xforms.Compose([xforms.Resize([32, 32]), xforms.ToTensor()])
    data = MNIST("data/mnist", train=True, download=True, transform=_xform)
    val_data = MNIST("data/mnist", train=False, transform=_xform)

    # Initialize asynchronous dataloaders
    loader = DataLoader(data, batch_size=args.bs, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=16, num_workers=1)

    # Instantiate a model
    model = MNISTClassifier()

    # Checkpointer to save/recall model parameters
    checkpointer = ttools.Checkpointer(os.path.join(args.out, "checkpoints"), model=model, prefix="classifier_")

    # resume from a previous checkpoint, if any
    checkpointer.load_latest()

    # Setup a training interface for the model
    if th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")
    interface = MNISTInterface(model, device, lr=args.lr)

    # Create a training looper with the interface we defined
    trainer = ttools.Trainer(interface)

    # Adds several callbacks, that will be called by the trainer --------------
    # A periodic checkpointing operation
    LOG.info("This demo uses a Visdom to display the loss and accuracy, make sure you have a visdom server running! ('make visdom_server')")
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer))
    # A simple progress bar
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=["loss", "accuracy"], val_keys=["loss", "accuracy"]))
    # A volatile logging using visdom
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=["loss", "accuracy"], val_keys=["loss", "accuracy"],
        port=8080, env="mnist_demo"))
    # -------------------------------------------------------------------------

    # Start the training
    LOG.info("Training started, press Ctrl-C to interrupt.")
    trainer.train(loader, num_epochs=args.epochs, val_dataloader=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: subparsers
    parser.add_argument("data", help="directory where we download and store the MNIST dataset.")
    parser.add_argument("out", help="directory where we write the checkpoints and visualizations.")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for the optimizer.")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train for.")
    parser.add_argument("--bs", type=int, default=64, help="number of elements per batch.")
    args = parser.parse_args()
    ttools.training.set_logger(True)  # activate debug prints
    train(args)
