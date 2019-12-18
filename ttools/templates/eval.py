"""Dummy evaluation code."""
import argparse
import os
import imageio

import torch as th
import numpy as np
from torch.utils.data import DataLoader

import ttools

import {{name}}

LOG = ttools.get_logger(__name__)


def main(args):
    data = {{name}}.Dataset(args.data)
    dataloader = DataLoader(data, batch_size=1, num_workers=1, shuffle=False)

    LOG.info("Loading model %s", os.path.basename(args.checkpoint_dir))
    meta = ttools.Checkpointer.load_meta(args.checkpoint_dir)

    if meta is None:
        LOG.error("No metadata found to instantiate the model, aborting.")
        return
    if "config" not in meta.keys():
        LOG.error("No configuration found in the metadata, aborting.")
        return

    LOG.info("Initializing from checkpoint metadata.")
    cfg = meta["config"]
    model_params = cfg["model"]

    model = {{name}}.BasicModel(**model_params)

    LOG.info("Running evaluation")

    cuda = th.cuda.is_available()
    interface = {{name}}.BasicInterface(model, lr=0, cuda=cuda)

    os.makedirs(args.output, exist_ok=True)
    for idx, batch in enumerate(dataloader):
        # Only evaluate one sample
        out = interface.forward(batch)
        out = ttools.tensor2image(out)
        imageio.imwrite(os.path.join(args.output, "model_output.png"),
                        out)
        break
    LOG.info("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_dir", help="directory where model is saved.")
    parser.add_argument(
        "data",
        help="path to a .txt file listing the test images to evaluate.")
    parser.add_argument(
        "output", help="Output root directory.")
    args = parser.parse_args()
    ttools.set_logger()
    main(args)
