#!/bin/env python
"""Train a model."""
import os

from torch.utils.data import DataLoader

import ttools

import {{name}}

LOG = ttools.get_logger(__name__)


def main(args):
    # Load config
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    cfg = ttools.parse_config(args.config, default=os.path.join(
        root, "config", "default.yml"))
    LOG.info("Configuration: %s", cfg)

    # Load model parameters from checkpoint, if any
    meta = ttools.Checkpointer.load_meta(args.checkpoint_dir)
    if meta is None:
        LOG.info("No metadata in checkpoint (or no checkpoint)")
        meta = {}

    model_params = cfg["model"]

    # Verify model params match those previously used (if any)
    if "config" in meta.keys():
        if model_params != meta["config"]["model"]:
            LOG.warning("Config does not match"
                        " checkpoint:\nconfig:%s\nchkpt:%s",
                        meta["config"]["model"], model_params)

    # Store/override config in checkpoint metadata
    meta["config"] = cfg

    # Initialize datasets
    data = {{name}}.Dataset(args.data)
    dataloader = DataLoader(data, batch_size=args.bs,
                            num_workers=args.num_worker_threads,
                            shuffle=True)
    val_dataloader = None
    if args.val_data is not None:
        val_data = {{name}}.Dataset(args.val_data)
        val_dataloader = DataLoader(val_data, batch_size=1)

    # Initialize model
    LOG.info("Model params: %s", model_params)
    model = {{name}}.BasicModel(**model_params)

    # Resume from checkpoint, if any
    checkpointer = ttools.Checkpointer(args.checkpoint_dir, model, meta=meta)
    extras, meta = checkpointer.load_latest()

    # Hook interface
    interface = {{name}}.BasicInterface(model, lr=args.lr, cuda=args.cuda)

    trainer = ttools.Trainer(interface)

    # Add callbacks
    keys = ["loss"]
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=keys, val_keys=keys))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=keys, val_keys=keys, env=args.env, port=args.port))
    trainer.add_callback({{name}}.BasicCallback(
        env=args.env, win="custom_callback", port=args.port))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(
        checkpointer, max_files=2, interval=600, max_epochs=2))

    # Start training
    trainer.train(dataloader, num_epochs=args.num_epochs,
                  val_dataloader=val_dataloader)


if __name__ == "__main__":
    parser = ttools.BasicArgumentParser()

    # You can add arguments here:
    # parser.add_argument("myarg")

    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)
