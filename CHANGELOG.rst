0.0.13
------

- update GAN interfaces
- adds .yml config parser
- updates perceptual losses

0.0.13
-----

- update entrypoint build

0.0.12
-----

- adds im2video script

0.0.11
-----

- Bug fixes in Tensorboard logger
- Allows debug to print np.ndarray in addition to th.Tensor

0.0.10
-----

- Adds tile extractor for numpy array and torch tensors.
- Fixes CheckpointingCallback: no longer delete end of epoch checkpoints by
  default according to `max_files`.
- Adds error to ExperimentLoggerCallback and CSVLoggingCallback, not
  implemented yet.

0.0.9
-----

- Cleanup GAN interface

0.0.8
-----

- Disable GAN when weight = 0

0.0.7
-----

- Added LPIPS and ELPIPS perceptual losses to ttools/modules/losses.pyj:w

0.0.6
-----

- Minor changes to the GAN interfaces

0.0.5
-----

- Bug fixes in Tensorboard Callbacks
- Fixes a bug in the UNet channel counts with non-integral "increase_factor".

0.0.4
-----

- Adds GAN interfaces

0.0.3
-----

- Fixes a bug in ResidualBlock
- Adds tests for ResidualBlock
- moves set_logger and get_logger from training.py to utils.py
