0.0.29
------

- De-activate 'requires_grad' flag in GAN interfaces.

0.0.28
------

- Allows multiple inputs in discriminators

0.0.27
------

- Adds Multiplot callback

0.0.26
------

- Adds fixupResNet

0.0.25
------

- Makes padding optional in resnet

0.0.24
------

- Fix crop_like to accept odd sizes

0.0.23
------

- Fix WGAN constructor

0.0.22
------

- adds --server flag for remote visdom server to callbacks and argparser.

0.0.21
------

- gradient clipping in GAN interface
- separate extra losses in GAN interface

0.0.20
------

- re-enable image scripts

0.0.19
------

- resources in MANIFEST

0.0.18
------

- templates registered as resources

0.0.17
------

- adds a scaffolding mechanism to initialize a ttemplate repo with `ttools.new`

0.0.16
------

- fixes device bug in rgb2ycc

0.0.15
------

- adds yaml deps

0.0.14
------

- adds resize script

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
