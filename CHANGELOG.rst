0.1.5
-----

- Remove coloredlogs by default

0.1.4
-----

- Adds a label to progress bar

0.1.3
-----

- Adds identity op so that padding can be turned on/off without weight loading
  mismatch.

0.1.2
-----

- Adds reflection padding


0.1.1
-----

- adds a "use_sobel" flag to derivative filter.

0.1.0
-----

- Breaking changes in the ModelInterface and Callback API:
  - replaces forward/backward methods and update_validation with training_step, validation_step.

t0.0.36
------

- Adds a scheduler callback and option to save its state in the callback

0.0.35
------

- Fix multiplot callback

0.0.34
------

- Adds conv-based bilinear and bicubic upsampling

0.0.33
------

- Adds a base_url parameter to visdom callbacks

0.0.30
------

- Adds optimizers state_dict i/o to checkpointers

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
