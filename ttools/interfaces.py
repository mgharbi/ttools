"""A collection of fully-specified model interfaces."""
import abc
import logging

import torch as th

from . import ModelInterface


LOG = logging.getLogger(__name__)


class GANInterface(ModelInterface, abc.ABC):
    """Abstract GAN interface.

    Args:
        gen(th.nn.Module): generator.
        discrim(th.nn.Module): discriminator.
        lr(float): learning rate for both discriminator and generator.
        ncritic(int): number of discriminator updates per generator update.
        opt(str): optimizer type for both discriminator and generator.
        cuda(bool): whether or not to use CUDA.
    """

    def __init__(self, gen, discrim, lr=1e-4, ncritic=1, opt="rmsprop",
                 cuda=th.cuda.is_available(), loss_scale=None):
        super(GANInterface, self).__init__()
        self.gen = gen
        self.discrim = discrim
        self.ncritic = ncritic
        self.loss_scale = loss_scale

        self.iter = 0

        self.cuda = cuda
        if cuda:
            self.gen.cuda()
            self.discrim.cuda()

        if opt == "sgd":
            self.opt_g = th.optim.SGD(gen.parameters(), lr=lr)
            self.opt_d = th.optim.SGD(discrim.parameters(), lr=lr)
        elif opt == "adam":
            LOG.warn("Using a momentum-based optimizer in the discriminator, this can be problematic.")
            self.opt_g = th.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
            self.opt_d = th.optim.Adam(discrim.parameters(), lr=lr, betas=(0.5, 0.999))
        elif opt == "rmsprop":
            self.opt_g = th.optim.RMSprop(gen.parameters(), lr=lr)
            self.opt_d = th.optim.RMSprop(discrim.parameters(), lr=lr)
        else:
            raise ValueError("invalid optimizer %s" % opt)

    def forward(self, batch):
        """Generate a sample.

        Args:
            batch(2-tuple of th.Tensor): input and label.
        """
        real, label_ = batch
        if self.cuda:
            real = real.cuda()
            label_ = label_.cuda()

        # Sample a latent vector
        z = self.gen.sample_z(real)
        if self.cuda:
            z = z.cuda()

        # Generate a sample
        generated = self.gen(z)

        return {"generated": generated, "z": z}

    def _extra_generator_loss(self, batch, fwd_data):
        """Computes extra losses if needed.

        Returns:
            th.Tensor with shape [1], the total extra loss.
        """
        return None

    def _discriminator_input(self, batch, fwd_data, fake=False):
        real, label_ = batch
        if self.cuda:
            real = real.cuda()
            label_ = label_.cuda()
        generated = fwd_data["generated"]

        if fake:
            return generated
        else:
            return real

    def backward(self, batch, fwd_data):
        loss = self._extra_generator_loss(batch, fwd_data)
        if self.iter < self.ncritic:  # Update discriminator
            fake_pred = self.discrim(self._discriminator_input(batch, fwd_data, True).detach())
            real_pred = self.discrim(self._discriminator_input(batch, fwd_data, False))
            loss_d = self._update_discriminator(fake_pred, real_pred)
            loss_g = None
            self.iter += 1
        else:  # Update generator
            self.iter = 0
            fake_pred_g = self.discrim(self._discriminator_input(batch, fwd_data, True))
            loss_g = self._update_generator(fake_pred_g, loss)
            loss_d = None

        if loss is not None:
            loss = loss.item()

        return { "loss_g": loss_g, "loss_d": loss_d, "loss": loss}

    @abc.abstractmethod
    def _update_discriminator(self, fake_pred, real_pred):
        pass

    @abc.abstractmethod
    def _update_generator(self, fake_pred, extra_loss):
        pass


class LSGANInterface(GANInterface):
    """Least-squares GAN interface [Mao2017].
    """

    def __init__(self, *args, **kwargs):
        super(LSGANInterface, self).__init__(*args, **kwargs)
        self.mse = th.nn.MSELoss()

    def _update_discriminator(self, fake_pred, real_pred):
        fake_loss = self.mse(fake_pred, th.zeros_like(fake_pred))
        real_loss = self.mse(real_pred, th.ones_like(real_pred))
        loss_d = 0.5*(fake_loss + real_loss)
        self.opt_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()
        return loss_d.item()

    def _update_generator(self, fake_pred, extra_loss):
        loss_g = self.mse(fake_pred, th.ones_like(fake_pred))
        if self.loss_scale is not None:
            loss_g *= self.loss_scale
        if extra_loss is not None:
            loss_g = loss_g + extra_loss
        self.opt_g.zero_grad()
        loss_g.backward()

        clip = 1
        actual = th.nn.utils.clip_grad_norm_(self.gen.parameters(), clip)
        if actual > clip:
          LOG.info("clipped gradients {} -> {}".format(clip, actual))

        self.opt_g.step()
        return loss_g.item()


class WGANInterface(GANInterface):
    """Wasserstein GAN.

    Args:
        c (float): clipping parameter for the Lipschitz constant
                   of the discriminator.
    """
    def __init__(self, gen, discrim, lr=1e-4, c=0.1, ncritic=5, opt="rmsprop"):
        super(WGANInterface, self).__init__(gen, discrim, lr=lr, ncritic=ncritic, opt=opt)
        assert c > 0, "clipping param should be positive."
        self.c = c

    def _update_discriminator(self, fake_pred, real_pred):
        # minus sign for gradient ascent
        loss_d = - (real_pred.mean() - fake_pred.mean())

        self.opt_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()

        # Clip discriminator parameters to enforce Lipschitz constraint
        for p in self.discrim.parameters():
            p.data.clamp_(-self.c, self.c)

        return loss_d.item()

    def _update_generator(self, fake_pred, extra_loss):
        loss_g = -fake_pred.mean()
        if self.loss_scale is not None:
            loss_g *= self.loss_scale
        if extra_loss is not None:
            loss_g = loss_g + loss
        self.opt_g.zero_grad()
        loss_g.backward()
        self.opt_g.step()
        return loss_g.item()
