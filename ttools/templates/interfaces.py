"""Dummy Training interfaces."""
import torch as th

import ttools
from ttools.training import ModelInterface


LOG = ttools.get_logger(__name__)


__all__ = ["BasicInterface"]


class BasicInterface(ModelInterface):
    """Dummy interface."""
    def __init__(self, model, lr=1e-4, cuda=False):
        super(BasicInterface, self).__init__()

        self.device = "cpu"
        self.model = model

        if cuda:
            LOG.debug("Using CUDA interface")
            self.device = "cuda"
            self.model.cuda()

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, batch):
        out = self.model(batch[0].to(self.device))
        return out

    def backward(self, batch, fwd_data):
        ref = batch[1].to(self.device)
        loss = th.nn.functional.mse_loss(fwd_data, ref)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def init_validation(self):
        return {"count": 0, "loss": 0}

    def update_validation(self, batch, fwd, running_data):
        with th.no_grad():
            ref = batch[1].to(self.device)
            loss = th.nn.functional.mse_loss(fwd, ref)
            n = ref.shape[0]

        return {
            "loss": running_data["loss"] + loss.item()*n,
            "count": running_data["count"] + n
        }

    def finalize_validation(self, running_data):
        return {
            "loss": running_data["loss"] / running_data["count"]
        }
