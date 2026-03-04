import torch.nn as nn


class BaseModel(nn.Module):
    """Abstract base class"""

    def train_step(self, batch, criterion):
        raise NotImplementedError

    def test_step(self, batch):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def after_optimizer_step(self):
        pass
