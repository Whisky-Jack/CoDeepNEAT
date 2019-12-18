from torch import nn, tensor
import torch.nn.functional as F
from src2.Configuration import config


class PadUp(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    # TODO this is inefficient, size check should only be done once, not every forward call
    def forward(self, x: tensor):
        shape = x.size()
        if shape[3] < config.min_square_dim:
            return F.pad(x,
                         [config.min_square_dim, config.min_square_dim, config.min_square_dim, config.min_square_dim])

        return x