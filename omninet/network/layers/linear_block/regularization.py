import torch
from torch import nn, Tensor
import torch.nn.functional as F

from omninet.options import Options
from omninet.network.layers.linear_block.masking import create_masking

# Implementing StochasticDepth, LayerScale, and TalkingHeadAttention as described earlier
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        # --------------
        # x: [B, T, D]
        # --------------
        if self.training:
            keep_prob = 1 - self.drop_prob
            # Creating a tensor of shape [batch_size, 1, 1, ...] to match x's shape
            random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
            # Random tensor values greater than 1 are set to 1, otherwise 0
            random_tensor = torch.floor(random_tensor)
            return x * random_tensor
        return x


class LayerScale(nn.Module):
    def __init__(self, init_values, projection_dim):
        super(LayerScale, self).__init__()
        self.init_values = init_values
        self.projection_dim = projection_dim
        self.gamma = nn.Parameter(torch.full((projection_dim,), self.init_values), requires_grad = True)

    def forward(self, inputs):
        return inputs * self.gamma

