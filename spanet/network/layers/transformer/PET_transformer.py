import torch
from torch import nn, Tensor
import torch.nn.functional as F

from spanet.options import Options
from spanet.network.layers.linear_block.masking import create_masking
from spanet.network.layers.linear_block.regularization import StochasticDepth, LayerScale


class PointEdgeTransformerBlock(nn.Module):
  def __init__(self, options: Options):
    super(PointEdgeTransformerBlock, self).__init__()
    self.projection_dim = options.hidden_dim
    self.num_heads      = options.PET_num_heads
    self.drop_probability = options.PET_drop_probability
    self.dropout        = options.PET_dropout
    self.layer_scale    = options.PET_layer_scale
    self.layer_scale_init = options.PET_layer_scale_init
    self.talking_head   = options.PET_talking_head

    self.group_norm1    = nn.GroupNorm(1, options.nMaxJet)
    self.group_norm2    = nn.GroupNorm(1, options.nMaxJet)
    self.dense1         = nn.Sequential(nn.Linear(self.projection_dim, 2 * self.projection_dim), nn.GELU())
    self.dense2         = nn.Linear(2*self.projection_dim, self.projection_dim)
    self.dropout_block  = nn.Dropout(self.dropout)
    self.multihead_attn = nn.MultiheadAttention(self.projection_dim, self.num_heads)
    self.stochastic_depth = StochasticDepth(self.drop_probability)
    self.layer_scale_fn1   = LayerScale(self.layer_scale_init, self.projection_dim)
    self.layer_scale_fn2   = LayerScale(self.layer_scale_init, self.projection_dim)
    self.masking         = create_masking(options.masking)

  def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:

    # -------------------------
    # Transformer block for PET module
    # x: [T, B, D]
    # -------------------------

    encoded = x.transpose(0,1).contiguous() # [B, T, D]
    x1 = self.group_norm1(encoded).transpose(0,1) # [T, B, D]
    updates, _ = self.multihead_attn(x1, x1, x1, key_padding_mask=padding_mask)
    updates    = updates.transpose(0,1) # [B, T, D]
    if self.layer_scale:
      updates = self.masking(self.layer_scale_fn1(updates), sequence_mask.transpose(0,1)) # [B, T, D]

    updates = self.stochastic_depth(updates) 
    x2 = updates + encoded

    x3 = self.group_norm2(x2)
    x3 = self.dense1(x3)
    x3 = self.dropout_block(x3)
    x3 = self.dense2(x3) # [B, T, D]

    if self.layer_scale:
      x3 = self.masking(self.layer_scale_fn2(updates), sequence_mask.transpose(0,1))

    x3 = self.stochastic_depth(x3)
    encoded = self.masking((x3 + x2).transpose(0,1), sequence_mask) # [T, B D]

    return encoded

class PointEdgeTransformer(nn.Module):
  def __init__(self, options: Options, num_layers: int):
    super(PointEdgeTransformer, self).__init__()
    self.masking = create_masking(options.masking)
    self.transformers = nn.ModuleList([PointEdgeTransformerBlock(options) for i in range(num_layers)])

  def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
    encoded = x.contiguous()
    for transformer in self.transformers:
      encoded = transformer(encoded, padding_mask, sequence_mask)

    return encoded
