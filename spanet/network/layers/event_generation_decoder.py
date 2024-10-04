from typing import Dict, List, Tuple
from collections import OrderedDict

import torch
from torch import Tensor, nn

from spanet.options import Options
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from spanet.network.layers.embedding.normalizer import Normalizer
from spanet.network.layers.diffusion.time_conditioned_resnet import TimeConditionedResNet
from spanet.dataset.types import InputType, Source, DistributionInfo

class EventGenerationDecoder(nn.Module):

  def __init__(self, options: Options, training_dataset: JetReconstructionDataset):
    super(EventGenerationDecoder, self).__init__()
    
    self.input_features = training_dataset.event_info.input_features
    self.input_types           = training_dataset.event_info.input_types
    self.input_dim             = 0

    for name, source in self.input_features.items():
      if (training_dataset.event_info.input_types[name] == InputType.Sequential):
        self.input_dim += 1 # sequential variable having one global num_vector_mean
      if (training_dataset.event_info.input_types[name] == InputType.Global):
        self.input_dim += len(source)

    self.projection_dim = options.hidden_dim
    self.resnet = TimeConditionedResNet(options, options.diff_resnet_nlayer, self.input_dim, self.projection_dim, 2* self.projection_dim)

  def forward(self, sources: Tuple[Tensor, Tensor], source_time: Tensor) -> Tuple[Tensor, Tensor]:

    combined_x, mask = sources
    output  = self.resnet(combined_x, source_time)
    return output, mask
