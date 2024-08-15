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
    
    mean_num_vector, std_num_vector = training_dataset.compute_num_vector_statistics()
    mean, std                       = training_dataset.compute_source_statistics()

    self.mean_num_vector       = mean_num_vector
    self.std_num_vector        = std_num_vector
    self.mean                  = mean
    self.std                   = std

    self.num_vector_normalizer = []
    self.normalizer            = []
    self.input_types           = training_dataset.event_info.input_types
    self.input_dim             = 0
    self.output_dict           = OrderedDict()
    self.num_vector_output_dict= OrderedDict()

    output_index = 0

    for name, source in mean_num_vector.items():
      self.num_vector_normalizer.append(Normalizer(source, std_num_vector[name]))
      self.input_dim += 1 # num_vector_mean always has only one element
      self.num_vector_output_dict[name] = [output_index]
      output_index += 1

    self.num_vector_normalizer = nn.ModuleList(self.num_vector_normalizer)

    for name, source in mean.items():
      if (training_dataset.event_info.input_types[name] == InputType.Global):
        self.normalizer.append(Normalizer(source, std[name]))
        self.input_dim += source.shape[-1]
        self.output_dict[name] = []
        for i in range(source.shape[-1]):
          self.output_dict[name].append(output_index)
          output_index += 1
      else:
        self.normalizer.append(nn.Identity())

    self.normalizer = nn.ModuleList(self.normalizer)


    self.projection_dim = options.hidden_dim
    self.resnet = TimeConditionedResNet(options, options.diff_resnet_nlayer, self.input_dim, self.projection_dim, 2* self.projection_dim)

  def forward(self, sources: Tuple[Tensor, Tensor], source_time: Tensor) -> Tuple[Tensor, Tensor]:

    combined_x, mask = sources
    output  = self.resnet(combined_x, source_time)
    return output, mask
