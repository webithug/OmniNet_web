import torch
from torch import nn, Tensor
from spanet.options import Options
from spanet.dataset.types import InputType, Source, DistributionInfo, InputType
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict

class Diffusion_Sampler():
  def __init__(self, options: Options, training_dataset: JetReconstructionDataset, mean = Dict, std = Dict, mean_num_vector = Dict, std_num_vector = Dict):

    self.input_types = training_dataset.event_info.input_types
    self.mean        = mean
    self.std         = std
    self.mean_num_vector = mean_num_vector
    self.std_num_vector  = std_num_vector

  def logsnr_schedule_cosine(self, time: Tensor, logsnr_min: float = -20., logsnr_max: float = 20.) -> Tensor:
    logsnr_min = Tensor([logsnr_min]).to(time.device)
    logsnr_max = Tensor([logsnr_max]).to(time.device)
    b = torch.atan(torch.exp(-0.5 * logsnr_max)).to(time.device)
    a = (torch.atan(torch.exp(-0.5 * logsnr_min)) - b).to(time.device)
    return -2.0 * torch.log(torch.tan( a * time.to(torch.float32) + b))

  def get_logsnr_alpha_sigma(self, time: Tensor, shape=None):
    logsnr = self.logsnr_schedule_cosine(time)
    alpha  = torch.sqrt(torch.sigmoid(logsnr))
    sigma  = torch.sqrt(torch.sigmoid(-logsnr))

    if shape is not None:
      logsnr = logsnr.view(shape).to(torch.float32)
      alpha  = alpha.view(shape).to(torch.float32)
      sigma  = sigma.view(shape).to(torch.float32)

    return logsnr, alpha, sigma

  def add_perturbation(self, sources: Tuple[Source, ...], source_time: Tensor, diffusion_type: InputType) -> Tuple[Tuple[Source, ...], Tuple[Source, ...]]:
      logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(source_time.unsqueeze(-1))
      output_sources = []
      output_vectors = []
      for input_index, name in enumerate(self.input_types):
        source_before_norm, mask = sources[input_index]
        source = (source_before_norm - self.mean[name].to(source_before_norm.device)) / self.std[name].to(source_before_norm.device)

        if (self.input_types[name] == diffusion_type):
          eps = torch.randn(source.size(), device = sourc.device, dtype = torch.float32)
          perturbed_x = Source(data = source * alpha + eps * sigma,
                               mask = mask)
          score       = Source(data = eps * alpha - source * sigma,
                               mask = mask)
        else:
          perturbed_x = Source(data = source,
                               mask = mask)
          score       = Source(data = torch.zeros_like(source),
                               mask = mask)

        output_sources.append(perturbed_x)
        output_vectors.append(score)

      return tuple(output_sources), tuple(output_vectors)

  def add_perturbation_dict(self, source: Dict[str, Tensor], source_time: Tensor):
      logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(source_time.unsqueeze(-1))
      perturbed_source = OrderedDict()
      score            = OrderedDict()
      for name, source in source.items():
        source_reshaped = source.view(source.shape[0], 1, 1).contiguous()
        source_reshaped = (source_reshaped - self.mean_num_vector[name].to(source.device)) / self.std_num_vector[name].to(source.device)
        eps = torch.randn(source_reshaped.size(), device = source_reshaped.device, dtype = torch.float32)
        perturbed_source[name] = (alpha * source_reshaped + eps * sigma).view(source.size())
        score[name] = (eps * alpha - source_reshaped * sigma).view(source.size())
      return perturbed_source, score


  def prepare_format(self, sources: List[Tuple[Tensor, Tensor]], source_time: Tensor, source_num_vectors: Dict[str, Tensor], source_score: Optional[List[Tuple[Tensor, Tensor]]] = None, source_num_vectors_score: Optional[Dict[str,Tensor]] = None):

    perturbed_num_vectors_output = DistributionInfo()
    perturbed_global_source      = DistributionInfo()
    perturbed_sequential_source  = DistributionInfo()

    vector = dict()
    vector["Global"] = DistributionInfo()
    vector["Sequential"] = DistributionInfo()

    for name, source in source_num_vectors.items():
      source = source.view(source.shape[0], 1, 1).contiguous()
      perturbed_num_vectors_output[name] = Source(data = source, 
                                                  mask = torch.ones_like(source).bool())
      if source_num_vectors_score is None:
        vector["Global"][name] = Source(data = None, mask = None)
    
      else:
        vector["Global"][name] = Source(data = source_num_vectors_score[name].view(source.shape[0], 1, 1), mask = torch.ones_like(source_num_vectors_score[name]).bool())

    for input_index, name in enumerate(self.input_types):
      source, mask = sources[input_index]
      if (self.input_types[name] == InputType.Global):
        perturbed_global_source[name] = Source(data = source, mask = mask)
        if source_score is None:
          vector["Global"][name] = Source(data = None, mask = None)
        else:
          vector["Global"][name] = source_score[input_index]
      else:
        perturbed_sequential_source[name] = Source(data = source, mask = mask)
        if source_score is None:
          vector["Sequential"][name] = Source(None, None)
        else:
          vector["Sequential"][name] = source_score[input_index]

    return perturbed_num_vectors_output, perturbed_global_source, perturbed_sequential_source, vector

  def prepare_prior_distribution(self, batch_size, device):
    prior_num_vectors_output = DistributionInfo()
    prior_global_source      = DistributionInfo()
    prior_sequential_source  = DistributionInfo()

    for name, _ in self.mean_num_vector.items():
      prior_num_vectors_output[name] = Source(self.prior_sde((batch_size, 1, 1), device), torch.ones((batch_size, 1, 1), device = device).bool())

    for name, mean in self.mean.items():
      if(self.input_types[name] == InputType.Global):
        num_features = mean.shape[-1]
        prior_global_source[name] = Source(self.prior_sde((batch_size, 1, num_features), device), torch.ones((batch_size, 1, num_features), device = device).bool())

    return prior_num_vectors_output, prior_global_source, prior_sequential_source

  def generate_global_distribution(self, batch_size, device):
    prior_num_vectors_output, prior_global_source, prior_sequential_source = self.prepare_prior_distribution(batch_size, device)


  def second_order_correction(self, time_step, x, pred_images, pred_noises, alphas, sigmas, w, model, num_steps, second_order_alpha, shape):

    step_size = 1.0 / num_steps 
    t = time_step

  def DDPMSampler(self, model, data_shape, const_shape, jet, w, num_steps, mask, device, batch_size, mode, x_num_vector, x_global, x_sequential):
    
    for time_step in range(num_steps, 0, -1):
      batch_time = torch.ones((batch_size, 1), dtype = torch.int32, device = device) * time_step / num_steps
      batch_next_time = torch.ones((batch_size, 1), dtype = torch.int32, device = device) * (time_step - 1) / num_steps
      logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(batch_time, shape = const_shape)
      logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(batch_next_time, shape = const_shape)
      if mode == "Global":
        pred_v_global, pred_v_num_vector = model(x_global, batch_time, x_num_vector)
        masks = None
        pred_v_sequential = 1.0

      mean_num_vector = alpha * x_num_vector - sigma * pred_v_num_vector
      eps_num_vector  = pred_v_num_vector * alpha + x_num_vector * sigma


  def prior_sde(self, shape, device = 'cpu'):
    return torch.randn(shape, dtype=torch.float32, device = device)
