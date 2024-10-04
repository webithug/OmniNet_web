from typing import Tuple, Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from omninet.options import Options
from omninet.dataset.types import Batch, Source, AssignmentTargets, DistributionInfo
from omninet.dataset.regressions import regression_loss
from omninet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from omninet.network.utilities.divergence_losses import assignment_cross_entropy_loss, jensen_shannon_divergence


def numpy_tensor_array(tensor_list):
    output = np.empty(len(tensor_list), dtype=object)
    output[:] = tensor_list

    return output


class JetReconstructionTraining(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionTraining, self).__init__(options, torch_script)

        self.log_clip = torch.log(10 * torch.scalar_tensor(torch.finfo(torch.float32).eps)).item()

        self.event_particle_names = list(self.training_dataset.event_info.product_particles.keys())
        self.product_particle_names = {
            particle: self.training_dataset.event_info.product_particles[particle][0]
            for particle in self.event_particle_names
        }

    def particle_symmetric_loss(self, assignment: Tensor, detection: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        assignment_loss = assignment_cross_entropy_loss(assignment, target, mask, self.options.focal_gamma)
        detection_loss = F.binary_cross_entropy_with_logits(detection, mask.float(), reduction='none')

        return torch.stack((
            self.options.assignment_loss_scale * assignment_loss,
            self.options.detection_loss_scale * detection_loss
        ))

    def compute_symmetric_losses(self, assignments: List[Tensor], detections: List[Tensor], targets):
        symmetric_losses = []

        # TODO think of a way to avoid this memory transfer but keep permutation indices synced with checkpoint
        # Compute a separate loss term for every possible target permutation.
        for permutation in self.event_permutation_tensor.cpu().numpy():

            # Find the assignment loss for each particle in this permutation.
            current_permutation_loss = tuple(
                self.particle_symmetric_loss(assignment, detection, target, mask)
                for assignment, detection, (target, mask)
                in zip(assignments, detections, targets[permutation])
            )

            # The loss for a single permutation is the sum of particle losses.
            symmetric_losses.append(torch.stack(current_permutation_loss))

        # Shape: (NUM_PERMUTATIONS, NUM_PARTICLES, 2, BATCH_SIZE)
        return torch.stack(symmetric_losses)

    def combine_symmetric_losses(self, symmetric_losses: Tensor) -> Tuple[Tensor, Tensor]:
        # Default option is to find the minimum loss term of the symmetric options.
        # We also store which permutation we used to achieve that minimal loss.
        # combined_loss, _ = symmetric_losses.min(0)
        total_symmetric_loss = symmetric_losses.sum((1, 2))
        index = total_symmetric_loss.argmin(0)

        combined_loss = torch.gather(symmetric_losses, 0, index.expand_as(symmetric_losses))[0]

        # Simple average of all losses as a baseline.
        if self.options.combine_pair_loss.lower() == "mean":
            combined_loss = symmetric_losses.mean(0)

        # Soft minimum function to smoothly fuse all loss function weighted by their size.
        if self.options.combine_pair_loss.lower() == "softmin":
            weights = F.softmin(total_symmetric_loss, 0)
            weights = weights.unsqueeze(1).unsqueeze(1)
            combined_loss = (weights * symmetric_losses).sum(0)

        return combined_loss, index

    def symmetric_losses(
        self,
        assignments: List[Tensor],
        detections: List[Tensor],
        targets: Tuple[Tuple[Tensor, Tensor], ...]
    ) -> Tuple[Tensor, Tensor]:
        # We are only going to look at a single prediction points on the distribution for more stable loss calculation
        # We multiply the softmax values by the size of the permutation group to make every target the same
        # regardless of the number of sub-jets in each target particle
        assignments = [prediction + torch.log(torch.scalar_tensor(decoder.num_targets))
                       for prediction, decoder in zip(assignments, self.branch_decoders)]

        # Convert the targets into a numpy array of tensors so we can use fancy indexing from numpy
        targets = numpy_tensor_array(targets)

        # Compute the loss on every valid permutation of the targets
        symmetric_losses = self.compute_symmetric_losses(assignments, detections, targets)

        # Squash the permutation losses into a single value.
        return self.combine_symmetric_losses(symmetric_losses)

    def symmetric_divergence_loss(self, predictions: List[Tensor], masks: Tensor) -> Tensor:
        divergence_loss = []

        for i, j in self.event_info.event_transpositions:
            # Symmetric divergence between these two distributions
            div = jensen_shannon_divergence(predictions[i], predictions[j])

            # ERF term for loss
            loss = torch.exp(-(div ** 2))
            loss = loss.masked_fill(~masks[i], 0.0)
            loss = loss.masked_fill(~masks[j], 0.0)

            divergence_loss.append(loss)

        return torch.stack(divergence_loss).mean(0)
        # return -1 * torch.stack(divergence_loss).sum(0) / len(self.training_dataset.unordered_event_transpositions)

    def add_kl_loss(
            self,
            total_loss: List[Tensor],
            assignments: List[Tensor],
            masks: Tensor,
            weights: Tensor,
            mode: str
    ) -> List[Tensor]:
        if len(self.event_info.event_transpositions) == 0:
            return total_loss

        # Compute the symmetric loss between all valid pairs of distributions.
        kl_loss = self.symmetric_divergence_loss(assignments, masks)
        kl_loss = (weights * kl_loss).sum() / masks.sum()

        with torch.no_grad():
            self.log("loss/{mode}/symmetric_loss", kl_loss, sync_dist=True)
            if torch.isnan(kl_loss):
                raise ValueError("Symmetric KL Loss has diverged.")

        return total_loss + [self.options.kl_loss_scale * kl_loss]

    def add_regression_loss(
            self,
            total_loss: List[Tensor],
            predictions: Dict[str, Tensor],
            targets:  Dict[str, Tensor],
            weights:  Tensor,
            mode: str
    ) -> List[Tensor]:

        # weights: [B, 1]
      
        regression_terms = []

        for key in targets:
            current_target_type = self.training_dataset.regression_types[key]
            current_prediction = predictions[key]
            current_target = targets[key]

            current_mean = self.regression_decoder.networks[key].mean
            current_std = self.regression_decoder.networks[key].std

            current_mask = ~torch.isnan(current_target)

            current_loss = regression_loss(current_target_type)(
                current_prediction[current_mask],
                current_target[current_mask],
                current_mean,
                current_std
            )
            current_loss = current_loss * weights
            current_loss = torch.mean(current_loss)

            with torch.no_grad():
                self.log(f"loss/{mode}/regression/{key}", current_loss, sync_dist=True)

            regression_terms.append(self.options.regression_loss_scale * current_loss)

        return total_loss + regression_terms

    def add_classification_loss(
            self,
            total_loss: List[Tensor],
            predictions: Dict[str, Tensor],
            targets: Dict[str, Tensor],
            weights: Tensor,
            mode: str
    ) -> List[Tensor]:
        classification_terms = []

        for key in targets:
            current_prediction = predictions[key]
            current_target = targets[key]

            weight = None if not self.balance_classifications else self.classification_weights[key]
            current_loss = F.cross_entropy(
                current_prediction,
                current_target,
                ignore_index=-1,
                weight=weight,
                reduction = 'none'
            )
            current_loss = torch.mean(current_loss * weights)
            classification_terms.append(self.options.classification_loss_scale * current_loss)

            with torch.no_grad():
                self.log(f"loss/{mode}/classification/{key}", current_loss, sync_dist=True)

        return total_loss + classification_terms

    def add_generation_loss(
      self,
      total_loss: List[Tensor],
      predict_score: Tuple[Tensor, Tensor],
      target_score:  Tuple[Tensor, Tensor],
      name: str
    ) -> List[Tensor]:

      generation_terms = []


      v_prediction, mask_prediction = predict_score
      v_target,  mask_target        = target_score
      if name == "Sequential":
        current_loss       = torch.sum(((v_prediction - v_target) ** 2) * (mask_prediction.float())) / (torch.sum((mask_prediction.float()))) / v_prediction.shape[-1]
      else:
        current_loss       = torch.mean((v_prediction - v_target) ** 2)
      generation_terms.append(self.options.generation_loss_scale * current_loss)
      with torch.no_grad():
        self.log(f"loss/generation/{name}", current_loss, sync_dist=True)
      return total_loss + generation_terms


    def training_step(self, batch: Batch, batch_nb: int) -> Dict[str, Tensor]:
        total_loss = []
        total_loss += self.training_func(batch, batch_nb, "classifier")
        if (self.options.generation_loss_scale > 0):
          total_loss += self.training_func(batch, batch_nb, "generator")
        # ===================================================================================================
        # Combine and return the loss
        # ---------------------------------------------------------------------------------------------------
        total_loss = torch.cat([loss.view(-1) for loss in total_loss])

        self.log("loss/total_loss", total_loss.sum(), sync_dist=True)

        return total_loss.mean()
    def training_func(self, batch: Batch, batch_nb: int, mode = "classifier") -> Dict[str, Tensor]:
        # ===================================================================================================
        # Network Forward Pass
        # ---------------------------------------------------------------------------------------------------
        batch_size = self.options.batch_size
        if mode == "classifier":
          source_time = torch.zeros(self.options.batch_size, 1).to(self.device)
          alpha = torch.ones_like(source_time).view(self.options.batch_size)
        else:
          source_time = torch.rand(self.options.batch_size, 1).to(self.device)
          _, alpha, _ = self.diffusion_sampler.get_logsnr_alpha_sigma(source_time, (self.options.batch_size))

        outputs = self.forward(batch.sources, source_time, batch.num_sequential_vectors, mode)

        # ===================================================================================================
        # Initial log-likelihood loss for classification task
        # ---------------------------------------------------------------------------------------------------
        symmetric_losses, best_indices = self.symmetric_losses(
            outputs.assignments,
            outputs.detections,
            batch.assignment_targets
        )

        # Construct the newly permuted masks based on the minimal permutation found during NLL loss.
        permutations = self.event_permutation_tensor[best_indices].T
        masks = torch.stack([target.mask for target in batch.assignment_targets])
        masks = torch.gather(masks, 0, permutations)

        # ===================================================================================================
        # Balance the loss based on the distribution of various classes in the dataset.
        # ---------------------------------------------------------------------------------------------------

        # Default unity weight on correct device.
        weights = torch.ones_like(symmetric_losses) # [:, :, batch_size]

        # Balance based on the particles present - only used in partial event training
        if self.balance_particles:
            class_indices = (masks * self.particle_index_tensor.unsqueeze(1)).sum(0)
            weights *= self.particle_weights_tensor[class_indices]

        # Balance based on the number of jets in this event
        if self.balance_jets:
            weights *= self.jet_weights_tensor[batch.num_vectors]

        weights_alpha = (alpha**2).unsqueeze(0).unsqueeze(0) # [1, 1, batch_size]
        weights = weights * weights_alpha

        # Take the weighted average of the symmetric loss terms.
        masks = masks.unsqueeze(1)
        symmetric_losses = (weights * symmetric_losses).sum(-1) / torch.clamp(masks.sum(-1), 1, None)
        assignment_loss, detection_loss = torch.unbind(symmetric_losses, 1)

        # ===================================================================================================
        # Some basic logging
        # ---------------------------------------------------------------------------------------------------
        with torch.no_grad():
            for name, l in zip(self.training_dataset.assignments, assignment_loss):
                self.log(f"loss/{mode}/{name}/assignment_loss", l, sync_dist=True)

            for name, l in zip(self.training_dataset.assignments, detection_loss):
                self.log(f"loss/{mode}/{name}/detection_loss", l, sync_dist=True)

            if torch.isnan(assignment_loss).any():
                raise ValueError("Assignment loss has diverged!")

            if torch.isinf(assignment_loss).any():
                raise ValueError("Assignment targets contain a collision.")

        # ===================================================================================================
        # Start constructing the list of all computed loss terms.
        # ---------------------------------------------------------------------------------------------------
        total_loss = []

        if self.options.assignment_loss_scale > 0:
            total_loss.append(assignment_loss)

        if self.options.detection_loss_scale > 0:
            total_loss.append(detection_loss)

        # ===================================================================================================
        # Auxiliary loss terms which are added to reconstruction loss for alternative targets.
        # ---------------------------------------------------------------------------------------------------
        if self.options.kl_loss_scale > 0:
            total_loss = self.add_kl_loss(total_loss, outputs.assignments, masks, weights, mode)

        batch_weights = (alpha**2).view(batch_size, 1)
        if self.options.regression_loss_scale > 0:
            total_loss = self.add_regression_loss(total_loss, outputs.regressions, batch.regression_targets, batch_weights, mode)

        if self.options.classification_loss_scale > 0:
            total_loss = self.add_classification_loss(total_loss, outputs.classifications, batch.classification_targets, batch_weights, mode)

        if (self.options.generation_loss_scale > 0 and not (mode == 'classifier')):
            total_loss = self.add_generation_loss(total_loss, outputs.pred_score["Global"], outputs.true_score["Global"], "Global")
            total_loss = self.add_generation_loss(total_loss, outputs.pred_score["Sequential"], outputs.true_score["Sequential"], "Sequential")


        return total_loss
