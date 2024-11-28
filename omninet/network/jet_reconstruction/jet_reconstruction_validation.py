from typing import Dict, Callable
import warnings

import numpy as np
import torch

from sklearn import metrics as sk_metrics

from omninet.options import Options
from omninet.dataset.evaluator import SymmetricEvaluator
from omninet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
import wandb
import matplotlib.pyplot as plt


class JetReconstructionValidation(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionValidation, self).__init__(options, torch_script)
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)

    @property
    def particle_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            "accuracy": sk_metrics.accuracy_score,
            "sensitivity": sk_metrics.recall_score,
            "specificity": lambda t, p: sk_metrics.recall_score(~t, ~p),
            "f_score": sk_metrics.f1_score
        }

    @property
    def particle_score_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            # "roc_auc": sk_metrics.roc_auc_score,
            # "average_precision": sk_metrics.average_precision_score
        }

    def compute_metrics(self, jet_predictions, particle_scores, stacked_targets, stacked_masks, sources=None):
        event_permutation_group = self.event_permutation_tensor.cpu().numpy()
        num_permutations = len(event_permutation_group)
        num_targets, batch_size = stacked_masks.shape
        particle_predictions = particle_scores >= 0.5

        def get_list_shape(lst):
            if isinstance(lst, list) or isinstance(lst, np.ndarray):
                return [len(lst)] + get_list_shape(lst[0])
            else:
                return []

        # print(f"jet_predictions: {get_list_shape(jet_predictions)}") # list [num_targets, batch_size, num_target_children] (for TTHad: [2, 128, 3])
        # print(jet_predictions)
        # print(f"particle_scores: {particle_scores.shape}") # ndarray: [num_targets, batch_size]
        # print(particle_scores)
        # print(f"stacked_targets: {[len(stacked_targets)]+[stacked_targets[0].shape]}") # ndarray: [num_targets, batch_size, num_target_children]
        # print(stacked_targets)
        # print(f"stacked_masks: {stacked_masks.shape}") # ndarray: [num_targets, batch_size]
        # print(stacked_masks)

        # print(f"particle predictions: {particle_predictions}") # 




        # Compute all possible target permutations and take the best performing permutation
        # First compute raw_old accuracy so that we can get an accuracy score for each event
        # This will also act as the method for choosing the best permutation to compare for the other metrics.
        jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        particle_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        for i, permutation in enumerate(event_permutation_group): # loop over permutations [target1, target2] or [target2, target1]
            # print(f"permutation: {permutation}") # shows [0 1] or [1 0]
            for j, (prediction, target) in enumerate(zip(jet_predictions, stacked_targets[permutation])): # loop over targets
                jet_accuracies[i, j] = np.all(prediction == target, axis=1) # output is a vector of size=batch_size
                # print(f"prediction: {prediction}")
                # print(f"target: {target}")
                # print(f"jet_accuracies[i, j]: {jet_accuracies[i, j]}")
            particle_accuracies[i] = stacked_masks[permutation] == particle_predictions

        # print(f"jet_predictions: {jet_predictions}")
        # raise Exception("done")


        jet_accuracies = jet_accuracies.sum(1)
        particle_accuracies = particle_accuracies.sum(1)

        # Select the primary permutation which we will use for all other metrics.
        chosen_permutations = self.event_permutation_tensor[jet_accuracies.argmax(0)].T
        chosen_permutations = chosen_permutations.cpu()
        permuted_masks = torch.gather(torch.from_numpy(stacked_masks), 0, chosen_permutations).numpy()

        # permute jet_predictions according to chosen_permutation
        jet_pred_tensor = torch.tensor(np.array(jet_predictions))
        chosen_permutations = chosen_permutations.unsqueeze(-1) # add an extra dim 
        permuted_jet_pred = torch.gather(jet_pred_tensor, 0, chosen_permutations.expand(-1, -1, jet_pred_tensor.size(-1))) # use gather to permute data along one dim

        # print(f"jet predictions original: {jet_predictions}")
        # print(f"jet predictions tensor: {jet_pred_tensor}")
        # print(f"chosen permutation: {chosen_permutations}")
        # print(f"jet predictions permuted: {permuted_jet_pred}") #[num_targets, evts, jets]
        
        # print(f"chosen permutation: {chosen_permutations}") # each event has its own permutation 

        # Compute final accuracy vectors for output
        num_particles = stacked_masks.sum(0)
        jet_accuracies = jet_accuracies.max(0)
        particle_accuracies = particle_accuracies.max(0)

        # print(f"jet_accuracy: {len(jet_accuracies)}")
        # print(jet_accuracies)

        # print(f"particle_accuracy: {len(particle_accuracies)}")
        # print(particle_accuracies)

        # compute mass plots using jet_pred_permuted and sources
        if sources is not None:
            # print(f"sources: {sources}")
            # print(f"sources len: {len(sources)}") # SEQUENTIAL data and GLOBAL data
            # check sequential data!!!
            source_data = sources[0][0] 
            source_mask = sources[0][1]
            # print(f"data: {source_data}") 
            # print(f"mask: {source_mask}") 
            masked_data = source_data[source_mask]
            # print(f"masked data: {masked_data}") # should i use the masked data?

            # print(f"sources[0][0] : {sources[0][0]}") #[evt, jets, feats]
            # print(f"source mass: {sources[0][0][:,:,0]}") #[evts, jet_mass]
            # print(f"source pt: {sources[0][0][:,:,1]}") 
            # print(f"source eta: {sources[0][0][:,:,2]}")
            # print(f"source phi: {sources[0][0][:,:,3]}") 

            # denormalize the source data
            normalizer = self.normalizer
            # print(f"normalizer: {normalizer}")
            # print(f"normalizer[0]: {normalizer[0]}")
            # print(f"normalizer[1]: {normalizer[1]}")
            # print(f"source after norm: {source_data}")
            source_denorm = normalizer[0].denormalize(source_data, source_mask)
            # print(f"source before norm: {source_denorm}")
            # print(f"normalizer mean: {normalizer[0].mean}")
            # print(f"normalizer std: {normalizer[0].std}")


            # get 4-vector of all jets
            jets_mass = torch.exp(source_denorm[:,:,0]).expand(num_targets, -1, -1)
            jets_pt = torch.exp(source_denorm[:,:,1]).expand(num_targets, -1, -1)
            jets_mass = torch.round(jets_mass)
            jets_pt = torch.round(jets_pt)
            # jets_mass = source_denorm[:,:,0].expand(num_targets, -1, -1)
            # jets_pt = source_denorm[:,:,1].expand(num_targets, -1, -1)
            jets_eta = source_denorm[:,:,2].expand(num_targets, -1, -1)
            jets_phi = source_denorm[:,:,3].expand(num_targets, -1, -1)
            # print(f"jet mass: {jets_mass}")
            # print(f"jets_pt: {jets_pt}")
            # print(f"jets_eta: {jets_eta}")
            # print(f"jets_phi: {jets_phi}")

            # mass_greater_than_20 = (jets_mass > 20).any()
            # pt_greater_than_20 = (jets_pt > 20).any()

            # print(f"mass > 10: {mass_greater_than_20}")
            # print(f"pt > 10: {pt_greater_than_20}")

            # raise Exception("Done")


            # get the 4-vector of predicted jets
            jets_mass = jets_mass.to(permuted_jet_pred.device) # move data to same device (cpu or gpu)
            jets_pt = jets_pt.to(permuted_jet_pred.device)
            jets_eta = jets_eta.to(permuted_jet_pred.device)
            jets_phi = jets_phi.to(permuted_jet_pred.device)
            jets_mass_pred = torch.gather(jets_mass, 2, permuted_jet_pred) # permute jet_feats according to permuted_jet_pred
            jets_pt_pred = torch.gather(jets_pt, 2, permuted_jet_pred)
            jets_eta_pred = torch.gather(jets_eta, 2, permuted_jet_pred)
            jets_phi_pred = torch.gather(jets_phi, 2, permuted_jet_pred)

            # print(f"permutation: {permuted_jet_pred}")
            # print(f"jet mass pred: {jets_mass_pred}")
            # print(f"jet pt pred: {jets_pt_pred}")
            # print(f"jet eta pred: {jets_eta_pred}")
            # print(f"jet phi pred: {jets_phi_pred}")
            # raise Exception("Done")

            # reconstruct four-momentum of jets
            jets_energy = torch.sqrt(jets_mass_pred**2 + jets_pt_pred**2 * torch.cosh(jets_eta_pred)**2)
            jets_px = jets_pt_pred * torch.cos(jets_phi_pred)
            jets_py = jets_pt_pred * torch.sin(jets_phi_pred)
            jets_pz = jets_pt_pred * torch.sinh(jets_eta_pred)
            jets_four_momentum = torch.stack( (jets_energy, jets_px, jets_py, jets_pz), dim=-1) # (num_targets, num_events, num_jets, 4)

            # print(f"jets_four_momentum: {jets_four_momentum}") 
            
            # reconstruct four-momentum of resonance
            resonance_four_momentum = jets_four_momentum.sum(dim=2) # (num_targets, num_events, 1, 4)
            resonance_energy = resonance_four_momentum[..., 0]
            resonance_px = resonance_four_momentum[..., 1]
            resonance_py = resonance_four_momentum[..., 2]
            resonance_pz = resonance_four_momentum[..., 3]

            # resonance_mass^2 = E^2 - p^2
            resonance_mass = torch.sqrt(resonance_energy**2 - resonance_px**2 - resonance_py**2 - resonance_pz**2) # (num_targets, num_events)
            # print(f"t1 mass: {resonance_mass[0]}") 
            # print(f"t2 mass: {resonance_mass[1]}") 

            # raise Exception("Done")


        # raise Exception("done") 



        # Create the logging dictionaries
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
    
            metrics = {f"jet/accuracy_{i}_of_{j}": (jet_accuracies[num_particles == j] >= i).mean()
                    for j in range(1, num_targets + 1)
                    for i in range(1, j + 1)}

            metrics.update({f"particle/accuracy_{i}_of_{j}": (particle_accuracies[num_particles == j] >= i).mean()
                            for j in range(1, num_targets + 1)
                            for i in range(1, j + 1)})

            metrics.update({f"target{i}_mass": resonance_mass[i-1]
                            for i in range(1, num_targets + 1)})

        particle_scores = particle_scores.ravel()
        particle_targets = permuted_masks.ravel()
        particle_predictions = particle_predictions.ravel()

        # print(f"final particle_predictions: {len(particle_predictions)}")
        # print(f"final particle_predictions: {particle_predictions}")

        


        for name, metric in self.particle_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_predictions)

        for name, metric in self.particle_score_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_scores)

        # Compute the sum accuracy of all complete events to act as our target for
        # early stopping, hyperparameter optimization, learning rate scheduling, etc.
        metrics["validation_accuracy"] = metrics[f"jet/accuracy_{num_targets}_of_{num_targets}"]

        return metrics

    def validation_step(self, batch, batch_idx) -> Dict[str, np.float32]:
        # Run the base prediction step
        sources, num_jets, targets, regression_targets, classification_targets, num_seq_jets = batch

        batch_size = num_jets.shape[0]
        num_targets = len(targets)

        source_time = torch.rand(batch_size, 1).to(self.device)
        jet_predictions, particle_scores, regressions, classifications = self.predict(sources, source_time, num_seq_jets)
        # Stack all of the targets into single array, we will also move to numpy for easier the numba computations.
        stacked_targets = np.zeros(num_targets, dtype=object)
        stacked_masks = np.zeros((num_targets, batch_size), dtype=bool)
        for i, (target, mask) in enumerate(targets):
            stacked_targets[i] = target.detach().cpu().numpy()
            stacked_masks[i] = mask.detach().cpu().numpy()

        regression_targets = {
            key: value.detach().cpu().numpy()
            for key, value in regression_targets.items()
        }

        classification_targets = {
            key: value.detach().cpu().numpy()
            for key, value in classification_targets.items()
        }

        metrics = self.evaluator.full_report_string(jet_predictions, stacked_targets, stacked_masks, prefix="Purity/")

        # Apply permutation groups for each target
        for target, prediction, decoder in zip(stacked_targets, jet_predictions, self.branch_decoders):
            for indices in decoder.permutation_indices:
                if len(indices) > 1:
                    prediction[:, indices] = np.sort(prediction[:, indices])
                    target[:, indices] = np.sort(target[:, indices])

        metrics.update(self.compute_metrics(jet_predictions, particle_scores, stacked_targets, stacked_masks, sources))


        
        for key in regressions:
            delta = regressions[key] - regression_targets[key]
            
            percent_error = np.abs(delta / regression_targets[key])
            self.log(f"REGRESSION/{key}_percent_error", percent_error.mean(), sync_dist=True)

            absolute_error = np.abs(delta)
            self.log(f"REGRESSION/{key}_absolute_error", absolute_error.mean(), sync_dist=True)

            percent_deviation = delta / regression_targets[key]
            #self.logger.experiment.add_histogram(f"REGRESSION/{key}_percent_deviation", percent_deviation, self.global_step) # TensorBoard
            #percent_deviation = wandb.plot.histogram(np.array(percent_deviation), "percent deviation")
            #self.logger.experiment.log(f"REGRESSION/{key}_percent_deviation", percent_deviation)

            absolute_deviation = delta
#            self.logger.experiment.add_histogram(f"REGRESSION/{key}_absolute_deviation", absolute_deviation, self.global_step)

        for key in classifications:
            accuracy = (classifications[key] == classification_targets[key])
            self.log(f"CLASSIFICATION/{key}_accuracy", accuracy.mean(), sync_dist=True)

        for name, value in metrics.items():
            if isinstance(value, torch.Tensor): 

                # flatten and convert tensor to numpy
                flat_data = value.cpu().numpy().flatten()
                nan_mask = np.isnan(flat_data)
                if nan_mask.any():
                    flat_data[nan_mask] = np.nanmean(flat_data)

                # print("a tensor!")
                # print(f"{name}: {value}")
                self.log(f"{name}_mean", value.mean().item(), sync_dist=True)
                self.log(f"{name}_std", value.std().item(), sync_dist=True)

                if self.trainer.is_global_zero and self.current_epoch % 10 == 1: # plot mass hist every 10 epochs

                    # print(f"flat data: {flat_data}")

                    # plot the histogram as an image
                    plt.figure()
                    plt.hist(flat_data, bins=50, range=(flat_data.min(), flat_data.max()))
                    plt.title(name)
                    plt.xlabel("Mass")
                    plt.ylabel("Counts")
                    plt.close()

                    # save the plot to W&B as an image
                    self.logger.experiment.log({name: wandb.Image(plt)}, commit=False)

            # log the scalar values
            elif not np.isnan(value):
                self.log(name, value, sync_dist=True)

        return metrics

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
