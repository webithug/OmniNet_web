import numpy as np
import torch
from torch import nn, Tensor

from spanet.options import Options
from spanet.dataset.types import Tuple, Outputs, Source, Predictions, DistributionInfo, InputType
from typing import Dict

from spanet.network.layers.vector_encoder import JetEncoder
from spanet.network.layers.branch_decoder import BranchDecoder
from spanet.network.layers.embedding import MultiInputVectorEmbedding
from spanet.network.layers.embedding.local_embedding import LocalEmbedding
from spanet.network.layers.regression_decoder import RegressionDecoder
from spanet.network.layers.classification_decoder import ClassificationDecoder
from spanet.network.layers.event_generation_decoder import EventGenerationDecoder
from spanet.network.layers.jet_generation_decoder import JetGenerationDecoder

from spanet.network.prediction_selection import extract_predictions
from spanet.network.jet_reconstruction.jet_reconstruction_base import JetReconstructionBase
from spanet.network.layers.diffusion.sampler import Diffusion_Sampler
from collections import OrderedDict
TArray = np.ndarray


class JetReconstructionNetwork(JetReconstructionBase):
    def __init__(self, options: Options, torch_script: bool = False):
        """ Base class defining the SPANet architecture.

        Parameters
        ----------
        options: Options
            Global options for the entire network.
            See network.options.Options
        """
        super(JetReconstructionNetwork, self).__init__(options)

        compile_module = torch.jit.script if torch_script else lambda x: x

        self.hidden_dim = options.hidden_dim

        self.embedding = compile_module(MultiInputVectorEmbedding(
            options,
            self.training_dataset
        ))


        self.encoder = compile_module(JetEncoder(
            options,
        ))

        self.branch_decoders = nn.ModuleList([
            BranchDecoder(
                options,
                event_particle_name,
                self.event_info.product_particles[event_particle_name].names,
                product_symmetry,
                self.enable_softmax
            )
            for event_particle_name, product_symmetry
            in self.event_info.product_symmetries.items()
        ])

        self.regression_decoder = compile_module(RegressionDecoder(
            options,
            self.training_dataset
        ))

        self.classification_decoder = compile_module(ClassificationDecoder(
            options,
            self.training_dataset
        ))


        self.event_generation_decoder = compile_module(EventGenerationDecoder(
            options,
            self.training_dataset
        ))

        self.diffusion_sampler = Diffusion_Sampler(options,
                                                   self.training_dataset,
                                                   self.event_generation_decoder.mean,
                                                   self.event_generation_decoder.std,
                                                   self.event_generation_decoder.mean_num_vector,
                                                   self.event_generation_decoder.std_num_vector)

        self.jet_generation_decoder  = compile_module(JetGenerationDecoder(
          options,
          self.diffusion_sampler.num_sequential
        ))

        self.input_features = self.training_dataset.event_info.input_features
        self.input_types = self.diffusion_sampler.input_types
        self.output_dim_mapping = self.diffusion_sampler.output_dim_mapping

        # An example input for generating the network's graph, batch size of 2
        # self.example_input_array = tuple(x.contiguous() for x in self.training_dataset[:2][0])

    @property
    def enable_softmax(self):
        return True

    def forward(self, sources: Tuple[Source, ...], source_time: Tensor, source_num_vector: Dict[str, Tensor]) -> Outputs:

        ############################################################
        ## Perform normalization & Add perturbation for diffusion ##
        ############################################################

        # Perturbed only global source
        sources_global_perturbed, sources_score_global_perturbed = self.diffusion_sampler.add_perturbation(sources, source_time, InputType.Global) # Output results are already normalized
        # Perturbed only sequential, source
        sources_seq_perturbed, sources_score_seq_perturbed        = self.diffusion_sampler.add_perturbation(sources, source_time, InputType.Sequential) # Output are already normalized
        # Perturb number of vectors(jets)
        perturbed_source_num_vector, source_num_vector_score = self.diffusion_sampler.add_perturbation_dict(source_num_vector, source_time) # Output are already normalized

        # Combine global information for the input of global generation head
        x_global_perturbed, score_global_perturbed = self.diffusion_sampler.prepare_format(sources_global_perturbed, source_time, perturbed_source_num_vector, sources_score_global_perturbed, source_num_vector_score)
        x_seq_perturbed, score_seq_perturbed = self.diffusion_sampler.prepare_format(sources_seq_perturbed, source_time, perturbed_source_num_vector, sources_score_seq_perturbed, source_num_vector_score)

        # Embed all of the different input regression_vectors into the same latent space.
        embeddings, padding_masks, sequence_masks, global_masks = self.embedding(sources_seq_perturbed, source_time)

        # Extract features from data using transformer
        hidden, event_vector = self.encoder(embeddings, padding_masks, sequence_masks)


        pred_v_global = self.event_generation_decoder(x_global_perturbed["Global"], source_time) # pred_v_global: Source(data, mask)
        pred_v_sequential = self.jet_generation_decoder(embeddings, source_time, padding_masks, sequence_masks, global_masks) 

        pred_score = dict()
        pred_score["Global"] = pred_v_global
        pred_score["Sequential"] = pred_v_sequential

        true_score = dict()
        true_score["Global"] = score_global_perturbed["Global"]
        true_score["Sequential"] = score_seq_perturbed["Sequential"]


        # Create output lists for each particle in event.
        assignments = []
        detections = []

        encoded_vectors = {
            "EVENT": event_vector
        }

        # Pass the shared hidden state to every decoder branch
        for decoder in self.branch_decoders:
            (
                assignment,
                detection,
                assignment_mask,
                event_particle_vector,
                product_particle_vectors
            ) = decoder(hidden, padding_masks, sequence_masks, global_masks)

            assignments.append(assignment)
            detections.append(detection)

            # Assign the summarising vectors to their correct structure.
            encoded_vectors["/".join([decoder.particle_name, "PARTICLE"])] = event_particle_vector
            for product_name, product_vector in zip(decoder.product_names, product_particle_vectors):
                encoded_vectors["/".join([decoder.particle_name, product_name])] = product_vector

        # Predict the valid regressions for any real values associated with the event.
        regressions = self.regression_decoder(encoded_vectors)

        # Predict additional classification targets for any branch of the event.
        classifications = self.classification_decoder(encoded_vectors)


        return Outputs(
            assignments,
            detections,
            encoded_vectors,
            regressions,
            classifications,
            true_score,
            pred_score
        )

    def predict(self, sources: Tuple[Source, ...], source_time: Tensor, source_num_vectors: Dict[str, Tensor]) -> Predictions:
        with torch.no_grad():
            outputs = self.forward(sources, source_time, source_num_vectors)

            # Extract assignment probabilities and find the least conflicting assignment.
            assignments = extract_predictions([
                np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
                for assignment in outputs.assignments
            ])

            # Convert detection logits into probabilities and move to CPU.
            detections = np.stack([
                torch.sigmoid(detection).cpu().numpy()
                for detection in outputs.detections
            ])

            # Move regressions to CPU and away from torch.
            regressions = {
                key: value.cpu().numpy()
                for key, value in outputs.regressions.items()
            }

            classifications = {
                key: value.cpu().argmax(1).numpy()
                for key, value in outputs.classifications.items()
            }

        return Predictions(
            assignments,
            detections,
            regressions,
            classifications
        )

    def predict_assignments(self, sources: Tuple[Source, ...]) -> np.ndarray:
        # Run the base prediction step
        with torch.no_grad():
            assignments = [
                np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
                for assignment in self.forward(sources).assignments
            ]

        # Find the optimal selection of jets from the output distributions.
        return extract_predictions(assignments)

    def predict_assignments_and_detections(self, sources: Tuple[Source, ...]) -> Tuple[TArray, TArray]:
        assignments, detections, regressions, classifications = self.predict(sources)

        # Always predict the particle exists if we didn't train on it
        if self.options.detection_loss_scale == 0:
            detections += 1

        return assignments, detections >= 0.5

    def predict_sequential_vector(self, sources: Tuple[Source, ...], source_time: Tensor, source_num_vector: Dict[str, Tensor]) -> Tuple[Source, ...]:

      x_seq_perturbed, _ = self.diffusion_sampler.prepare_format(sources, source_time, source_num_vector, sources, source_num_vector)
      embeddings, padding_masks, sequence_masks, global_masks = self.embedding(sources, source_time)
      pred_v_sequential = self.jet_generation_decoder(embeddings, source_time, padding_masks, sequence_masks, global_masks)

      output_v = []

      for input_index, name in enumerate(self.input_types):
        data, mask = sources[input_index]
        if (self.input_types[name] == InputType.Global):
          output_v.append(Source(torch.zeros_like(data), mask))
        else:
          pred_v = pred_v_sequential[0][..., self.output_dim_mapping["Sequential"][name]]
          output_v.append(Source(pred_v, mask))

      return tuple(output_v), global_masks

    def generate(self, batch_size: int):
      Generated_Feature = self.diffusion_sampler.generate(self, batch_size, num_steps = 100)
      return Generated_Feature


    def get_reference_sample(self, sources: Tuple[Source, ...], source_time: Tensor, source_num_vector: Dict[str, Tensor]) -> Outputs:

      perturbed_sources, sources_score = self.diffusion_sampler.add_perturbation(sources, source_time, InputType.Global) # Output results are already normalized
      perturbed_source_num_vector, source_num_vector_score = self.diffusion_sampler.add_perturbation_dict(source_num_vector, source_time) # Output are already normalized
      perturbed_x, perturbed_score = self.diffusion_sampler.prepare_format(perturbed_sources, source_time, perturbed_source_num_vector, sources_score, source_num_vector_score)

      sample = self.diffusion_sampler.decode_tensor(perturbed_x["Global"], "Global")
      sample_seq = self.diffusion_sampler.decode_tensor(perturbed_x["Sequential"], "Sequential")
      sample = self.diffusion_sampler.denormalizer(sample, sample_seq)
      return sample
