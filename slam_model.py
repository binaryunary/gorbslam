from genericpath import exists
import json
from keras import Model
from keras.models import load_model
from keras.layers import Normalization
import numpy as np
from os import path


class SLAMModel(Model):
    def __init__(self, source_normalizer, target_normalizer):
        self.model = None
        self.source_normalizer = source_normalizer
        self.target_normalizer = target_normalizer

    def build_model(self, best_hps):
        # Build the final model using the best hyperparameters and train it.
        self.model = self.build(best_hps)

    # def fit(self, source_data, target_data, epochs, batch_size):
    #     # Fit the model
    #     # ...
    #     pass

    def predict(self, slam_trajectory):
        X = self.source_normalizer(slam_trajectory)

        predicted_trajectory_norm = self.model.predict(X)

        # Denormalize the predictions
        predicted_trajectory_denorm = self._denormalize(predicted_trajectory_norm)

        return predicted_trajectory_denorm

    def save(self, filepath):
        self.model.save(filepath)

        # Save normalizers' configurations
        normalizers_config = {
            'source_normalizer': self.source_normalizer.get_config(),
            'target_normalizer': self.target_normalizer.get_config(),
        }

        with open(filepath + '_normalizers.json', 'w') as f:
            json.dump(normalizers_config, f)

    def _denormalize(self, data):
        variance = self.target_normalizer.variance.numpy()
        mean = self.target_normalizer.mean.numpy()
        std = np.sqrt(variance)

        return data * std + mean

    @staticmethod
    def load(processed_results_dir):
        if not path.exists(processed_results_dir):
            return None

        loaded_model = load_model(processed_results_dir)
        n_inputs = loaded_model.layers[0].input_shape[-1]

        # Load normalizers' configurations
        with open(processed_results_dir + '_normalizers.json', 'r') as f:
            normalizers_config = json.load(f)

        # Create and configure normalizers
        source_normalizer = Normalization.from_config(normalizers_config['source_normalizer'])
        target_normalizer = Normalization.from_config(normalizers_config['target_normalizer'])

        # Create a new TrajectoryModel instance with the loaded weights and normalizers
        trajectory_model = SLAMModel(source_normalizer, target_normalizer)
        trajectory_model.build((None, n_inputs))
        trajectory_model.set_weights(loaded_model.get_weights())

        return trajectory_model
