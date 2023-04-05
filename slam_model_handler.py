import json
from os import path

import numpy as np
from slam_hypermodel import SLAMHyperModel
from keras.models import load_model
from keras.layers import Normalization
from keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch

from utils import NumpyEncoder


class SLAMModelHandler:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model_path = path.join(self.model_dir, 'model.keras')
        self.normalizers_path = path.join(self.model_dir, 'normalizers.json')
        self.source_normalizer = None
        self.source_normalizer = None
        self.model = None

    def _search_model(self, source_trajectory, target_trajectory):
        # Create and configure normalizers
        self.source_normalizer = Normalization(input_shape=(3,))
        self.source_normalizer.adapt(source_trajectory)
        self.target_normalizer = Normalization(input_shape=(3,))
        self.target_normalizer.adapt(target_trajectory)

        hypermodel = SLAMHyperModel()
        tuner = RandomSearch(
            hypermodel,
            overwrite=True,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=2,
            directory=self.model_dir,
            project_name='keras_log'
        )

        callbacks = [
            EarlyStopping(monitor='loss', patience=2),
        ]

        X = self.source_normalizer(source_trajectory)
        y = self.target_normalizer(target_trajectory)

        tuner.search(X, y, validation_split=0.2, callbacks=callbacks)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        self.model = tuner.hypermodel.build(best_hps)

    def _train_model(self, source_trajectory, target_trajectory, epochs=100, batch_size=32):
        callbacks = [
            EarlyStopping(monitor='loss', patience=2),
        ]

        X = self.source_normalizer(source_trajectory)
        y = self.target_normalizer(target_trajectory)

        self.model.fit(X, y, callbacks=callbacks, epochs=epochs, batch_size=batch_size, verbose=True)

    def _save_model(self):
        self.model.save(self.model_path, save_format='keras')

        # Save normalizers' configurations
        normalizers_config = {
            'source_normalizer': json.dumps(serialize_normalizer(self.source_normalizer), cls=NumpyEncoder),
            'target_normalizer': json.dumps(serialize_normalizer(self.target_normalizer), cls=NumpyEncoder),
        }

        with open(self.normalizers_path, 'w') as f:
            json.dump(normalizers_config, f)

    def _denormalize(self, data):
      variance = self.target_normalizer.variance.numpy()
      mean = self.target_normalizer.mean.numpy()
      std = np.sqrt(variance)

      return data * std + mean

    def load_model(self):
        if not path.exists(self.model_path) or not path.exists(self.normalizers_path):
            return False

        # Load the model
        self.model = load_model(self.model_path)

        # Load normalizers' configurations
        with open(self.normalizers_path, 'r') as f:
            normalizers_config = json.load(f)

        # Create and configure normalizers
        self.source_normalizer = deserialize_normalizer(json.loads(normalizers_config['source_normalizer']))
        self.target_normalizer = deserialize_normalizer(json.loads(normalizers_config['target_normalizer']))

        return True

    def create_model(self, source_trajectory, target_trajectory):
        self._search_model(source_trajectory, target_trajectory)
        self._train_model(source_trajectory, target_trajectory)
        self._save_model()


    def predict(self, slam_trajectory):
        predicted_trajectory_norm = self.model.predict(self.source_normalizer(slam_trajectory))

        # Denormalize the predictions
        return self._denormalize(predicted_trajectory_norm)



def serialize_normalizer(normalizer):
    return {
        'mean': normalizer.mean.numpy(),
        'variance':normalizer.variance.numpy()
    }


def deserialize_normalizer(config):
    normalizer = Normalization(mean=np.asarray(config['mean']),
                               variance=np.asarray(config['variance']))
    normalizer.build((3,))
    return normalizer
