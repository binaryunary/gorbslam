import json
import math
from os import path

import numpy as np
from slam_hypermodel import SLAMHyperModel
from keras.models import load_model
from keras.layers import Normalization
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras_tuner import RandomSearch, BayesianOptimization, Hyperband
from keras_tuner import Objective

from utils import NumpyEncoder, downsample


class SLAMModelHandler:
    def __init__(self, model_dir, keras_logs_dir):
        self.model_dir = model_dir
        self.project_name = path.basename(path.dirname(model_dir)) # TODO: Find a cleaner way to do this
        self.keras_logs_dir = keras_logs_dir
        self.model_path = path.join(self.model_dir, 'model.keras')
        self.normalizers_path = path.join(self.model_dir, 'normalizers.json')
        self.source_normalizer = None
        self.source_normalizer = None
        self.best_hps = None
        self.model = None
        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-7),
            TensorBoard(log_dir=self.keras_logs_dir),
            #  ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
        ]


    def _search_model(self, source_trajectory, target_trajectory, val_source_trajectory, val_target_trajectory):
        # Create and configure normalizers
        self.source_normalizer = Normalization(input_shape=(3,))
        self.source_normalizer.adapt(source_trajectory)
        self.target_normalizer = Normalization(input_shape=(3,))
        self.target_normalizer.adapt(target_trajectory)

        hypermodel = SLAMHyperModel()
        # tuner = RandomSearch(
        #     hypermodel,
        #     overwrite=True,
        #     # objective='val_loss',
        #     objective=Objective('val_euclidean_distance', direction='min'),
        #     max_trials=100,
        #     executions_per_trial=1,
        #     directory=self.model_dir,
        #     project_name=self.project_name
        # )

        # tuner = BayesianOptimization(
        #     hypermodel,
        #     overwrite=True,
        #     objective='val_loss',
        #     # objective=Objective('val_euclidean_distance', direction='min'),
        #     max_trials=100,
        #     directory=self.keras_logs_dir,
        #     project_name=self.project_name
        # )

        tuner = Hyperband(
            hypermodel,
            overwrite=True,
            objective='val_loss',
            # objective=Objective('val_euclidean_distance', direction='min'),
            max_epochs=210,
            factor=3,
            seed=42,
        )

        # Downsample data to speed up hp search
        n_data  = math.ceil(source_trajectory.shape[0] * 0.6)
        slam = self.source_normalizer(downsample(source_trajectory, n_data))
        gt = self.target_normalizer(downsample(target_trajectory, n_data))
        val_slam = self.source_normalizer(downsample(val_source_trajectory, n_data))
        val_gt = self.target_normalizer(downsample(val_target_trajectory, n_data))

        tuner.search(slam, gt, validation_data=(val_slam, val_gt), epochs=100, callbacks=self.callbacks)
        self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.model = hypermodel.build(self.best_hps)

    def _train_model(self, source_trajectory, target_trajectory, val_source_trajectory, val_target_trajectory):
        slam = self.source_normalizer(source_trajectory)
        gt = self.target_normalizer(target_trajectory)

        n_data = source_trajectory.shape[0]

        val_slam = self.source_normalizer(downsample(val_source_trajectory, n_data))
        val_gt = self.target_normalizer(downsample(val_target_trajectory, n_data))


        # best_epochs = self.best_hps.get('epochs')
        best_batch_size = self.best_hps.get('batch_size')
        self.model.fit(slam, gt, validation_data=(val_slam, val_gt), callbacks=self.callbacks, epochs=200, batch_size=best_batch_size, verbose=True)

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

    def create_model(self, source_trajectory, target_trajectory, val_source_trajectory, val_target_trajectory):
        self._search_model(source_trajectory, target_trajectory, val_source_trajectory, val_target_trajectory)
        self._train_model(source_trajectory, target_trajectory, val_source_trajectory, val_target_trajectory)
        self._save_model()

    def predict(self, slam_trajectory):
        predicted_trajectory_norm = self.model.predict(self.source_normalizer(slam_trajectory))

        # Denormalize the predictions
        return self._denormalize(predicted_trajectory_norm)


def serialize_normalizer(normalizer):
    return {
        'mean': normalizer.mean.numpy(),
        'variance': normalizer.variance.numpy()
    }


def deserialize_normalizer(config):
    normalizer = Normalization(mean=np.asarray(config['mean']),
                               variance=np.asarray(config['variance']))
    normalizer.build((3,))
    return normalizer
