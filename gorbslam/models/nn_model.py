import json
import math
from os import path
import os

import numpy as np
from gorbslam.common.slam_trajectory import SLAMTrajectory
from gorbslam.models.orbslam_corrector_model import ORBSLAMCorrectorModel

from gorbslam.common.utils import (
    CustomJSONEncoder,
    create_training_splits,
    downsample,
    to_xyz,
)
from keras.models import load_model
from keras.layers import Normalization
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras_tuner import Hyperband
from gorbslam.models.nn_hypermodel import NNHyperModel


class NNModel(ORBSLAMCorrectorModel):
    def __init__(self, model_dir):
        self._model_dir = model_dir
        self._project_name = path.basename(
            path.dirname(model_dir)
        )  # TODO: Find a cleaner way to do this
        self._keras_logs_dir = path.join(os.getcwd(), "keras_logs")
        self._model_path = path.join(self._model_dir, "model.keras")
        self._model_params_path = path.join(self._model_dir, "model_params.json")
        self._normalizers_path = path.join(self._model_dir, "normalizers.json")
        self._source_normalizer = None
        self._target_normalizer = None
        self._model_params = None
        self._model = None
        self._is_loaded = False
        self._callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, verbose=1),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=5, verbose=1, min_lr=1e-7
            ),
            TensorBoard(
                log_dir=path.join(
                    self._keras_logs_dir, "tensorboard", self._project_name
                )
            ),
        ]

    @property
    def model(self):
        return self._model

    @property
    def model_params(self):
        return self._model_params

    @property
    def is_loaded(self):
        return self._is_loaded

    def save_model(self):
        self._model.save(self._model_path, save_format="keras")

        # Save normalizers' configurations
        normalizers_config = {
            "source_normalizer": json.dumps(
                serialize_normalizer(self._source_normalizer), cls=CustomJSONEncoder
            ),
            "target_normalizer": json.dumps(
                serialize_normalizer(self._target_normalizer), cls=CustomJSONEncoder
            ),
        }

        with open(self._normalizers_path, "w") as f:
            json.dump(normalizers_config, f)

        with open(self._model_params_path, "w") as f:
            json.dump(self._model_params.get_config(), f)

    def load_model(self):
        if not path.exists(self._model_path) or not path.exists(self._normalizers_path):
            self._is_loaded = False
        else:
            self._model = load_model(self._model_path)
            with open(self._normalizers_path, "r") as f:
                normalizers_config = json.load(f)
            self._source_normalizer = deserialize_normalizer(
                json.loads(normalizers_config["source_normalizer"])
            )
            self._target_normalizer = deserialize_normalizer(
                json.loads(normalizers_config["target_normalizer"])
            )
            self._is_loaded = True

        return self._is_loaded

    def create_model(
        self,
        training_data: SLAMTrajectory,
        validation_data: SLAMTrajectory = None,
    ):
        source_trajectory = to_xyz(training_data.slam.slam)
        target_trajectory = to_xyz(training_data.gt.utm)
        val_source_trajectory = None
        val_target_trajectory = None
        if validation_data is not None:
            val_source_trajectory = to_xyz(validation_data.slam.slam)
            val_target_trajectory = to_xyz(validation_data.gt.utm)

        self._search_model(
            (source_trajectory, target_trajectory),
            (val_source_trajectory, val_target_trajectory),
        )
        self._train_model(
            (source_trajectory, target_trajectory),
            (val_source_trajectory, val_target_trajectory),
        )
        self.save_model()

    def predict(self, slam_trajectory):
        predicted_trajectory_norm = self._model.predict(
            self._source_normalizer(slam_trajectory)
        )

        # Denormalize the predictions
        return self._denormalize(predicted_trajectory_norm)

    def _search_model(self, training_data, validation_data=None):
        source_trajectory, target_trajectory = training_data

        # Create and configure normalizers
        self._source_normalizer = Normalization(input_shape=(3,))
        self._source_normalizer.adapt(source_trajectory)
        self._target_normalizer = Normalization(input_shape=(3,))
        self._target_normalizer.adapt(target_trajectory)

        hypermodel = NNHyperModel()
        tuner = Hyperband(
            hypermodel,
            overwrite=True,
            objective="val_loss",
            # objective=Objective('val_euclidean_distance', direction='min'),
            max_epochs=300,
            factor=3,
            seed=42,
            directory=self._keras_logs_dir,
            project_name=self._project_name,
        )

        # Downsample data to speed up hp search
        n_data = math.ceil(source_trajectory.shape[0] * 0.6)
        train_slam = downsample(source_trajectory, n_data)
        train_gt = downsample(target_trajectory, n_data)

        # If validation data is provided, split the training data into training and validation
        if validation_data is not None:
            val_source_trajectory, val_target_trajectory = validation_data
            training, validation, testing = create_training_splits(
                (train_slam, train_gt),
                (val_source_trajectory, val_target_trajectory),
                0.2,
            )

            slam = self._source_normalizer(training[0])
            gt = self._target_normalizer(training[1])
            val_slam = self._source_normalizer(validation[0])
            val_gt = self._target_normalizer(validation[1])

            tuner.search(
                slam,
                gt,
                validation_data=(val_slam, val_gt),
                epochs=100,
                callbacks=self._callbacks,
            )
        # If no validation data is provided, use take validation_split samples from the training data
        else:
            slam = self._source_normalizer(source_trajectory)
            gt = self._target_normalizer(target_trajectory)
            tuner.search(slam, gt, validation_split=0.2, callbacks=self._callbacks)

        self._model_params = tuner.get_best_hyperparameters(num_trials=1)[0]
        self._model = hypermodel.build(self._model_params)

    def _denormalize(self, data):
        variance = self._target_normalizer.variance.numpy()
        mean = self._target_normalizer.mean.numpy()
        std = np.sqrt(variance)

        return data * std + mean

    def _train_model(self, training_data, validation_data):
        source_trajectory, target_trajectory = training_data

        best_batch_size = self._model_params.get("batch_size")

        # If validation data is provided, split the training data into training and validation
        if validation_data is not None:
            val_source_trajectory, val_target_trajectory = validation_data
            training, validation, testing = create_training_splits(
                (source_trajectory, target_trajectory),
                (val_source_trajectory, val_target_trajectory),
                0.2,
            )
            slam = self._source_normalizer(training[0])
            gt = self._target_normalizer(training[1])
            val_slam = self._source_normalizer(validation[0])
            val_gt = self._target_normalizer(validation[1])

            self._model.fit(
                slam,
                gt,
                validation_data=(val_slam, val_gt),
                callbacks=self._callbacks,
                epochs=280,
                batch_size=best_batch_size,
                verbose=True,
            )
        # If no validation data is provided, use take validation_split samples from the training data
        else:
            slam = self._source_normalizer(source_trajectory)
            gt = self._target_normalizer(target_trajectory)
            self.model.fit(
                slam, gt, validation_split=0.2, epochs=280, callbacks=self._callbacks
            )

        # slam = self._source_normalizer(training[0])
        # gt = self._target_normalizer(training[1])
        # val_slam = self._source_normalizer(validation[0])
        # val_gt = self._target_normalizer(validation[1])

        # self._model.fit(
        #     slam,
        #     gt,
        #     validation_data=(val_slam, val_gt),
        #     callbacks=self._callbacks,
        #     epochs=200,
        #     batch_size=best_batch_size,
        #     verbose=True,
        # )


def serialize_normalizer(normalizer):
    return {"mean": normalizer.mean.numpy(), "variance": normalizer.variance.numpy()}


def deserialize_normalizer(config):
    normalizer = Normalization(
        mean=np.asarray(config["mean"]), variance=np.asarray(config["variance"])
    )
    normalizer.build((3,))
    return normalizer
