import math

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import Normalization
from keras_tuner import Hyperband
from tensorflow_addons.optimizers import CyclicalLearningRate

from gorbslam.common.utils import create_training_splits, downsample
from gorbslam.models.nn_hypermodel import NNHyperModel
from gorbslam.models.fcnn_model import FCNNModel


class FCNNCLRModel(FCNNModel):
    def __init__(self, model_dir):
        super().__init__(model_dir)
        self._callbacks = [
            EarlyStopping(monitor="val_loss", patience=20, verbose=1),
        ]

    def _search_model(self, training_data, validation_data=None):
        source_trajectory, target_trajectory = training_data

        # Create and configure normalizers
        self._source_normalizer = Normalization(input_shape=(3,))
        self._source_normalizer.adapt(source_trajectory)
        self._target_normalizer = Normalization(input_shape=(3,))
        self._target_normalizer.adapt(target_trajectory)

        clr = CyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=2000,
            scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
            scale_mode="cycle",
        )

        hypermodel = NNHyperModel(clr)
        tuner = Hyperband(
            hypermodel,
            overwrite=True,
            objective="val_loss",
            # objective=Objective('val_euclidean_distance', direction='min'),
            max_epochs=210,
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
            tuner.search(
                slam, gt, validation_split=0.2, epochs=100, callbacks=self._callbacks
            )

        self._model_params = tuner.get_best_hyperparameters(num_trials=1)[0]
        self._model = hypermodel.build(self._model_params)
