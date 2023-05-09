from abc import ABC, abstractmethod

import numpy as np

from gorbslam.common.slam_trajectory import SLAMTrajectory


class ORBSLAMCorrectorModel(ABC):
    """
    Abstraction layer for ORB-SLAM corrector models that transform a trajectory in SLAM coordinates into a
    trajectory in UTM coordinates.
    """

    @property
    @abstractmethod
    def model(self):
        """
        Returns the underlying model.
        """
        pass

    @property
    @abstractmethod
    def model_params(self):
        """
        Returns the model parameters.
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self):
        """
        Returns whether the model is loaded.
        """
        pass

    @abstractmethod
    def create_model(
        self,
        training_data: SLAMTrajectory,
        validation_data: SLAMTrajectory = None,
    ):
        """
        Creates the model, actual implementation depends on the model type.
        :param training_data: Training data. Mapping trajectory with corresponding ground truth.
        :param validation_data: Validation data. Self-localization trajectory with corresponding ground truth.
        """
        pass

    @abstractmethod
    def save_model(self):
        """
        Saves the model to disk.
        """
        pass

    @abstractmethod
    def load_model(self):
        """
        Loads the model from disk.
        """
        pass

    @abstractmethod
    def predict(self, slam_trajectory: np.ndarray) -> np.ndarray:
        """
        Transform a trajectory in SLAM coordinates into a trajectory in UTM coordinates.
        :param slam_trajectory: Trajectory in SLAM coordinates, (n, 3) array of (x, y, z) trajectory points.
        :return: Trajectory in UTM coordinates, (n, 3) array of (x', y', z') trajectory points.
        """
        pass
