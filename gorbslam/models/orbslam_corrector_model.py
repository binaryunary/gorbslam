from abc import ABC, abstractmethod

import numpy as np

from gorbslam.common.slam_trajectory import SLAMTrajectory


class ORBSLAMCorrectorModel(ABC):
    @property
    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def model_params(self):
        pass

    @property
    @abstractmethod
    def is_loaded(self):
        pass

    @abstractmethod
    def create_model(
        self,
        training_data: SLAMTrajectory,
        validation_data: SLAMTrajectory = None,
    ):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, slam_trajectory: np.ndarray) -> np.ndarray:
        pass
