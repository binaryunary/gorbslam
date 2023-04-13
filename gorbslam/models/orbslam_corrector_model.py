from abc import ABC, abstractmethod


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
        source_trajectory,
        target_trajectory,
        val_source_trajectory,
        val_target_trajectory,
    ):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, slam_trajectory):
        pass
