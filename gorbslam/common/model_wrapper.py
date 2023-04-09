from abc import ABC, abstractmethod

class ModelWrapper(ABC):

    @property
    def model(self):
        return self._model

    @abstractmethod
    def create_model(self, source_trajectory, target_trajectory, val_source_trajectory, val_target_trajectory):
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
