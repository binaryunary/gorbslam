from dataclasses import dataclass
import json
from os import path
import numpy as np
import pandas as pd
from gorbslam.common.linear_transforms import umeyama_alignment
from gorbslam.common.slam_trajectory import SLAMTrajectory
from gorbslam.common.utils import CustomJSONEncoder, to_xyz
from gorbslam.models.orbslam_corrector_model import ORBSLAMCorrectorModel


@dataclass(frozen=True)
class UmeyamaParams:
    R: np.ndarray
    t: np.ndarray
    c: float


class UmeyamaModel(ORBSLAMCorrectorModel):
    def __init__(self, model_dir):
        self._model_dir = model_dir
        self._model_params_path = path.join(self._model_dir, "model_params.json")
        self._model_params: UmeyamaParams = None
        self._is_loaded = False

    @property
    def model(self):
        return None

    @property
    def model_params(self):
        return self._model_params

    @property
    def is_loaded(self):
        return self._is_loaded

    def create_model(
        self, training_data: SLAMTrajectory, validation_data: SLAMTrajectory = None
    ):
        source_trajectory = to_xyz(training_data.slam.slam)
        target_trajectory = to_xyz(training_data.gt.utm)

        R, t, c = umeyama_alignment(
            source_trajectory.T,
            target_trajectory.T,
            True,
        )
        self._model_params = UmeyamaParams(R, t, c)
        self.save_model()

    def save_model(self):
        with open(self._model_params_path, "w") as f:
            json.dump(self._model_params, f, cls=CustomJSONEncoder)

    def load_model(self):
        if not path.exists(self._model_params_path):
            self._is_loaded = False
        else:
            with open(self._model_params_path, "r") as f:
                self._model_params = json.loads(
                    f.read(),
                    object_hook=lambda d: UmeyamaParams(
                        R=np.array(d["R"]),
                        t=np.array(d["t"]),
                        c=d["c"],
                    ),
                )
                self._is_loaded = True

        return self._is_loaded

    def predict(self, slam_trajectory):
        R = self._model_params.R
        t = self._model_params.t
        c = self._model_params.c

        return np.array([t + c * R @ p for p in slam_trajectory])
