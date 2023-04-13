import json
from os import path

import joblib
from sklearn import ensemble
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from gorbslam.models.orbslam_corrector_model import ORBSLAMCorrectorModel


class SVRModel(ORBSLAMCorrectorModel):
    def __init__(self, model_dir):
        self._model_dir = model_dir
        self._model_path = path.join(self._model_dir, 'model.joblib')
        self._model_params_path = path.join(self._model_dir, 'model_params.json')
        self._source_scaler_path = path.join(self._model_dir, 'source_normalizer.joblib')
        self._target_scaler_path = path.join(self._model_dir, 'target_normalizer.joblib')
        self._source_scaler = None
        self._target_scaler = None
        self._model_params = None
        self._model = None
        self._is_loaded = False

    @property
    def model(self):
        return self._model

    @property
    def model_params(self):
        return self._model_params

    @property
    def is_loaded(self):
        return self._is_loaded

    def create_model(self, source_trajectory, target_trajectory, val_source_trajectory, val_target_trajectory):
        self._search_model(source_trajectory, target_trajectory, val_source_trajectory, val_target_trajectory)
        self.save_model()

    def save_model(self):
        joblib.dump(self._model, self._model_path)
        joblib.dump(self._source_scaler, self._source_scaler_path)
        joblib.dump(self._target_scaler, self._target_scaler_path)
        with open(self._model_params_path, 'w') as f:
            json.dump(self._model_params, f)

    def load_model(self):
        if not path.exists(self._model_path) or \
           not path.exists(self._source_scaler_path) or \
           not path.exists(self._target_scaler_path):
            self._is_loaded = False
        else:
            self._model = joblib.load(self._model_path)
            self._source_scaler = joblib.load(self._source_scaler_path)
            self._target_scaler = joblib.load(self._target_scaler_path)
            self._is_loaded = True

        return self._is_loaded

    def predict(self, slam_trajectory):
        slam_trajectory_normalized = self._source_scaler.transform(slam_trajectory)
        predicted_trajectory_normalized = self._model.predict(slam_trajectory_normalized)
        predicted_trajectory = self._target_scaler.inverse_transform(predicted_trajectory_normalized)

        return predicted_trajectory

    def _search_model(self, source_trajectory, target_trajectory, val_source_trajectory, val_target_trajectory):
        self._source_scaler = StandardScaler()
        self._source_scaler.fit(source_trajectory)

        self._target_scaler = StandardScaler()
        self._target_scaler.fit(target_trajectory)

        estimator = MultiOutputRegressor(SVR(verbose=1))

        # Define the hyperparameters grid for GradientBoostingRegressor
        param_grid = {
            "estimator__kernel": ["linear", "poly", "rbf"],
            "estimator__degree": [2, 3, 4],
            "estimator__C": [0.1, 1, 10],
            "estimator__epsilon": [0.1, 0.2, 0.3],
        }

        grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

        X_train = self._source_scaler.transform(source_trajectory)
        y_train = self._target_scaler.transform(target_trajectory)

        grid_search.fit(X_train, y_train)
        self._model_params = grid_search.best_params_

        print(f"Best BGR hyperparameters: {self._model_params}")

        # Retrieve the best model
        self._model = grid_search.best_estimator_
