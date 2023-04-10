from os import path

import joblib
from sklearn import ensemble
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


from gorbslam.common.model_wrapper import ModelWrapper


class SVRModelWrapper(ModelWrapper):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model_path = path.join(self.model_dir, 'model.joblib')
        self.source_scaler_path = path.join(self.model_dir, 'source_normalizer.joblib')
        self.target_scaler_path = path.join(self.model_dir, 'target_normalizer.joblib')
        self.source_scaler = None
        self.target_scaler = None
        self.best_hps = None
        self._model = None

    def _search_model(self, source_trajectory, target_trajectory, val_source_trajectory, val_target_trajectory):
        self.source_scaler = StandardScaler()
        self.source_scaler.fit(source_trajectory)

        self.target_scaler = StandardScaler()
        self.target_scaler.fit(target_trajectory)

        estimator = MultiOutputRegressor(SVR(verbose=1))

        # Define the hyperparameters grid for GradientBoostingRegressor
        param_grid = {
            "estimator__kernel": ["linear", "poly", "rbf"],
            "estimator__degree": [2, 3, 4],
            "estimator__C": [0.1, 1, 10],
            "estimator__epsilon": [0.1, 0.2, 0.3],
        }

        grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

        X_train = self.source_scaler.transform(source_trajectory)
        y_train = self.target_scaler.transform(target_trajectory)

        grid_search.fit(X_train, y_train)
        self.best_hps = grid_search.best_params_

        print(f"Best BGR hyperparameters: {self.best_hps}")

        # Retrieve the best model
        self._model = grid_search.best_estimator_

    def create_model(self, source_trajectory, target_trajectory, val_source_trajectory, val_target_trajectory):
        self._search_model(source_trajectory, target_trajectory, val_source_trajectory, val_target_trajectory)
        self.save_model()

    def save_model(self):
        joblib.dump(self._model, self.model_path)
        joblib.dump(self.source_scaler, self.source_scaler_path)
        joblib.dump(self.target_scaler, self.target_scaler_path)

    def load_model(self):
        if not path.exists(self.model_path) or \
           not path.exists(self.source_scaler_path) or \
           not path.exists(self.target_scaler_path):
            return False

        self._model = joblib.load(self.model_path)
        self.source_scaler = joblib.load(self.source_scaler_path)
        self.target_scaler = joblib.load(self.target_scaler_path)

        return True

    def predict(self, slam_trajectory):
        slam_trajectory_normalized = self.source_scaler.transform(slam_trajectory)
        predicted_trajectory_normalized = self._model.predict(slam_trajectory_normalized)
        predicted_trajectory = self.target_scaler.inverse_transform(predicted_trajectory_normalized)

        return predicted_trajectory
