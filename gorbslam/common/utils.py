import json
import math

import numpy as np
import pyproj


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def reshape_input(points: np.ndarray, n_inputs: int) -> np.ndarray:
    mod = points.size % n_inputs

    if mod != 0:
        rng = np.random.default_rng()
        n_rows = points.shape[0]
        rows_to_remove = mod // 3
        return np.delete(points, rng.choice(n_rows, rows_to_remove, replace=False), axis=0).reshape(-1, n_inputs)

    return np.array(points).reshape(-1, n_inputs)


def downsample(data: np.ndarray, n: int, start=0) -> np.ndarray:
    downsampled_indices = np.linspace(start, data.shape[0] - 1, n, dtype=int)
    downsampled_data = data[downsampled_indices]
    return downsampled_data


def create_training_splits(training_data: tuple[np.ndarray, np.ndarray], extra_data: tuple[np.ndarray, np.ndarray], validation_split: float):
    n_training = training_data[0].shape[0]
    n_total = math.floor(n_training / (1 - validation_split))
    n_not_training = n_total - n_training
    n_validation = math.floor(n_not_training / 2)
    n_test = n_not_training- n_validation

    val_slam = downsample(extra_data[0], n_validation, 0)
    val_true = downsample(extra_data[1], n_validation, 0)

    test_slam = downsample(extra_data[0], n_test, 1)
    test_true = downsample(extra_data[1], n_test, 1)

    return training_data, (val_slam, val_true), (test_slam, test_true)


def utm2wgs(trajectory_utm: np.ndarray) -> np.ndarray:
    # Create transformer for UTM35N -> WGS84
    utm2wgs = pyproj.Transformer.from_crs(32635, 4326)
    return np.array([utm2wgs.transform(p[0], p[1], p[2]) for p in trajectory_utm])


def wgs2utm(trajectory_wgs: np.ndarray) -> np.ndarray:
    # Create transformer for WGS84 -> UTM35N
    wgs2utm = pyproj.Transformer.from_crs(4326, 32635)
    return np.array([wgs2utm.transform(p[0], p[1], p[2]) for p in trajectory_wgs])


def assert_trajectory_shape(trajectory: np.ndarray):
    assert trajectory.shape[1] == 3


def replace_tum_xyz(tum_data: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    assert_trajectory_shape(xyz)
    tum_data_copy = tum_data.copy()
    tum_data_copy[:, 1:4] = xyz
    return tum_data_copy


def read_tum_file(filename: str) -> np.ndarray:
    with open(filename, 'r') as file:
        return np.array([tuple(map(float, line.split())) for line in file])


def write_tum_file(filename: str, data: np.ndarray):
    with open(filename, 'w') as file:
        for row in data:
            file.write(' '.join(map(str, row)) + '\n')


def read_map_points(filename: str) -> np.ndarray:
    with open(filename, 'r') as file:
        return np.array([tuple(map(float, line.split())) for line in file])

