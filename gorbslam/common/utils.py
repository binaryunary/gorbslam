import dataclasses
import json
import math
import os

import numpy as np
import pandas as pd
import pyproj

from evo.core import metrics, sync, trajectory


class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that can handle numpy arrays and dataclasses.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return json.JSONEncoder.default(self, obj)


def downsample(data: np.ndarray, n: int, start=0) -> np.ndarray:
    """
    Uniformly downsamples an array of lenght m to an array of length n.

    :param data: The array to downsample.
    :param n: The length of the downsampled array.
    :param start: The index of the first element to include in the downsampled array.
    """
    downsampled_indices = np.linspace(start, data.shape[0] - 1, n, dtype=int)
    downsampled_data = data[downsampled_indices]
    return downsampled_data


def create_training_splits(
    training_data: tuple[np.ndarray, np.ndarray],
    extra_data: tuple[np.ndarray, np.ndarray],
    validation_split: float,
):
    """
    Creates training, validation and test splits from the given data.
    It treats training_data as the validation_split of the total data and will create the validation and test splits from extra_data.
    """
    n_training = training_data[0].shape[0]
    n_total = math.floor(n_training / (1 - validation_split))
    n_not_training = n_total - n_training
    n_validation = math.floor(n_not_training / 2)
    n_test = n_not_training - n_validation

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


def read_tum(file_path: str) -> pd.DataFrame:
    """
    Reads a TUM file and returns a pandas DataFrame with the data.
    """
    return pd.read_csv(
        file_path,
        names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"],
        delim_whitespace=True,
    )


def read_tum_gt(file_path: str) -> pd.DataFrame:
    """
    Reads a TUM-like ground truth file where x, y, z are replaced with lat, lon, alt,
    and returns a pandas DataFrame with the data.
    """
    return pd.read_csv(
        file_path,
        names=["timestamp", "lat", "lon", "alt", "qx", "qy", "qz", "qw"],
        delim_whitespace=True,
    )


def write_tum_file(filename: str, data: np.ndarray):
    """
    Writes a TUM file from a numpy array.
    """
    with open(filename, "w") as file:
        for row in data:
            file.write(" ".join(map(str, row)) + "\n")


def write_tum_file_df(file_path: str, df: pd.DataFrame):
    """
    Writes a TUM file from a pandas DataFrame.
    """
    df.to_csv(file_path, sep=" ", header=False, index=False)


def read_map_points(file_path: str) -> pd.DataFrame:
    """
    Reads a Map Points file and returns a pandas DataFrame with the data.
    """
    return pd.read_csv(file_path, names=["x", "y", "z"], delim_whitespace=True)


def ensure_dir(dirpath: str):
    """
    Creates a directory if it does not exist.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def to_xyz(tum_df: pd.DataFrame) -> np.ndarray:
    """
    Converts a TUM DataFrame to a numpy array of shape (n, 3) containing only the x, y, z columns.
    """
    return tum_df[["x", "y", "z"]].to_numpy()


def create_pose_trajectory(data: np.ndarray) -> trajectory.PoseTrajectory3D:
    """
    Converts a numpy array of shape (n, 8) containing TUM data to a PoseTrajectory3D.
    """
    stamps = data[:, 0]  # n x 1
    xyz = data[:, 1:4]  # n x 3
    quat = data[:, 4:]  # n x 4
    quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column

    return trajectory.PoseTrajectory3D(xyz, quat, stamps)


def calculate_ape(
    trajectory: trajectory.PoseTrajectory3D, trajectory_gt: trajectory.PoseTrajectory3D
) -> metrics.APE:
    """
    Calculates the Absolute Pose Error (APE) between two trajectories.
    """
    # Synchronize the two trajectories based on timestamps
    max_diff = 0.01  # Maximum timestamp difference for synchronization (e.g., 10 ms)
    synced_ground_truth_traj, synced_estimated_traj = sync.associate_trajectories(
        trajectory_gt, trajectory, max_diff
    )

    # Calculate Absolute Pose Error (APE)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((synced_ground_truth_traj, synced_estimated_traj))

    return ape_metric
