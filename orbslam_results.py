import glob
from os import path

import numpy as np
import pandas as pd
import pyproj
from keras.layers import Dense, Normalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import L2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils import denormalize, reshape_data, umeyama_alignment


class GPSPos:
    def __init__(self, lat, lon, alt):
        self.lat = lat
        self.lon = lon
        self.alt = alt


class KeyFrame:
    def __init__(self, timestamp, x, y, z, qx, qy, qz, qw):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw


class MapPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class WorldPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class SLAMTrajectory:
    def __init__(self, trajectory, trajectory_gt_wgs):
        # Create transformers for WGS84 <-> UTM35N
        self.wgs2utm = pyproj.Transformer.from_crs(4326, 32635)
        self.utm2wgs = pyproj.Transformer.from_crs(32635, 4326)

        self.trajectory = trajectory
        self._trajectory_utm = None
        self.trajectory_wgs = None
        self.trajectory_gt_wgs = trajectory_gt_wgs
        self.trajectory_gt_utm = np.array([self.wgs2utm.transform(p[0], p[1], p[2])
                                           for p in self.trajectory_gt_wgs])

    @property
    def trajectory_utm(self):
        return self._trajectory_utm

    @trajectory_utm.setter
    def trajectory_utm(self, value):
        self._trajectory_utm = value
        self.trajectory_wgs = np.array([self.utm2wgs.transform(p[0], p[1], p[2])
                                           for p in value])

class ORBSLAMResults:
    def __init__(self, results_root):
        root = path.expanduser(results_root)
        self.map_points, self.mapping = read_mapping_data(root)
        self.localization = read_localization_trajectories(root)


class ORBSLAMTrajectoryProcessor:
    def __init__(self, orbslam_out_dir):
        self.orbslam = ORBSLAMResults(orbslam_out_dir)

        self.n_inputs = 3 * 1
        self._source_normalizer = None
        self._target_normalizer = None
        self._model = None

        # R, t, c = umeyama_alignment(self.orbslam_results.mapping.trajectory.T,
        #                             self.orbslam_results.mapping.trajectory_gt_utm.T,
        #                             True)

    def fit(self):
        self._fit_trajectory(self.orbslam.mapping.trajectory, self.orbslam.mapping.trajectory_gt_utm)
        self.orbslam.mapping.trajectory_utm = self.predict_trajectory(self.orbslam.mapping.trajectory)
        for localization in self.orbslam.localization.values():
            localization.trajectory_utm = self.predict_trajectory(localization.trajectory)

    def _fit_trajectory(self, source_points, target_points, epochs=400, batch_size=8):
        assert source_points.shape == target_points.shape

        print(f"Training model with {source_points.shape}")
        print(f"Target points: {target_points.shape}")

        callbacks = [
            EarlyStopping(monitor='loss', patience=10),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
        ]

        source = reshape_data(source_points, self.n_inputs)
        target = reshape_data(target_points, self.n_inputs)

        print("Resized source: ", source.shape)
        print("Resized target: ", target.shape)

        source_normalizer = Normalization(input_shape=(self.n_inputs,))
        source_normalizer.adapt(source)

        target_normalizer = Normalization(input_shape=(self.n_inputs,))
        target_normalizer.adapt(target)

        X = source_normalizer(source)
        y = target_normalizer(target)

        regularizer = L2(l2=0.001)

        n_nodes = 64

        # Define the neural network architecture
        model = Sequential()
        model.add(Dense(n_nodes, activation='tanh', input_shape=(self.n_inputs,)))
        # model.add(Dense(n_nodes, activation='tanh'))
        # model.add(Dense(n_nodes, activation='tanh'))
        model.add(Dense(self.n_inputs))

        optimizer = Adam()
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=True)

        self._model = model
        self._source_normalizer = source_normalizer
        self._target_normalizer = target_normalizer


    def predict_trajectory(self, source_points):
        print(f"Predicting target trajectory with {source_points.shape}")

        source = reshape_data(source_points, self.n_inputs)
        print(f"Resized source: {source.shape}")

        normalized_source_points = self._source_normalizer(source)

        X = normalized_source_points

        # Predict the target trajectory
        normalized_predicted_target_points = self._model.predict(X)
        denormalized_predicted_target_points = denormalize(self._target_normalizer, normalized_predicted_target_points)

        return denormalized_predicted_target_points.reshape(-1, 3)




#  TODO: Move below functions to ORBSLAMResults
def read_kf_trajectory(filename: str) -> np.array:
    trajectory = []
    with open(filename, 'r') as file:
        for line in file:
            kf = tuple(map(float, line.split()))
            trajectory.append(WorldPoint(*kf[1:4])) # skip timestamp, pick only x, y, z
    return np.array([(p.x, p.y, p.z) for p in trajectory])


def read_kf_trajectory_gt(filename: str) -> np.array:
    trajectory = []
    with open(filename, 'r') as file:
        for line in file:
          gps = tuple(map(float, line.split()))
          trajectory.append(GPSPos(*gps[1:])) # skip timestamp, pick only lat, lon, alt
    return np.array([(pos.lat, pos.lon, pos.alt) for pos in trajectory])


def read_trajectory(filename: str) -> np.array:
    path_points = []
    with open(filename, 'r') as file:
        path_points = [WorldPoint(*map(float, line.split())) for line in file]
    return np.array([(p.x, p.y, p.z) for p in path_points])


def read_trajectory_gt(filename: str) -> np.array:
    trajectory = []
    with open(filename, 'r') as file:
        for line in file:
          gps = tuple(map(float, line.split()))
          trajectory.append(GPSPos(*gps))
    return np.array([(pos.lat, pos.lon, pos.alt) for pos in trajectory])


def read_map_points(filename: str) -> list[WorldPoint]:
    return read_trajectory(filename)



# TODO: Deprecated
def read_gps_estimate(filename: str) -> list[GPSPos]:
    with open(filename, 'r') as f:
        return [GPSPos(*map(float, line.split())) for line in f]


def read_mapping_data(results_root):
    mapping_files = glob.glob('m_*_*.txt', root_dir=results_root)
    mapping_files.sort()
    mapping_data = {}
    map_sizes = {}
    # Sort to get triplets of map points, SLAM trajectories and ground truths
    # m_*_map_points.txt
    # m_*_trajectory.txt
    # m_*_trajectory_gt.txt
    chunk = 3
    for i in range(0, len(mapping_files), chunk):
        mps, t, t_gt = mapping_files[i:i+chunk]
        idx = int(t.split('_', 2)[1])

        map_points = read_map_points(path.join(results_root, mps))
        kf_trajectory = read_kf_trajectory(path.join(results_root, t))
        kf_trajectory_gt = read_kf_trajectory_gt(path.join(results_root, t_gt))

        map_size = kf_trajectory.shape[0] # number of KFs in a map
        mapping_data[idx] = (map_points, SLAMTrajectory(kf_trajectory, kf_trajectory_gt))
        map_sizes[map_size] = idx

    # Get the index of the largest map
    m = map_sizes[max(map_sizes.keys())]

    # For now just return the largest map and associated data
    return mapping_data[m]


def read_localization_trajectories(results_root):
    localization_files = glob.glob('l_*_*.txt', root_dir=results_root)
    # Sort to get pairs of estimates and ground truths
    # l_*_trajectory.txt
    # l_*_trajectory_gt.txt
    localization_files.sort()
    localization_data = {}

    chunk = 2
    for i in range(0, len(localization_files), chunk):
        t, t_gt = localization_files[i:i+chunk]
        idx = int(t.split('_', 2)[1])
        trajectory = read_trajectory(path.join(results_root, t))
        trajectory_gt = read_trajectory_gt(path.join(results_root, t_gt))
        localization_data[idx] = SLAMTrajectory(trajectory, trajectory_gt)

    return localization_data

