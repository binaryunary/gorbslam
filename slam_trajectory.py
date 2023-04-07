import glob
from os import path

import numpy as np
from utils import downsample, read_map_points, read_tum_file, write_tum_file
from utils import replace_tum_xyz, utm2wgs, wgs2utm


class SLAMTrajectory:
    def __init__(self, trajectory_tum, trajectory_tum_gt_wgs, trajectory_name):
        self._trajectory_tum = trajectory_tum
        self._trajectory_tum_gt_wgs = trajectory_tum_gt_wgs
        self.trajectory_name = trajectory_name
        self.trajectory = trajectory_tum[:, 1:4] # pick only x, y, z - skip everything else
        self._trajectory_fitted_utm = None
        self.trajectory_fitted_wgs = None
        self._trajectory_scaled_utm = None
        self.trajectory_scaled_wgs = None
        self.trajectory_gt_wgs = trajectory_tum_gt_wgs[:, 1:4] # pick only x, y, z - skip everything else
        self.trajectory_gt_utm = wgs2utm(self.trajectory_gt_wgs)

    @property
    def trajectory_fitted_utm(self):
        return self._trajectory_fitted_utm

    @trajectory_fitted_utm.setter
    def trajectory_fitted_utm(self, value):
        self._trajectory_fitted_utm = value
        self.trajectory_fitted_wgs = utm2wgs(value)

    @property
    def trajectory_scaled_utm(self):
        return self._trajectory_scaled_utm

    @trajectory_scaled_utm.setter
    def trajectory_scaled_utm(self, value):
        self._trajectory_scaled_utm = value
        self.trajectory_scaled_wgs = utm2wgs(value)

    def save(self, dirname: str):
        # Save the ground truth
        write_tum_file(path.join(dirname, f'{self.trajectory_name}_gt_utm.txt'),
                       replace_tum_xyz(self._trajectory_tum_gt_wgs, self.trajectory_gt_utm))

        # Save the fitted trajectory
        if self.trajectory_fitted_utm is not None:
            write_tum_file(path.join(dirname, f'{self.trajectory_name}_fitted_utm.txt'),
                           replace_tum_xyz(self._trajectory_tum, self.trajectory_fitted_utm))

        # Save the scaled trajectory
        if self.trajectory_scaled_utm is not None:
            write_tum_file(path.join(dirname, f'{self.trajectory_name}_scaled_utm.txt'),
                           replace_tum_xyz(self._trajectory_tum, self.trajectory_scaled_utm))



def read_mapping_data(results_root) -> tuple[np.ndarray, SLAMTrajectory]:
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
        kf_trajectory = read_tum_file(path.join(results_root, t))
        kf_trajectory_gt = read_tum_file(path.join(results_root, t_gt))

        map_size = kf_trajectory.shape[0]  # number of KFs in a map
        trajectory_name = path.splitext(t)[0] # name of the trajectory file
        mapping_data[idx] = (map_points, SLAMTrajectory(kf_trajectory, kf_trajectory_gt, trajectory_name))
        map_sizes[map_size] = idx

    # Get the index of the largest map
    m = map_sizes[max(map_sizes.keys())]

    # For now just return the largest map and associated data
    return mapping_data[m]


def read_localization_data(results_root) -> dict[int, SLAMTrajectory]:
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
        trajectory = read_tum_file(path.join(results_root, t))
        trajectory_gt = read_tum_file(path.join(results_root, t_gt))
        trajectory_name = path.splitext(t)[0] # name of the trajectory file
        localization_data[idx] = SLAMTrajectory(trajectory, trajectory_gt, trajectory_name)

    return localization_data
