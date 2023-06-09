from enum import Enum
import glob
from os import path

import numpy as np
import pandas as pd

from gorbslam.common.utils import (
    read_map_points,
    read_tum,
    read_tum_gt,
    utm2wgs,
    wgs2utm,
    write_tum_file,
    write_tum_file_df,
)


class TrackType(Enum):
    SLAM = 0
    WGS84 = 1
    UTM35N = 2


class Track:
    """
    An abstraction layer for a sequence of 3D points.
    This class allows to create a Track object from a SLAM trajectory, a WGS84 trajectory or a UTM35N trajectory,
    and to convert between these coordinate systems.
    """

    @staticmethod
    def fromSLAM(track_data: pd.DataFrame):
        """
        Creates a Track object from a SLAM trajectory.
        """
        return Track(track_data, TrackType.SLAM)

    @staticmethod
    def fromWGS84(track_data: pd.DataFrame):
        """
        Creates a Track object from a WGS84 trajectory.
        """
        return Track(track_data, TrackType.WGS84)

    @staticmethod
    def fromUTM35N(track_data: pd.DataFrame):
        """
        Creates a Track object from a UTM35N trajectory.
        """
        return Track(track_data, TrackType.UTM35N)

    def __init__(self, track_data: pd.DataFrame, track_type: TrackType):
        self._slam = None
        self._wgs = None
        self._utm = None

        if track_type == TrackType.SLAM:
            self._slam = track_data
        elif track_type == TrackType.WGS84:
            self._wgs = track_data
        elif track_type == TrackType.UTM35N:
            self._utm = track_data
        else:
            raise ValueError(f"Invalid track type {track_type}")

    @property
    def slam(self):
        """
        Returns the track in SLAM coordinates.
        """
        return self._slam

    @property
    def utm(self):
        """
        Returns the track in UTM35N coordinates.
        """
        if self._utm is None and self._wgs is None:
            raise ValueError(
                "Cannot convert to UTM35N without WGS84 data.\
                             If this is a SLAM track, make sure to run fit() first."
            )
        if self._utm is None:
            utm_coords = wgs2utm(self._wgs[["lat", "lon", "alt"]].to_numpy())
            utm_track = self._wgs.drop(["lat", "lon", "alt"], axis=1)
            # insert x, y, z
            utm_track.insert(1, "x", utm_coords[:, 0])
            utm_track.insert(2, "y", utm_coords[:, 1])
            utm_track.insert(3, "z", utm_coords[:, 2])
            self._utm = utm_track

        return self._utm

    @property
    def wgs(self):
        """
        Returns the track in WGS84 coordinates.
        """
        if self._wgs is None and self._utm is None:
            raise ValueError(
                "Cannot convert to WGS84 without UTM35N data.\
                             If this is a SLAM track, make sure to run fit() first."
            )
        if self._wgs is None:
            wgs_coords = utm2wgs(self._utm[["x", "y", "z"]].to_numpy())
            wgs_track = self._utm.drop(["x", "y", "z"], axis=1)
            # insert lat, lon, alt
            wgs_track.insert(1, "lat", wgs_coords[:, 0])
            wgs_track.insert(2, "lon", wgs_coords[:, 1])
            wgs_track.insert(3, "alt", wgs_coords[:, 2])
            self._wgs = wgs_track

        return self._wgs

    def fit(self, model_predictor):
        """
        Transforms the SLAM track to UTM35N coordinates using the given model.
        """
        predicted_utm = model_predictor(self._slam[["x", "y", "z"]].to_numpy())
        tmp = self._slam.copy()
        tmp[["x", "y", "z"]] = predicted_utm
        self._utm = tmp


class SLAMTrajectory:
    """
    A container for a SLAM trajectory and its ground truth.
    """

    def __init__(self, trajectory_tum, trajectory_tum_gt_wgs, trajectory_name):
        self._slam = Track.fromSLAM(trajectory_tum)
        self._gt = Track.fromWGS84(trajectory_tum_gt_wgs)
        self._trajectory_name = trajectory_name

    @property
    def slam(self):
        """
        Returns the SLAM track of the trajectory.
        """
        return self._slam

    @property
    def gt(self):
        """
        Returns the ground truth track of the trajectory.
        """
        return self._gt

    @property
    def trajectory_name(self):
        return self._trajectory_name

    def save(self, dirname: str):
        """
        Saves the SLAM trajectory and its ground truth to the given directory.
        """
        # Save the ground truth
        write_tum_file(
            path.join(dirname, f"{self.trajectory_name}_gt_tum.txt"),
            self.gt.utm,
        )

        write_tum_file_df(
            path.join(dirname, f"{self.trajectory_name}_fitted_utm.txt"), self.slam.utm
        )


def read_mapping_data(results_root) -> tuple[np.ndarray, SLAMTrajectory]:
    """
    Reads the mapping data from the given directory.
    ORB-SLAM3 may create multiple maps if it loses tracking, we just return the longest one.
    """
    mapping_files = glob.glob("m_*_*.txt", root_dir=results_root)
    mapping_files.sort()
    mapping_data = {}
    map_sizes = {}
    # Sort to get triplets of map points, SLAM trajectories and ground truths
    # m_*_map_points.txt
    # m_*_trajectory.txt
    # m_*_trajectory_gt.txt
    chunk = 3
    for i in range(0, len(mapping_files), chunk):
        mps, t, t_gt = mapping_files[i : i + chunk]
        idx = int(t.split("_", 2)[1])

        map_points = read_map_points(path.join(results_root, mps))
        kf_slam = read_tum(path.join(results_root, t))
        kf_gt = read_tum_gt(path.join(results_root, t_gt))

        map_size = kf_slam.shape[0]  # number of KFs in a map
        trajectory_name = path.splitext(t)[0]  # name of the trajectory file
        mapping_data[idx] = (
            map_points,
            SLAMTrajectory(kf_slam, kf_gt, trajectory_name),
        )
        map_sizes[map_size] = idx

    # Get the index of the largest map
    m = map_sizes[max(map_sizes.keys())]

    # For now just return the largest map and associated data
    return mapping_data[m]


def read_localization_data(results_root) -> dict[int, SLAMTrajectory]:
    """
    Reads all localization files (l_*_trajectory.txt, l_*_trajectory_gt.txt) from the given directory.
    """
    localization_files = glob.glob("l_*_*.txt", root_dir=results_root)
    # Sort to get pairs of estimates and ground truths
    # l_*_trajectory.txt
    # l_*_trajectory_gt.txt
    localization_files.sort()
    localization_data = {}

    chunk = 2
    for i in range(0, len(localization_files), chunk):
        t, t_gt = localization_files[i : i + chunk]
        idx = int(t.split("_", 2)[1])
        slam = read_tum(path.join(results_root, t))
        gt = read_tum_gt(path.join(results_root, t_gt))
        trajectory_name = path.splitext(t)[0]  # name of the trajectory file
        localization_data[idx] = SLAMTrajectory(slam, gt, trajectory_name)

    return localization_data
