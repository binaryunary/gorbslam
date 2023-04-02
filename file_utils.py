import glob
from os import path
import numpy as np
from models import GPSPos, MapPoint, SLAMTrajectory, WorldPoint


def read_kf_trajectory(filename: str) -> np.array:
    trajectory = []
    with open(filename, 'r') as file:
        for line in file:
            kf = tuple(map(float, line.split()))
            trajectory.append(WorldPoint(*kf[1:4]))  # skip timestamp, pick only x, y, z
    return np.array([(p.x, p.y, p.z) for p in trajectory])


def read_kf_trajectory_gt(filename: str) -> np.array:
    trajectory = []
    with open(filename, 'r') as file:
        for line in file:
          gps = tuple(map(float, line.split()))
          trajectory.append(GPSPos(*gps[1:]))  # skip timestamp, pick only lat, lon, alt
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


def read_mapping_data(results_root) -> tuple[list[MapPoint], SLAMTrajectory]:
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

        map_size = kf_trajectory.shape[0]  # number of KFs in a map
        mapping_data[idx] = (map_points, SLAMTrajectory(kf_trajectory, kf_trajectory_gt))
        map_sizes[map_size] = idx

    # Get the index of the largest map
    m = map_sizes[max(map_sizes.keys())]

    # For now just return the largest map and associated data
    return mapping_data[m]


def read_localization_trajectories(results_root) -> dict[int, SLAMTrajectory]:
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
