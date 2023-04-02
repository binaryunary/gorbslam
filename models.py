from dataclasses import dataclass

from utils import utm2wgs, wgs2utm


@dataclass(frozen=True)
class GPSPos:
    lat: float
    lon: float
    alt: float


@dataclass(frozen=True)
class KeyFrame:
    timestamp: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass(frozen=True)
class MapPoint:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class WorldPoint:
    x: float
    y: float
    z: float


class SLAMTrajectory:
    def __init__(self, trajectory, trajectory_gt_wgs):
        self.trajectory = trajectory
        self._trajectory_utm = None
        self.trajectory_wgs = None
        self.trajectory_gt_wgs = trajectory_gt_wgs
        self.trajectory_gt_utm = wgs2utm(trajectory_gt_wgs)

    @property
    def trajectory_utm(self):
        return self._trajectory_utm

    @trajectory_utm.setter
    def trajectory_utm(self, value):
        self._trajectory_utm = value
        self.trajectory_wgs = utm2wgs(value)
