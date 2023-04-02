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
        self._trajectory_fitted_utm = None
        self.trajectory_fitted_wgs = None
        self._trajectory_scaled_utm = None
        self.trajectory_scaled_wgs = None
        self.trajectory_gt_wgs = trajectory_gt_wgs
        self.trajectory_gt_utm = wgs2utm(trajectory_gt_wgs)

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
