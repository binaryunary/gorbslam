from dataclasses import dataclass
from os import path


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
