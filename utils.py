import numpy as np
import pyproj


def reshape_input(points: np.ndarray, n_inputs: int) -> np.ndarray:
    mod = points.size % n_inputs

    if mod != 0:
        rng = np.random.default_rng()
        n_rows = points.shape[0]
        rows_to_remove = mod // 3
        return np.delete(points, rng.choice(n_rows, rows_to_remove, replace=False), axis=0).reshape(-1, n_inputs)

    return np.array(points).reshape(-1, n_inputs)


def utm2wgs(trajectory_utm: np.ndarray) -> np.ndarray:
    # Create transformer for UTM35N -> WGS84
    utm2wgs = pyproj.Transformer.from_crs(32635, 4326)
    return np.array([utm2wgs.transform(p[0], p[1], p[2]) for p in trajectory_utm])


def wgs2utm(trajectory_wgs: np.ndarray) -> np.ndarray:
    # Create transformer for WGS84 -> UTM35N
    wgs2utm = pyproj.Transformer.from_crs(4326, 32635)
    return np.array([wgs2utm.transform(p[0], p[1], p[2]) for p in trajectory_wgs])
