import typing
import numpy as np
from pykalman import KalmanFilter


class GPSPos:
    def __init__(self, lat, lon, alt):
        self.lat = lat
        self.lon = lon
        self.alt = alt


class KeyFrame:
    def __init__(self, timestamp, gps, x, y, z, qx, qy, qz, qw):
        self.timestamp = timestamp
        self.gps = gps
        self.x = x
        self.y = z
        self.z = y * -1
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw


class MapPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = z
        self.z = y * -1


def read_keyframe_trajectory(keyframe_file, gps_file):
    """Reads a keyframe trajectory from a file.
    Args:
        filename: The name of the file to read.
    Returns:
        A list of (lat, lon) tuples.
    """
    keyframes = []
    with open(keyframe_file, 'r') as kf_file, open(gps_file, 'r') as gps_file:
        for (kf_line, gps_line) in zip(kf_file, gps_file):
            kf = tuple(map(float, kf_line.split()))
            gps = tuple(map(float, gps_line.split()))
            keyframes.append(KeyFrame(kf[0], GPSPos(gps[1], gps[2], gps[3]),
                             kf[1], kf[2], kf[3], kf[4], kf[5], kf[6], kf[7]))
    return keyframes


def read_map_points(filename):
    """Reads a list of map points from a file.
    Args:
        filename: The name of the file to read.
    Returns:
        A list of (lat, lon) tuples.
    """
    with open(filename, 'r') as f:
        return [MapPoint(*map(float, line.split())) for line in f]


def smooth_elevation(elevations):
    # Set up the Kalman filter
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=0.01)

    observation_matrix = np.reshape(elevations, (len(elevations), 1))
    measurement_noise = np.ones_like(observation_matrix)

    # Run the Kalman filter
    filtered_state_means, filtered_state_covariances = kf.filter(observation_matrix)

    # # Get the smoothed estimates
    smoothed_state_means, smoothed_state_covariances = kf.smooth(elevations)

    return (smoothed_state_means[:, 0], filtered_state_means[:, 0])

# https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/


def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t


UmeyamaResult = typing.Tuple[np.ndarray, np.ndarray, float]


# https://github.com/MichaelGrupp/evo/blob/051e5bf63195172af58dc8256cc71618f079f224/evo/core/geometry.py#L35
def umeyama_alignment(x: np.ndarray, y: np.ndarray,
                      with_scale: bool = False) -> UmeyamaResult:
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        raise GeometryException("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise GeometryException("Degenerate covariance rank, "
                                "Umeyama alignment is not possible")

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c
