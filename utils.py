import typing
from os import path

import numpy as np
from pykalman import KalmanFilter
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


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


def read_gps_estimate(filename):
    with open(filename, 'r') as f:
        return [GPSPos(*map(float, line.split())) for line in f]


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


def lm_estimate_transform(A, B):
    def error_function(x, src, dst):
        # x[0:3] represents the translation vector
        translation = x[0:3]

        # x[3:6] represents the rotation vector (Rodrigues)
        rotation = R.from_rotvec(x[3:6])

        # x[6] represents the scale factor
        scale = x[6]

        # Apply the estimated rotation, translation, and scale to the source points
        transformed_src = scale * rotation.apply(src) + translation

        # Compute the difference (residual) between the transformed points and the destination points
        residuals = transformed_src - dst

        # Flatten the residuals to a 1D array
        return residuals.flatten()

    src_centroid = np.mean(A, axis=0)
    dst_centroid = np.mean(B, axis=0)
    src_points_normalized = A - src_centroid
    dst_points_normalized = B - dst_centroid
    src_scale = np.sqrt(np.sum(src_points_normalized**2) / len(src_points_normalized))
    dst_scale = np.sqrt(np.sum(dst_points_normalized**2) / len(dst_points_normalized))
    src_points_normalized /= src_scale
    dst_points_normalized /= dst_scale

    translation_init_guess = np.mean(B, axis=0) - np.mean(A, axis=0)
    rotation_init_guess = np.array([0, 0, 0])
    scale_init_guess = 1

    # initial guess
    # x0 = np.array([0, 0, 0, 0, 0, 0, 1])
    x0 = np.concatenate((translation_init_guess, rotation_init_guess, [scale_init_guess]), axis=None)
    result = least_squares(error_function, x0, args=(src_points_normalized, dst_points_normalized), method='lm')

    optimal_translation_normalized = result.x[0:3]
    optimal_rotation_normalized = R.from_rotvec(result.x[3:6])
    optimal_scale_normalized = result.x[6]

    # Extract the optimal translation, rotation, and scale parameters
    Rot = optimal_rotation_normalized
    c = optimal_scale_normalized * dst_scale / src_scale
    t = dst_centroid - c * Rot.apply(src_centroid) + optimal_translation_normalized * dst_scale

    return Rot.as_matrix(), c, t


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
