from gorbslam.common.slam_trajectory import read_localization_data, read_mapping_data


class ORBSLAMResults:
    """
    This class provides an abstraction over ORB-SLAM mapping and localization results for a single sequence.
    It stores the mapping and localization data and offers methods to read and process these results.

    :property map_points: Map Points obtained during mapping, 3D point cloud.
    :property mapping: Mapping trajectory in SLAM coordinates and corresponding ground truth.
    :property localization: Self-localization trajectory in SLAM coordinates and corresponding ground truth,
              and other localizations using the same map and their corresponding ground truths.
    """

    def __init__(self, orbslam_results_dir):
        self.map_points, self.mapping = read_mapping_data(orbslam_results_dir)
        self.localization = read_localization_data(orbslam_results_dir)
