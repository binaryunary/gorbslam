from gorbslam.common.slam_trajectory import read_localization_data, read_mapping_data


class ORBSLAMResults:
    def __init__(self, orbslam_results_dir):
        self.map_points, self.mapping = read_mapping_data(orbslam_results_dir)
        self.localization = read_localization_data(orbslam_results_dir)
