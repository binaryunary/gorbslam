from os import path

from file_utils import read_localization_trajectories, read_mapping_data
from slam_hypermodel import SLAMHyperModel


class ORBSLAMResults:
    def __init__(self, orbslam_out_dir):
        root = path.expanduser(orbslam_out_dir)
        self.map_points, self.mapping = read_mapping_data(root)
        self.localization = read_localization_trajectories(root)


class ORBSLAMTrajectoryProcessor:
    def __init__(self, orbslam_out_dir):
        self.orbslam = ORBSLAMResults(orbslam_out_dir)
        self.trajectory_name = path.basename(orbslam_out_dir)
        self.model = SLAMHyperModel(self.trajectory_name)

        # R, t, c = umeyama_alignment(self.orbslam_results.mapping.trajectory.T,
        #                             self.orbslam_results.mapping.trajectory_gt_utm.T,
        #                             True)

    def build_model(self):
        self.model.adapt_normalizers(self.orbslam.mapping.trajectory, self.orbslam.mapping.trajectory_gt_utm)
        self.model.search(self.orbslam.mapping.trajectory, self.orbslam.mapping.trajectory_gt_utm)
        self.model.train(self.orbslam.mapping.trajectory, self.orbslam.mapping.trajectory_gt_utm)

    def correct_trajectories(self):
        # self._fit_trajectory(self.orbslam.mapping.trajectory, self.orbslam.mapping.trajectory_gt_utm)
        self.orbslam.mapping.trajectory_utm = self.model.predict(self.orbslam.mapping.trajectory)
        for localization in self.orbslam.localization.values():
            localization.trajectory_utm = self.model.predict(localization.trajectory)
