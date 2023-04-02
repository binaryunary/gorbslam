from os import path

import numpy as np

from file_utils import read_localization_trajectories, read_mapping_data
from linear_transforms import linear_transform, umeyama_alignment
from plotting_utils import TraceColors, create_2d_fig, create_map_fig, create_scatter, create_scattermapbox
from slam_hypermodel import SLAMHyperModel


class ORBSLAMResults:
    def __init__(self, orbslam_out_dir):
        root = path.expanduser(orbslam_out_dir)
        self.map_points, self.mapping = read_mapping_data(root)
        self.localization = read_localization_trajectories(root)


class ORBSLAMProcessor:
    def __init__(self, orbslam_out_dir):
        self.orbslam = ORBSLAMResults(orbslam_out_dir)
        self.trajectory_name = path.basename(orbslam_out_dir)
        self.model = SLAMHyperModel(self.trajectory_name)

        R, t, c = umeyama_alignment(self.orbslam.mapping.trajectory.T,
                                    self.orbslam.mapping.trajectory_gt_utm.T, True)
        self.orbslam.mapping.trajectory_scaled_utm = linear_transform(self.orbslam.mapping.trajectory, R, t, c)
        for localization in self.orbslam.localization.values():
            localization.trajectory_scaled_utm = linear_transform(localization.trajectory, R, t, c)

    def build_model(self):
        self.model.adapt_normalizers(self.orbslam.mapping.trajectory, self.orbslam.mapping.trajectory_gt_utm)
        self.model.search(self.orbslam.mapping.trajectory, self.orbslam.mapping.trajectory_gt_utm)
        self.model.train(self.orbslam.mapping.trajectory, self.orbslam.mapping.trajectory_gt_utm)

    def fit_trajectories(self):
        self.orbslam.mapping.trajectory_fitted_utm = self.model.predict(self.orbslam.mapping.trajectory)
        for localization in self.orbslam.localization.values():
            localization.trajectory_fitted_utm = self.model.predict(localization.trajectory)

    def create_map_plot(self):
        traces = [
            create_scattermapbox(self.orbslam.mapping.trajectory_gt_wgs,
                                 'GPS (ground truth)', color=TraceColors.gt, mode='lines'),
            create_scattermapbox(self.orbslam.mapping.trajectory_fitted_wgs,
                                 'fitted SLAM (mapping)', color=TraceColors.slam, mode='lines'),
            create_scattermapbox(self.orbslam.mapping.trajectory_scaled_wgs, 'Umeyama SLAM (mapping)', color=TraceColors.slam_scaled, mode='lines'),
        ]

        for name, localization in self.orbslam.localization.items():
            traces.append(create_scattermapbox(localization.trajectory_fitted_wgs,
                          f'fitted loc_{name}', color=TraceColors.loc[name]))

        center_lat = np.mean(self.orbslam.mapping.trajectory_gt_wgs[:, 0])
        center_lon = np.mean(self.orbslam.mapping.trajectory_gt_wgs[:, 1])

        return create_map_fig(traces, (center_lat, center_lon))

    def create_2d_plot(self):
        traces = [
            create_scatter(self.orbslam.mapping.trajectory_gt_utm, 'GPS (ground truth)', color=TraceColors.gt, mode='lines'),
            create_scatter(self.orbslam.mapping.trajectory_fitted_utm, 'fitted SLAM (mapping)', color=TraceColors.slam, mode='lines'),
            create_scatter(self.orbslam.mapping.trajectory_scaled_utm, 'Umeyama SLAM (mapping)', color=TraceColors.slam_scaled, mode='lines'),
        ]

        for name, localization in self.orbslam.localization.items():
            traces.append(create_scatter(localization.trajectory_fitted_utm,
                          f'fitted loc_{name}', color=TraceColors.loc[name]))

        return create_2d_fig(traces)
