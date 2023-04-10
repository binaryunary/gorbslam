import os

import numpy as np
from gorbslam.common.linear_transforms import linear_transform, umeyama_alignment
from gorbslam.common.orbslam_results import ORBSLAMResults
from gorbslam.common.plotting_utils import TraceColors, create_2d_fig, create_map_fig, create_scatter, create_scattermapbox, create_slam_scatter

from gorbslam.common.slam_trajectory import read_localization_data, read_mapping_data
from gorbslam.common.utils import ensure_dir
from gorbslam.gbr.gbr_model_wrapper import GBRModelWrapper
from gorbslam.gbr.rfr_model_wrapper import RFRModelWrapper
from gorbslam.gbr.svr_model_wrapper import SVRModelWrapper


class ORBSLAMProcessor:
    def __init__(self, orbslam_results_dir, model_type='gbr'):
        self.orbslam_results_dir = os.path.expanduser(orbslam_results_dir)
        self.processed_results_dir = os.path.join(self.orbslam_results_dir, f'processed_{model_type}')
        self.trajectory_name = os.path.basename(self.orbslam_results_dir)

        ensure_dir(self.processed_results_dir)

        self.orbslam = ORBSLAMResults(self.orbslam_results_dir)

        if model_type == 'gbr':
            self.model = GBRModelWrapper(self.processed_results_dir)
        elif model_type == 'rfr':
            self.model = RFRModelWrapper(self.processed_results_dir)
        elif model_type == 'svr':
            self.model = SVRModelWrapper(self.processed_results_dir)
        else:
            raise ValueError(f'Invalid model type: {model_type}')


        self._scale_align_trajectories()

    def _scale_align_trajectories(self):
        R, t, c = umeyama_alignment(self.orbslam.mapping.trajectory.T,
                                    self.orbslam.mapping.trajectory_gt_utm.T, True)
        self.orbslam.mapping.trajectory_scaled_utm = linear_transform(self.orbslam.mapping.trajectory, R, t, c)
        for localization in self.orbslam.localization.values():
            localization.trajectory_scaled_utm = linear_transform(localization.trajectory, R, t, c)

    def initialize_model(self, overwrite=False):
        if overwrite or not self.model.load_model():
            self.model.create_model(self.orbslam.mapping.trajectory, self.orbslam.mapping.trajectory_gt_utm,
                                    self.orbslam.localization[0].trajectory, self.orbslam.localization[0].trajectory_gt_utm)

    def fit_trajectories(self):
        self.orbslam.mapping.trajectory_fitted_utm = self.model.predict(self.orbslam.mapping.trajectory)
        for localization in self.orbslam.localization.values():
            localization.trajectory_fitted_utm = self.model.predict(localization.trajectory)

    def save_trajectories(self):
        self.orbslam.mapping.save(self.processed_results_dir)
        for localization in self.orbslam.localization.values():
            localization.save(self.processed_results_dir)

    def create_map_plot(self):
        traces = [
            create_scattermapbox(self.orbslam.mapping.trajectory_gt_wgs,
                                 'GPS (ground truth)', color=TraceColors.gt, mode='lines'),
            create_scattermapbox(self.orbslam.mapping.trajectory_fitted_wgs,
                                 'fitted SLAM (mapping)', color=TraceColors.slam, mode='lines'),
            create_scattermapbox(self.orbslam.mapping.trajectory_scaled_wgs,
                                 'Umeyama SLAM (mapping)', color=TraceColors.slam_scaled, mode='lines'),
        ]

        for name, localization in self.orbslam.localization.items():
            traces.append(create_scattermapbox(localization.trajectory_fitted_wgs,
                          f'fitted loc_{name}', color=TraceColors.loc[name]))

        center_lat = np.mean(self.orbslam.mapping.trajectory_gt_wgs[:, 0])
        center_lon = np.mean(self.orbslam.mapping.trajectory_gt_wgs[:, 1])

        return create_map_fig(traces, (center_lat, center_lon))

    def create_2d_plot_utm(self):
        traces = [
            create_scatter(self.orbslam.mapping.trajectory_gt_utm,
                           'GPS (ground truth)', color=TraceColors.gt, mode='markers'),
            create_scatter(self.orbslam.mapping.trajectory_fitted_utm,
                           'fitted SLAM (mapping)', color=TraceColors.slam, mode='markers'),
            create_scatter(self.orbslam.mapping.trajectory_scaled_utm, 'Umeyama SLAM (mapping)',
                           color=TraceColors.slam_scaled, mode='lines'),
        ]

        for name, localization in self.orbslam.localization.items():
            traces.append(create_scatter(localization.trajectory_fitted_utm,
                          f'fitted loc_{name}', color=TraceColors.loc[name]))

        return create_2d_fig(traces, title='Trajectories in UTM coordinates')

    def create_2d_plot_slam(self):
        traces = [
            create_slam_scatter(self.orbslam.mapping.trajectory, 'SLAM (mapping)',
                                color=TraceColors.slam, mode='lines'),
        ]

        for name, localization in self.orbslam.localization.items():
            traces.append(create_slam_scatter(localization.trajectory,
                          f'SLAM loc_{name}', color=TraceColors.loc[name], mode='markers'))

        return create_2d_fig(traces, title='Trajectories in SLAM coordinates')
