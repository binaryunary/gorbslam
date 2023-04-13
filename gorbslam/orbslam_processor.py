from ctypes import ArgumentError
from enum import Enum
import os
from turtle import title

import numpy as np

from gorbslam.common.orbslam_results import ORBSLAMResults
from gorbslam.common.plotting_utils import (
    TraceColors,
    create_2d_fig,
    create_ape_fig_batch,
    create_ape_trace,
    create_map_fig,
    create_scatter,
    create_scattermapbox,
    create_slam_scatter,
)
from gorbslam.common.utils import (
    calculate_ape,
    create_trajectory_from_array,
    ensure_dir,
)

import plotly.express as px

from gorbslam.models import NNModel, GBRModel, RFRModel, SVRModel, UmeyamaModel


class ModelType(Enum):
    NN = 1
    GBR = 2
    RFR = 3
    SVR = 4
    UMEYAMA = 5


class ORBSLAMProcessor:
    def __init__(self, orbslam_results_dir: str, model_type: ModelType):
        self.orbslam_results_dir = os.path.expanduser(orbslam_results_dir)
        self.processed_results_dir = os.path.join(
            self.orbslam_results_dir, f"processed_{model_type.name.lower()}"
        )
        self.trajectory_name = os.path.basename(self.orbslam_results_dir)
        self.model_type = model_type

        ensure_dir(self.processed_results_dir)

        self.orbslam = ORBSLAMResults(self.orbslam_results_dir)

        if model_type == ModelType.NN:
            self.model = NNModel(self.processed_results_dir)
        elif model_type == ModelType.GBR:
            self.model = GBRModel(self.processed_results_dir)
        elif model_type == ModelType.RFR:
            self.model = RFRModel(self.processed_results_dir)
        elif model_type == ModelType.SVR:
            self.model = SVRModel(self.processed_results_dir)
        elif model_type == ModelType.UMEYAMA:
            self.model = UmeyamaModel(self.processed_results_dir)
        else:
            raise ArgumentError(f"Invalid model type: {model_type}")

    def initialize_model(self, overwrite=False):
        if overwrite or not self.model.load_model():
            self.model.create_model(self.orbslam.mapping, self.orbslam.localization[0])

    def fit_trajectories(self):
        self.orbslam.mapping.slam.fit(self.model.predict)
        for localization in self.orbslam.localization.values():
            localization.slam.fit(self.model.predict)

        # self.orbslam.mapping.trajectory_fitted_utm = self.model.predict(
        #     self.orbslam.mapping.trajectory
        # )
        # for localization in self.orbslam.localization.values():
        #     localization.trajectory_fitted_utm = self.model.predict(
        #         localization.trajectory
        #     )

    def save_trajectories(self):
        self.orbslam.mapping.save(self.processed_results_dir)
        for localization in self.orbslam.localization.values():
            localization.save(self.processed_results_dir)

    def create_map_plot(self):
        traces = [
            create_scattermapbox(
                self.orbslam.mapping.gt.wgs,
                "GPS (ground truth)",
                color=TraceColors.gt,
                mode="lines",
            ),
            create_scattermapbox(
                self.orbslam.mapping.slam.wgs,
                "fitted SLAM (mapping)",
                color=TraceColors.slam,
                mode="lines",
            ),
        ]

        for name, localization in self.orbslam.localization.items():
            traces.append(
                create_scattermapbox(
                    localization.slam.wgs,
                    f"fitted loc_{name}",
                    color=TraceColors.loc[name],
                )
            )

        center_lat = np.mean(self.orbslam.mapping.gt.wgs.lat)
        center_lon = np.mean(self.orbslam.mapping.gt.wgs.lon)

        return create_map_fig(traces, (center_lat, center_lon))

    def create_2d_plot_utm(self):
        traces = [
            create_scatter(
                self.orbslam.mapping.gt.utm,
                "GPS (ground truth)",
                color=TraceColors.gt,
                mode="markers",
            ),
            create_scatter(
                self.orbslam.mapping.slam.utm,
                "fitted SLAM (mapping)",
                color=TraceColors.slam,
                mode="markers",
            ),
        ]

        for name, localization in self.orbslam.localization.items():
            traces.append(
                create_scatter(
                    localization.slam.utm,
                    f"fitted loc_{name}",
                    color=TraceColors.loc[name],
                )
            )

        return create_2d_fig(traces, title="Trajectories in UTM coordinates")

    def create_2d_plot_slam(self):
        traces = [
            create_slam_scatter(
                self.orbslam.mapping.slam.slam,
                "SLAM (mapping)",
                color=TraceColors.slam,
                mode="lines",
            ),
        ]

        for name, localization in self.orbslam.localization.items():
            traces.append(
                create_slam_scatter(
                    localization.slam.slam,
                    f"SLAM loc_{name}",
                    color=TraceColors.loc[name],
                    mode="markers",
                )
            )

        return create_2d_fig(traces, title="Trajectories in SLAM coordinates")

    def create_ape_plot_all(self):
        traces = [
            create_ape_trace(
                self.orbslam.mapping.slam.utm, self.orbslam.mapping.gt.utm
            ),
        ]

        for name, localization in self.orbslam.localization.items():
            traces.append(create_ape_trace(localization.slam.utm, localization.gt.utm))

        subplot_titles = ["SLAM (fitted)"]
        for name, localization in self.orbslam.localization.items():
            subplot_titles.append(f"loc_{name}")

        fig = create_ape_fig_batch(
            traces,
            title=f"[{self.model_type.name}] Absolute Pose Error (APE)",
            subplot_titles=subplot_titles,
        )

        return fig
