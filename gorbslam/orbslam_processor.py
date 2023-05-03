import os
from ctypes import ArgumentError
from enum import Enum
from plotly.subplots import make_subplots

import numpy as np
from pyrsistent import inc

from gorbslam.common.orbslam_results import ORBSLAMResults
from gorbslam.common.plotting_utils import (
    TraceColors,
    create_2d_fig,
    create_3d_fig,
    create_3d_scatter,
    create_ape_fig,
    create_ape_fig_batch,
    create_ape_trace,
    create_map_fig,
    create_2d_scatter,
    create_scattermapbox,
    create_slam_2d_scatter,
)
from gorbslam.common.utils import calculate_ape, create_pose_trajectory, ensure_dir
from gorbslam.models import GBRModel, FCNNModel, RFRModel, SVRModel, UmeyamaModel
from gorbslam.models.fcnn_clr_model import FCNNCLRModel


class ModelType(Enum):
    FCNN = 1
    FCNN_CLR = 2
    GBR = 3
    RFR = 4
    SVR = 5
    UMEYAMA = 6


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

        if model_type == ModelType.FCNN:
            self.model = FCNNModel(self.processed_results_dir)
        elif model_type == ModelType.FCNN_CLR:
            self.model = FCNNCLRModel(self.processed_results_dir)
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

    def save_trajectories(self):
        self.orbslam.mapping.save(self.processed_results_dir)
        for localization in self.orbslam.localization.values():
            localization.save(self.processed_results_dir)

    def calculate_ape_metric_all(self):
        ape_errors = {}
        ape_errors["mapping"] = calculate_ape(
            create_pose_trajectory(self.orbslam.mapping.slam.utm.to_numpy()),
            create_pose_trajectory(self.orbslam.mapping.gt.utm.to_numpy()),
        )
        for name, localization in self.orbslam.localization.items():
            ape_errors[name] = calculate_ape(
                create_pose_trajectory(localization.slam.utm.to_numpy()),
                create_pose_trajectory(localization.gt.utm.to_numpy()),
            )

        return ape_errors

    def calculate_ape_stats_all(self):
        all_ape = self.calculate_ape_metric_all()
        all_ape_errors = [e for ape in all_ape.values() for e in ape.error]
        return {
            "mean": np.mean(all_ape_errors),
            "median": np.median(all_ape_errors),
            "std": np.std(all_ape_errors),
            "min": np.min(all_ape_errors),
            "max": np.max(all_ape_errors),
        }

    def create_map_plot(self):
        traces = [
            create_scattermapbox(
                self.orbslam.mapping.gt.wgs,
                "GPS (ground truth)",
                color=TraceColors.gt,
                mode="lines",
                bold=True,
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
            create_2d_scatter(
                self.orbslam.mapping.gt.utm,
                "GPS (ground truth)",
                color=TraceColors.gt,
                mode="markers",
            ),
            create_2d_scatter(
                self.orbslam.mapping.slam.utm,
                "fitted SLAM (mapping)",
                color=TraceColors.slam,
                mode="markers",
            ),
        ]

        for name, localization in self.orbslam.localization.items():
            traces.append(
                create_2d_scatter(
                    localization.slam.utm,
                    f"fitted loc_{name}",
                    color=TraceColors.loc[name],
                )
            )

        return create_2d_fig(traces, title="Trajectories in UTM coordinates")

    def create_2d_plot_slam(self, include_slam=True):
        traces = []

        if include_slam:
            traces.append(
                create_slam_2d_scatter(
                    self.orbslam.mapping.slam.slam,
                    "SLAM (mapping)",
                    color=TraceColors.slam,
                    mode="lines",
                ),
            )

        for name, localization in self.orbslam.localization.items():
            traces.append(
                create_slam_2d_scatter(
                    localization.slam.slam,
                    f"loc_{name}",
                    color=TraceColors.loc[name],
                    mode="markers",
                )
            )

        return create_2d_fig(
            traces, title="Localization trajectories in SLAM coordinates"
        )

    def create_2d_plot_slam_with_gt(self):
        gt_trace = create_2d_scatter(
            self.orbslam.mapping.gt.utm,
            "GPS (ground truth)",
            color=TraceColors.gt,
            mode="markers",
        )
        loc_traces = [
            create_slam_2d_scatter(
                self.orbslam.mapping.slam.slam,
                "SLAM (mapping)",
                color=TraceColors.slam,
                mode="lines",
            ),
        ]

        for name, localization in self.orbslam.localization.items():
            loc_traces.append(
                create_slam_2d_scatter(
                    localization.slam.slam,
                    f"loc_{name}",
                    color=TraceColors.loc[name],
                    mode="markers",
                )
            )

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Ground truth", "SLAM"),
            horizontal_spacing=0.05,
        )

        fig.add_trace(gt_trace, row=1, col=1)
        for trace in loc_traces:
            fig.add_trace(trace, row=1, col=2)

        fig.update_layout(
            yaxis1=dict(scaleanchor="x1", scaleratio=1),
            yaxis2=dict(scaleanchor="x2", scaleratio=1),
            legend=dict(itemsizing="constant"),
            margin={"t": 50, "b": 10, "l": 10, "r": 10},
            title_text=f"{self.trajectory_name}",
        )

        return fig

    def create_3d_plot_slam(self):
        traces = [
            create_3d_scatter(
                self.orbslam.mapping.slam.slam,
                "SLAM (mapping)",
                color=TraceColors.slam,
                mode="lines",
                bold=True,
            ),
            create_3d_scatter(
                self.orbslam.map_points,
                "Map points",
                color="black",
                mode="markers",
                size=1,
            ),
        ]

        # for name, localization in self.orbslam.localization.items():
        #     traces.append(
        #         create_3d_scatter(
        #             localization.slam.slam,
        #             f"SLAM loc_{name}",
        #             color=TraceColors.loc[name],
        #             mode="markers",
        #         )
        #     )

        return create_3d_fig(traces, title="Trajectories in SLAM coordinates")

    def create_ape_plot_all(self, include_slam=True):
        traces = []
        subplot_titles = []
        if include_slam:
            traces.append(
                create_ape_trace(
                    self.orbslam.mapping.slam.utm, self.orbslam.mapping.gt.utm
                ),
            )
            subplot_titles.append("SLAM (mapping)")

        for name, localization in self.orbslam.localization.items():
            traces.append(create_ape_trace(localization.slam.utm, localization.gt.utm))

        for name, localization in self.orbslam.localization.items():
            subplot_titles.append(f"loc_{name}")

        fig = create_ape_fig_batch(
            traces,
            title=f"{self.trajectory_name} - {self.model_type.name} - localization APE (m)",
            subplot_titles=subplot_titles,
        )

        return fig

    def create_ape_plot(self, loc: int = None):
        if loc is None:
            return create_ape_fig(
                self.orbslam.mapping.slam.utm,
                self.orbslam.mapping.gt.utm,
                "SLAM (fitted)",
            )
        else:
            if (loc > len(self.orbslam.localization)) or (loc < 0):
                raise ArgumentError(f"Invalid localization number: {loc}")
            return create_ape_fig(
                self.orbslam.localization[loc].slam.utm,
                self.orbslam.localization[loc].gt.utm,
                f"loc_{loc}",
            )
