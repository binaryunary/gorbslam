import typing
from dataclasses import dataclass
from os import path
from typing import List

import numpy as np
import plotly.graph_objects as go
import plotly.graph_objs as go
from evo.core.metrics import APE
from evo.core.trajectory import PoseTrajectory3D
from evo.tools.plot import plot_mode_to_idx, PlotMode
from keras.layers import Dense
from keras.models import Sequential
from plotly.subplots import make_subplots
from pykalman import KalmanFilter
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from gorbslam.common.utils import calculate_ape, create_trajectory_from_array, to_tum


FIG_HEIGHT = 800

@dataclass(frozen=True)
class TraceColors:
    gt = '#636EFA'
    slam = '#EF553B'
    slam_scaled = '#00CC96'
    loc = ['#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']


def create_scattermapbox(arr, name, color=None, bold=False, mode='markers'):
    return go.Scattermapbox(
        lat=arr[:,0],
        lon=arr[:,1],
        mode=mode,
        line=dict(width=5) if bold else None,
        marker=dict(color=color) if color else None,
        name=name
    )


def create_map_fig(traces: List[go.Scattermapbox], center, title=None) -> go.Figure:
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)

    fig.update_geos(projection_type="transverse mercator")
    fig.update_layout(mapbox_style="open-street-map",
                      mapbox=dict(center=dict(lat=center[0], lon=center[1]), zoom=16),
                      margin={"t": 20, "b": 0, "l": 0, "r": 0},
                      height=FIG_HEIGHT,
                      title=title)
    return fig


def create_scatter(arr, name, color=None, bold=False, mode='markers'):
    return go.Scatter(
        x=arr[:,0],
        y=arr[:,1],
        mode=mode,
        line=dict(width=5) if bold else None,
        marker=dict(color=color, size=3) if color else None,
        name=name
    )

def create_slam_scatter(arr, name, color=None, bold=False, mode='markers'):
    return go.Scatter(
        x=arr[:,0],
        y=arr[:,2],
        mode=mode,
        line=dict(width=5) if bold else None,
        marker=dict(color=color, size=3) if color else None,
        name=name
    )


def create_2d_fig(traces: List[go.Scatter], title=None) -> go.Figure:
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=FIG_HEIGHT
    )
    return fig


def create_ape_fig(trajectory_tum_utm: np.ndarray, trajectory_gt_tum_utm):
    trajectory = create_trajectory_from_array(trajectory_tum_utm)
    trajectory_gt = create_trajectory_from_array(trajectory_gt_tum_utm)

    ape_metric = calculate_ape(trajectory, trajectory_gt)

    print(f"Min APE: {ape_metric.error.min()}")
    print(f"Max APE: {ape_metric.error.max()}")

    x_idx, y_idx, z_idx = plot_mode_to_idx(PlotMode.xy)

    # ground_truth_xy = ground_truth_traj.positions_xyz[:, :2]
    ground_truth_x = trajectory_gt.positions_xyz[:, x_idx]
    ground_truth_y = trajectory_gt.positions_xyz[:, y_idx]

    # estimated_xy = estimated_traj.positions_xyz[:, :2]
    estimated_x = trajectory.positions_xyz[:, x_idx]
    estimated_y = trajectory.positions_xyz[:, y_idx]


    # Get APE values for color grading
    ape_values = ape_metric.error
    max_ape = np.max(ape_values)
    min_ape = np.min(ape_values)

    # Normalize APE values to the range [0, 1]
    # norm_ape_values = (ape_values - min_ape) / (max_ape - min_ape)

    fig = go.Figure()

    # Ground truth trajectory
    fig.add_trace(
        go.Scatter(
            x=ground_truth_x,
            y=ground_truth_y,
            # mode='lines+markers',
            name='Ground Truth',
            line=dict(color='gray', dash='dash')
        )
    )

    tick_min = ape_metric.error.min()
    tick_max = ape_metric.error.max()
    tick_mid = (tick_min + tick_max) / 2

    # Estimated trajectory
    fig.add_trace(
        go.Scatter(
            x=estimated_x,
            y=estimated_y,
            mode='markers',
            name='Estimated',
            marker=dict(
                # size=5,
                color=ape_values,
                colorscale='Turbo',
                showscale=True,
                colorbar=dict(
                    title="APE (m)",
                    titleside="top",
                    tickmode="array",
                    tickvals=[tick_min, tick_mid, tick_max],
                    # labelalias={100: "Hot", 50: "Mild", 2: "Cold"},
                    ticks="outside"
                ),
            ),
            # line=dict(color='Viridis', width=6)
        )
    )

    fig.update_layout(
        xaxis_title='x (m)',
        yaxis_title='y (m)',
        legend_orientation="h",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=FIG_HEIGHT,
    )

    return fig


# Placeholder for plotting using matplotlib accoring to evo's examples at
# https://github.com/MichaelGrupp/evo/blob/master/notebooks/metrics.py_API_Documentation.ipynb
def evo_plot():
    # ape_metric = calculate_ape(trajectory, trajectory_gt)

    # ape_stats = ape_metric.get_all_statistics()
    # pprint.pprint(ape_stats)

    # pose_relation = metrics.PoseRelation.rotation_angle_deg
    # delta = 1
    # delta_unit = metrics.Unit.frames
    # all_pairs = False  # activate

    # data = (trajectory_gt, trajectory)

    # rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
    # rpe_metric.process_data(data)
    # rpe_stats = rpe_metric.get_all_statistics()

    # traj_ref_plot = copy.deepcopy(trajectory_gt)
    # traj_est_plot = copy.deepcopy(trajectory)
    # traj_ref_plot.reduce_to_ids(rpe_metric.delta_ids)
    # traj_est_plot.reduce_to_ids(rpe_metric.delta_ids)
    # seconds_from_start = [t - trajectory.timestamps[0] for t in trajectory.timestamps[1:]]

    # rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
    # rpe_metric.process_data(data)

    # traj_by_label = {
    #     'fitted SLAM (mapping)': trajectory,
    #     'GPS (ground truth)': trajectory_gt
    # }

    # plot_mode = plot.PlotMode.xy
    # fig = plt.figure()
    # ax = plot.prepare_axis(fig, plot_mode)
    # plot.traj(ax, plot_mode, trajectory_gt, '--', "gray", "reference")
    # plot.traj_colormap(ax, traj_est_plot, rpe_metric.error, plot_mode, min_map=rpe_stats["min"], max_map=rpe_stats["max"])
    # ax.legend()
    # plt.show()
    pass
