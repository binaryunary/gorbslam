from dataclasses import dataclass
from typing import List, Tuple
from matplotlib.colors import cnames

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from evo.tools.plot import plot_mode_to_idx, PlotMode
from evo.core.metrics import StatisticsType
from plotly.subplots import make_subplots

from gorbslam.common.utils import calculate_ape, create_trajectory_from_array


FIG_HEIGHT = 800


@dataclass(frozen=True)
class TraceColors:
    gt = "#636EFA"
    slam = "#EF553B"
    slam_scaled = "#00CC96"
    loc = ["#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]


def create_scattermapbox(
    df: pd.DataFrame, name: str, color: str = None, bold=False, mode="markers"
):
    return go.Scattermapbox(
        lat=df.lat,
        lon=df.lon,
        mode=mode,
        line=dict(width=5) if bold else None,
        marker=dict(color=color) if color else None,
        name=name,
    )


def create_map_fig(
    traces: List[go.Scattermapbox], center: tuple[float, float], title=None
) -> go.Figure:
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)

    fig.update_geos(projection_type="transverse mercator")
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=center[0], lon=center[1]), zoom=16),
        margin={"t": 20, "b": 0, "l": 0, "r": 0},
        height=FIG_HEIGHT,
        title=title,
    )
    return fig


def create_scatter(df: pd.DataFrame, name, color=None, bold=False, mode="markers"):
    return go.Scatter(
        x=df.x,
        y=df.y,
        mode=mode,
        line=dict(width=5) if bold else None,
        marker=dict(color=color, size=3) if color else None,
        name=name,
    )


def create_slam_scatter(df: pd.DataFrame, name, color=None, bold=False, mode="markers"):
    return go.Scatter(
        x=df.x,
        y=df.z,  # y and z are swapped in case of monocular SLAM
        mode=mode,
        line=dict(width=5) if bold else None,
        marker=dict(color=color, size=3) if color else None,
        name=name,
    )


def create_2d_fig(traces: List[go.Scatter], title=None) -> go.Figure:
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)

    fig.update_layout(
        title=title, yaxis=dict(scaleanchor="x", scaleratio=1), height=FIG_HEIGHT
    )
    return fig


@dataclass(frozen=True, init=True)
class APETrace:
    predicted: go.Scatter
    reference_gt: go.Scatter
    ape_values: np.ndarray


def create_ape_trace(
    predicted: pd.DataFrame,
    reference_gt: pd.DataFrame,
):
    trajectory = create_trajectory_from_array(predicted.to_numpy())
    trajectory_gt = create_trajectory_from_array(reference_gt.to_numpy())
    ape_metric = calculate_ape(trajectory, trajectory_gt)

    x_idx, y_idx, z_idx = plot_mode_to_idx(PlotMode.xy)

    ground_truth_x = trajectory_gt.positions_xyz[:, x_idx]
    ground_truth_y = trajectory_gt.positions_xyz[:, y_idx]

    estimated_x = trajectory.positions_xyz[:, x_idx]
    estimated_y = trajectory.positions_xyz[:, y_idx]

    ape_values = ape_metric.error

    return APETrace(
        predicted=go.Scatter(
            x=estimated_x,
            y=estimated_y,
            mode="markers",
            name="Estimated",
            text=ape_values,
            marker=dict(
                size=3,
                color=ape_values,  # Use the normalized error values
                coloraxis="coloraxis",
            ),
            showlegend=False,
        ),
        reference_gt=go.Scatter(
            x=ground_truth_x,
            y=ground_truth_y,
            # mode='lines+markers',
            name="Ground Truth",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        ),
        ape_values=ape_values,
    )


def create_ape_fig_batch(
    traces: List[APETrace], title=None, subplot_titles=None
) -> go.Figure:
    n_cols = 2
    n_rows = len(traces) // n_cols + len(traces) % n_cols

    fig = make_subplots(
        n_rows,
        n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )
    for i, trace in enumerate(traces):
        col = i % n_cols + 1
        row = i // n_cols + 1
        fig.add_trace(trace.reference_gt, row, col)
        fig.add_trace(trace.predicted, row, col)

    # Lock the scale for all subplots
    x_c = 1
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_yaxes(
                scaleanchor=f"x{x_c if x_c > 1 else ''}", scaleratio=1, row=r, col=c
            )
            x_c += 1

    all_ape_values = np.concatenate([trace.ape_values for trace in traces])
    tick_min = all_ape_values.min()
    tick_mean = all_ape_values.mean()
    tick_median = np.median(all_ape_values)
    tick_p99 = np.percentile(all_ape_values, 99)

    fig.update_layout(
        title_text=title,
        height=1400,
        # width=1200,
        coloraxis=dict(
            colorscale="Turbo",
            cmin=0,
            # cmin=all_ape_values.min(),
            cmax=tick_p99,
            colorbar=dict(
                title="APE (m)",
                titleside="top",
                tickmode="array",
                tickvals=[tick_min, tick_median, tick_mean, tick_p99],
                ticktext=[
                    f"Min: {tick_min:.2f}",
                    f"Median: {tick_median:.2f}",
                    f"Mean: {tick_mean:.2f}",
                    f"p99: {tick_p99:.2f}",
                ],
                ticks="outside",
                orientation="h",
            ),
        ),
    )
    return fig


def create_ape_fig(predicted: pd.DataFrame, reference_gt: pd.DataFrame):
    trajectory = create_trajectory_from_array(predicted.to_numpy())
    trajectory_gt = create_trajectory_from_array(reference_gt.to_numpy())

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
            name="Ground Truth",
            line=dict(color="gray", dash="dash"),
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
            mode="markers",
            name="Estimated",
            marker=dict(
                # size=5,
                color=ape_values,
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(
                    title="APE (m)",
                    titleside="top",
                    tickmode="array",
                    tickvals=[tick_min, tick_mid, tick_max],
                    # labelalias={100: "Hot", 50: "Mild", 2: "Cold"},
                    ticks="outside",
                ),
            ),
            # line=dict(color='Viridis', width=6)
        )
    )

    stats_title = "<br>".join(
        [
            f"Mean APE: {ape_metric.get_statistic(StatisticsType.mean)}",
            f"Median APE: {ape_metric.get_statistic(StatisticsType.median)}",
            f"RMS APE: {ape_metric.get_statistic(StatisticsType.rmse)}",
        ]
    )

    fig.update_layout(
        # title = f'{ape_metric.get_statistic().}<br>b<br>c',
        title=stats_title,
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        legend_orientation="h",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=FIG_HEIGHT,
    )

    return fig
