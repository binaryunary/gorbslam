from dataclasses import dataclass
import typing
from os import path
from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from keras.layers import Dense
from keras.models import Sequential
from pykalman import KalmanFilter
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from evo.core.trajectory import PoseTrajectory3D
from evo.core.metrics import APE

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
                      height=1200,
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
        height=1200
    )
    return fig


def create_ape_fig(ground_truth_traj: PoseTrajectory3D, estimated_traj: PoseTrajectory3D, ape_metric: APE):
    ground_truth_xyz = np.array([pose.t for pose in ground_truth_traj.poses_se3])
    estimated_xyz = np.array([pose.t for pose in estimated_traj.poses_se3])

    # Get ATE values for color grading
    ate_values = ape_metric.error
    max_ate = np.max(ate_values)
    min_ate = np.min(ate_values)

    # Generate color map based on ATE values
    colorscale = cl.scales['9']['seq']['Viridis']
    norm_ate_values = (ate_values - min_ate) / (max_ate - min_ate)
    color_indices = (norm_ate_values * (len(colorscale) - 1)).astype(int)
    colors = np.array([colorscale[i] for i in color_indices])

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Ground truth trajectory
    fig.add_trace(
        go.Scatter3d(
            x=ground_truth_xyz[:, 0],
            y=ground_truth_xyz[:, 1],
            z=ground_truth_xyz[:, 2],
            mode='lines+markers',
            name='Ground Truth',
            line=dict(color='blue')
        )
    )

    # Estimated trajectory
    for i in range(1, len(estimated_xyz)):
        fig.add_trace(
            go.Scatter3d(
                x=estimated_xyz[i-1:i+1, 0],
                y=estimated_xyz[i-1:i+1, 1],
                z=estimated_xyz[i-1:i+1, 2],
                mode='lines+markers',
                name='Estimated' if i == 1 else None,  # Show name only for the first trace
                showlegend=False if i > 1 else True,
                line=dict(color=colors[i], width=6)
            )
        )

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

    return fig
