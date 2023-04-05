from dataclasses import dataclass
import typing
from os import path
from typing import List

import numpy as np
import plotly.graph_objects as go
from keras.layers import Dense
from keras.models import Sequential
from pykalman import KalmanFilter
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


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
