from typing import List


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyproj
import tensorflow as tf
from keras.layers import Dense, Normalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import L2
from matplotlib import pyplot as plt


from utils import ORBSLAMResults, umeyama_alignment, kabsch_umeyama, estimate_helmert_parameters, helmert_transform


n_inputs = 3 * 1


def reshape_data(points: np.ndarray, n_inputs: int) -> np.ndarray:
    mod = points.size % n_inputs

    if mod != 0:
        rng = np.random.default_rng()
        n_rows = points.shape[0]
        rows_to_remove = mod // 3
        return np.delete(points, rng.choice(n_rows, rows_to_remove, replace=False), axis=0).reshape(-1, n_inputs)

    return np.array(points).reshape(-1, n_inputs)


def fit_trajectory(source_points, target_points, epochs=400, batch_size=32):
    assert source_points.shape == target_points.shape

    print(f"Training model with {source_points.shape}")
    print(f"Target points: {target_points.shape}")

    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

    source = reshape_data(source_points, n_inputs)
    target = reshape_data(target_points, n_inputs)

    print("Resized source: ", source.shape)
    print("Resized target: ", target.shape)

    source_normalizer = Normalization(input_shape=(n_inputs,))
    source_normalizer.adapt(source)

    target_normalizer = Normalization(input_shape=(n_inputs,))
    target_normalizer.adapt(target)

    X = source_normalizer(source)
    y = target_normalizer(target)

    regularizer = L2(l2=0.001)


    n_nodes = 64

    # Define the neural network architecture
    model = Sequential()
    model.add(Dense(n_nodes, activation='tanh', input_shape=(n_inputs,)))
    # model.add(Dense(n_nodes, activation='tanh'))
    model.add(Dense(n_inputs))


    optimizer = Adam(learning_rate=0.001)
    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model with your data
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,  verbose=True)


    # plt.plot(history.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    return model, source_normalizer, target_normalizer


def denormalize(normalizer, data):
    mean = normalizer.mean.numpy()
    variance = normalizer.variance.numpy()
    std = np.sqrt(variance)
    return data * std + mean


def predict_trajectory(model, source_points, source_normalizer, target_normalizer):
    print(f"Predicting target trajectory with {source_points.shape}")

    source = reshape_data(source_points, n_inputs)
    print(f"Resized source: {source.shape}")

    normalized_source_points = source_normalizer(source)

    X = normalized_source_points

    # Predict the target trajectory
    normalized_predicted_target_points = model.predict(X)
    denormalized_predicted_target_points = denormalize(target_normalizer, normalized_predicted_target_points)

    return denormalized_predicted_target_points.reshape(-1, 3)


def create_scattermapbox(df, name, color=None, bold=False):
    return go.Scattermapbox(
        lat=df['lat'],
        lon=df['lon'],
        mode='lines',
        line=dict(width=5) if bold else None,
        marker=dict(color=color) if color else None,
        name=name
    )


def plot(traces: List[go.Scattermapbox], center):
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)

    fig.update_geos(projection_type="transverse mercator")
    fig.update_layout(mapbox_style="open-street-map",
                      mapbox=dict(center=dict(lat=center[0], lon=center[1]), zoom=16),
                      margin={"t": 20, "b": 0, "l": 0, "r": 0},
                      height=800)
    fig.show()


def main():
    results = ORBSLAMResults("data")

    keyframes = results.keyframes[1:] # Skip first keyframe, GPS at (0, 0, 0)

    gps_trajectory_wgs = pd.DataFrame([(kf.gps.lat, kf.gps.lon, kf.gps.alt)
                                      for kf in keyframes], columns=['lat', 'lon', 'alt'])
    slam_trajectory = np.array([(kf.x, kf.y, kf.z) for kf in keyframes])

    # Create transformers for WGS84 <-> UTM35N
    wgs2utm = pyproj.Transformer.from_crs(4326, 32635)
    utm2wgs = pyproj.Transformer.from_crs(32635, 4326)

    # Convert GPS trajectory (WGS84) to UTM35N
    gps_trajectory_utm = np.array([wgs2utm.transform(kf.gps.lat, kf.gps.lon, kf.gps.alt)
                                  for kf in keyframes])

    # Align SLAM trajectory to GPS trajectory
    R, t, c = umeyama_alignment(slam_trajectory.T, gps_trajectory_utm.T, True)
    # R, t, c = kabsch_umeyama(gps_trajectory_utm, slam_trajectory)
    aligned_slam_trajectory_utm = np.array([t + c * R @ p for p in slam_trajectory])

    # Find transformation between GPS and SLAM trajectories
    # helmert_params = estimate_helmert_parameters(slam_trajectory, gps_trajectory_utm)
    # aligned_slam_trajectory_utm, R, t, c = helmert_transform(helmert_params, slam_trajectory)

    # Convert SLAM trajectory (UTM35N) to WGS84
    aligned_slam_trajectory_wgs = pd.DataFrame([utm2wgs.transform(p[0], p[1], p[2])
                                                for p in aligned_slam_trajectory_utm], columns=['lat', 'lon', 'alt'])

    # model, source_normalizer, target_normalizer = fit_trajectory(
    #     aligned_slam_trajectory_utm, gps_trajectory_utm)

    model, source_normalizer, target_normalizer = fit_trajectory(
        slam_trajectory, gps_trajectory_utm)

    print("Fitting SLAM trajectory...")
    # fitted_slam_trajectory_utm = predict_trajectory(
    #     model, aligned_slam_trajectory_utm, source_normalizer, target_normalizer)
    fitted_slam_trajectory_utm = predict_trajectory(
        model, slam_trajectory, source_normalizer, target_normalizer)
    fitted_slam_trajectory_wgs = pd.DataFrame([utm2wgs.transform(p[0], p[1], p[2])
                                              for p in fitted_slam_trajectory_utm], columns=['lat', 'lon', 'alt'])

    slam_estimates = np.array([(e.lat, e.lon, e.alt) for e in results.slam_estimates])
    aligned_slam_estimate_utm = np.array([t + c * R @ p for p in slam_estimates])
    aligned_slam_estimate_wgs = pd.DataFrame([utm2wgs.transform(p[0], p[1], p[2])
                                              for p in aligned_slam_estimate_utm], columns=['lat', 'lon', 'alt'])
    print("Fitting SLAM estimate...")
    # fitted_slam_estimate_utm = predict_trajectory(
    #     model, np.delete(aligned_slam_estimate_utm, 0, 0), source_normalizer, target_normalizer)
    fitted_slam_estimate_utm = predict_trajectory(
        model, slam_estimates, source_normalizer, target_normalizer)
    fitted_slam_estimate_wgs = pd.DataFrame([utm2wgs.transform(p[0], p[1], p[2])
                                            for p in fitted_slam_estimate_utm], columns=['lat', 'lon', 'alt'])

    center_lat = np.mean(gps_trajectory_wgs['lat'])
    center_lon = np.mean(gps_trajectory_wgs['lon'])
    plot([
        create_scattermapbox(gps_trajectory_wgs, 'GPS', 'blue'),
        create_scattermapbox(aligned_slam_trajectory_wgs, 'SLAM (mapping)', 'red'),
        create_scattermapbox(fitted_slam_trajectory_wgs, 'fitted SLAM (mapping)'),
        create_scattermapbox(aligned_slam_estimate_wgs, 'SLAM (localization)', 'forestgreen'),
        create_scattermapbox(fitted_slam_estimate_wgs, 'fitted SLAM (localization)')
    ],
    (center_lat, center_lon))


if __name__ == '__main__':
    main()
