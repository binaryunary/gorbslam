from keras_tuner import HyperModel, HyperParameters
from keras.layers import Dense, Normalization
from keras.models import Sequential
from keras.losses import Huber
from keras import backend as K
import tensorflow as tf

from gorbslam.common.utils import assert_trajectory_shape


class NNHyperModel(HyperModel):
    def __init__(self):
        self.source_normalizer = None
        self.target_normalizer = None
        self.model = None

    def adapt_normalizers(self, slam_trajectory, gt_trajectory):
        assert_trajectory_shape(slam_trajectory)

        self.source_normalizer = Normalization(input_shape=(3, ))
        self.source_normalizer.adapt(slam_trajectory)

        self.target_normalizer = Normalization(input_shape=(3, ))
        self.target_normalizer.adapt(gt_trajectory)

    def build(self, hp: HyperParameters):
        model = Sequential()

        # Input layer
        il_units = hp.Int('il_units', min_value=16, max_value=1024, sampling='log')
        il_activation = hp.Choice('il_activation', values=['relu', 'tanh'])
        model.add(Dense(units=il_units, activation=il_activation, input_shape=(3,)))

        # Choose the number of hidden layers
        hl_num = hp.Int('num_hidden_layers', 0, 5)
        for i in range(hl_num):
            # ith hidden layer
            hl_units = hp.Int(f'hl_{i}_units', min_value=16, max_value=1024, sampling='log')
            hl_activation = hp.Choice(f'hl_{i}_activation', values=['relu', 'tanh'])
            # Tune the number of nodes and activation function for each layer
            model.add(Dense(units=hl_units, activation=hl_activation))

        # Output layer
        model.add(Dense(3))

        huber_delta = hp.Float('huber_delta', min_value=0.1, max_value=1.0, step=0.1)
        loss = Huber(delta=huber_delta)

        # Compile the model
        model.compile(optimizer='adam', loss=loss)

        return model

    def fit(self, hp: HyperParameters, model, X, y, **kwargs):
        # epochs = hp.Choice('epochs', values=[50, 100, 200, 300])
        batch_size = hp.Choice('batch_size', values=[32, 64, 128, 256, 512])
        return model.fit(X, y, batch_size=batch_size, **kwargs)


# Custom loss function to measure the euclidean distance between the predicted and ground truth trajectory points.
def euclidean_distance(y_true, y_pred):
    return tf.reduce_mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))
