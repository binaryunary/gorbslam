from keras_tuner import HyperModel, HyperParameters
from keras.layers import Dense, Normalization, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam, Nadam, Adamax, Adagrad, Adadelta
from keras.losses import Huber
from keras import backend as K
import tensorflow as tf


from utils import assert_trajectory_shape


class SLAMHyperModel(HyperModel):
    def __init__(self):
        # self.project_name = project_name
        # self.log_root = log_root
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

        # il_units = hp.Choice('il_units', values=[16, 32, 64, 128, 256])
        il_units = hp.Int('il_units', min_value=16, max_value=256, sampling='log')
        il_activation = hp.Choice('il_activation', values=['relu', 'tanh'])

        # Input layer
        model.add(Dense(units=il_units, activation=il_activation, input_shape=(3,)))

        hl_num = hp.Int('num_hidden_layers', 0, 5)
        # Choose the number of hidden layers
        for i in range(hl_num):
            # hl_units = hp.Choice(f'hl_{i}_units', values=[16, 32, 64, 128, 256])
            hl_units = hp.Int(f'hl_{i}_units', min_value=16, max_value=256, sampling='log')
            hl_activation = hp.Choice(f'hl_{i}_activation', values=['relu', 'tanh'])
            hl_is_regularizer = hp.Boolean(f'hl_{i}_is_regularizer', default=False)
            hl_regularizer = None
            if hl_is_regularizer:
                hl_regularizer = hp.Choice(f'hl_{i}_regularizer', values=['l1', 'l2'])

            # Tune the number of nodes and activation function for each layer
            model.add(Dense(units=hl_units, activation=hl_activation))

            # Add dropout to hidden layers
            # hl_is_dropout = hp.Boolean(f'hl_{i}_is_dropout', default=False)
            # if hl_is_dropout and i < hl_num - 1:  # don't add dropout to the last hidden layer before the output
            #     model.add(Dropout(rate=hp.Choice(f'hl_{i}_dropout_rate', values=[0.5, 0.6, 0.7, 0.8])))

        # Output layer
        model.add(Dense(3))

        # Choose the optimizer
        # optimizer_name = hp.Choice('optimizer', ['sgd', 'rmsprop', 'adam', 'adamax', 'nadam'])

        # if optimizer_name == 'sgd':
        #     optimizer = SGD(
        #         learning_rate=hp.Float('sgd_learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG'),
        #         momentum=hp.Float('sgd_momentum', min_value=0.0, max_value=0.9, step=0.1),
        #         nesterov=hp.Boolean('sgd_nesterov', default=False)
        #     )
        # elif optimizer_name == 'rmsprop':
        #     optimizer = RMSprop(
        #         learning_rate=hp.Float('rmsprop_learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG'),
        #         rho=hp.Float('rmsprop_rho', min_value=0.8, max_value=0.99, step=0.01)
        #     )
        # elif optimizer_name == 'adam':
        #     optimizer = Adam(
        #         learning_rate=hp.Float('adam_learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG'),
        #         beta_1=hp.Float('adam_beta_1', min_value=0.8, max_value=0.99, step=0.01),
        #         beta_2=hp.Float('adam_beta_2', min_value=0.9, max_value=0.999, step=0.001)
        #     )
        # elif optimizer_name == 'adamax':
        #     optimizer = Adamax(
        #         learning_rate=hp.Float('adamax_learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG'),
        #         beta_1=hp.Float('adamax_beta_1', min_value=0.8, max_value=0.99, step=0.01),
        #         beta_2=hp.Float('adamax_beta_2', min_value=0.9, max_value=0.999, step=0.001)
        #     )
        # elif optimizer_name == 'nadam':
        #     optimizer = Nadam(
        #         learning_rate=hp.Float('nadam_learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG'),
        #         beta_1=hp.Float('nadam_beta_1', min_value=0.8, max_value=0.99, step=0.01),
        #         beta_2=hp.Float('nadam_beta_2', min_value=0.9, max_value=0.999, step=0.001)
        #     )

         # Choose the loss function
        # loss_function = hp.Choice('loss_function', ['log_cosh', 'huber'])

        # Huber loss requires a delta value, so we need to include it when using Huber loss
        # if loss_function == 'huber':
        #     huber_delta = hp.Float('huber_delta', min_value=0.1, max_value=1.0, step=0.1)
        #     loss = Huber(delta=huber_delta)
        # else:
        #     loss = loss_function

        huber_delta = hp.Float('huber_delta', min_value=0.1, max_value=1.0, step=0.1)
        loss = Huber(delta=huber_delta)

        # Compile the model
        # model.compile(optimizer=optimizer, loss=loss, metrics=[euclidean_distance])
        model.compile(optimizer='adam', loss=loss, metrics=[euclidean_distance])

        return model

    def fit(self, hp: HyperParameters, model, X, y, **kwargs):
        # epochs = hp.Choice('epochs', values=[50, 100, 200, 300])
        batch_size = hp.Choice('batch_size', values=[32, 64, 128, 256, 512])
        return model.fit(X, y, batch_size=batch_size, **kwargs)


def euclidean_distance(y_true, y_pred):
    # using tf.reduce_mean(euclidean_distance) instead of euclidean_distance directly
    # would give you the average distance between the true and predicted values across
    # all points in the batch, rather than the average distance for each point
    # in the batch separately.
    return tf.reduce_mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))



