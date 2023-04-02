from keras_tuner import HyperModel, RandomSearch, HyperParameter
from keras.layers import Dense, Normalization, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np


class SLAMHyperModel(HyperModel):
    def __init__(self, project_name, log_root="keras_logs"):
        self.project_name = project_name
        self.log_root = log_root
        self.source_normalizer = None
        self.target_normalizer = None
        self.model = None

    def adapt_normalizers(self, slam_trajectory, gt_trajectory):
        self.__assert_input_shape(slam_trajectory)

        self.source_normalizer = Normalization(input_shape=(3, ))
        self.source_normalizer.adapt(slam_trajectory)

        self.target_normalizer = Normalization(input_shape=(3, ))
        self.target_normalizer.adapt(gt_trajectory)

    def search(self, slam_trajectory, gt_trajectory):
        tuner = RandomSearch(
            self,
            overwrite=True,
            objective='val_loss',
            max_trials=100,
            executions_per_trial=3,
            directory=self.log_root,
            project_name=self.project_name
        )

        callbacks = [
            EarlyStopping(monitor='loss', patience=2),
            TensorBoard(log_dir=self.log_root),
        ]

        X = self.source_normalizer(slam_trajectory)
        y = self.target_normalizer(gt_trajectory)

        tuner.search(X, y, validation_split=0.2, callbacks=callbacks)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.model = self.build(best_hps)

    def train(self, slam_trajectory, gt_trajectory, epochs=400, batch_size=32):
        callbacks = [
            EarlyStopping(monitor='loss', patience=2),
        ]

        X = self.source_normalizer(slam_trajectory)
        y = self.target_normalizer(gt_trajectory)

        self.model.fit(X, y, callbacks=callbacks, epochs=epochs, batch_size=batch_size, verbose=True)

    def build(self, hp: HyperParameter):
        model = Sequential()

        il_units = hp.Int('input_layer_units', min_value=8, max_value=256, step=8)
        il_activation = hp.Choice('input_layer_activation', values=['relu', 'tanh'])

        # Input layer
        model.add(Dense(units=il_units, activation=il_activation, input_shape=(3,)))

        num_hidden_layers = hp.Int('num_hidden_layers', 0, 3)
        # Choose the number of hidden layers
        for i in range(num_hidden_layers):
          hl_units = hp.Int(f'hidden_layer_{i}_units', min_value=8, max_value=256, step=8)
          hl_activation = hp.Choice(f'hidden_layer_{i}_activation', values=['relu', 'tanh'])
          hl_is_regularizer = hp.Boolean(f'is_hidden_layer_{i}_kernel_regularizer', default=False)
          hl_regularizer = None
          if hl_is_regularizer:
              hl_regularizer = hp.Choice(f'hidden_layer_{i}_kernel_regularizer', values=['l1', 'l2', 'l1_l2'])

          # Tune the number of nodes and activation function for each layer
          model.add(Dense(units=hl_units, activation=hl_activation, kernel_regularizer='l2' if hl_is_regularizer else None))
          model.add(Dense(units=hl_units, activation=hl_activation))

          # Add dropout to hidden layers
          hl_is_dropout = hp.Boolean(f'hidden_layer_{i}_dropout', default=False)
          if hl_is_dropout  and i < num_hidden_layers - 1: # don't add dropout to the last hidden layer before the output
              model.add(Dropout(rate=hp.Choice(f'hidden_layer_{i}_dropout_rate', values=[0.5, 0.6, 0.7, 0.8])))

        # Output layer
        model.add(Dense(3))

        # Tune the learning rate
        # learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
        optimizer_name = hp.Choice('optimizer', ['sgd', 'adam'])

        # if optimizer_name == 'sgd':
        #     optimizer = SGD(learning_rate=learning_rate,
        #                     momentum=hp.Float('momentum', min_value=0.0, max_value=0.9, step=0.1))
        # elif optimizer_name == 'rmsprop':
        #     optimizer = RMSprop(learning_rate=learning_rate)
        # elif optimizer_name == 'adam':
        #     optimizer = Adam(learning_rate=learning_rate)

        # Compile the model
        model.compile(optimizer=optimizer_name, loss='mean_squared_error')

        return model

    def predict(self, slam_trajectory):
        self.__assert_input_shape(slam_trajectory)

        X = self.source_normalizer(slam_trajectory)

        predicted_trajectory_norm = self.model.predict(X)

        # Denormalize the predictions
        predicted_trajectory_denorm = self._denormalize(predicted_trajectory_norm)

        return predicted_trajectory_denorm

    def __assert_input_shape(self, slam_trajectory):
        assert slam_trajectory.shape[1] == 3

    def _denormalize(self, data):
      variance = self.target_normalizer.variance.numpy()
      mean = self.target_normalizer.mean.numpy()
      std = np.sqrt(variance)

      return data * std + mean

