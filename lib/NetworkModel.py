import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import MSE
import tensorflow as tf
import numpy as numpy


class NetworkModel:

    def __init__(self):
        None

    def model_dense(self, input_size, output_size, hidden_units=[64], hidden_activation='relu',
                    output_activation='linear'):
        # Define the structure of the model.
        model = Sequential()
        model.add(Dense(units=input_size,
                        activation='linear'))
        for size in hidden_units:
            model.add(Dense(units=size,
                            activation=hidden_activation))

        model.add(Dense(units=output_size, activation=output_activation))
        return model

    def model_cnn_lagged(self, grouped_cols, output_size, lag_size=1, hidden_units=64,
                         hidden_activation='relu', output_activation='linear', cnn_padding='valid', use_bias=1,
                         pool_size=1, dropout=0.0):
        # Generate a model with a parallel cnn layer for each of the groups in grouped cols.
        inputs = []
        cnn_layers = []
        pool_layers = []
        flatten_layers = []
        for group in grouped_cols:
            group_size = len(group)
            input = keras.layers.Input(shape=(group_size, 1))
            padding = keras.layers.ZeroPadding1D(padding=1)(input)
            # dense = keras.layers.Dense(group_size*group_size, input_shape=(group_size,), use_bias=use_bias)(input)
            # reshape = keras.layers.Reshape(target_shape=(group_size, group_size,))(dense)
            cnn = keras.layers.Conv1D(filters=lag_size,
                                      kernel_size=3,
                                      strides=1,
                                      padding=cnn_padding,
                                      activation=hidden_activation,
                                      use_bias=use_bias,
                                      data_format="channels_last",
                                      kernel_initializer=keras.initializers.he_normal(seed=None))(padding)
            last = cnn
            if dropout > 0:
                last = keras.layers.Dropout(dropout)(cnn)
            pool = keras.layers.MaxPooling1D(pool_size=pool_size,
                                             strides=1,
                                             padding=cnn_padding,
                                             data_format='channels_last')(last)
            last2 = pool
            if dropout > 0:
                last2 = keras.layers.Dropout(dropout)(pool)
            flatten = keras.layers.Flatten()(last2)


            inputs.append(input)
            cnn_layers.append(cnn)
            pool_layers.append(pool)
            flatten_layers.append(flatten)

        # Merge the three layers togethor.
        merged = keras.layers.merge.concatenate(flatten_layers)

        # Add the output layer

        output = keras.layers.Dense(units=output_size,
                                    activation=output_activation,
                                    use_bias=use_bias,
                                    kernel_initializer=keras.initializers.he_normal(seed=None))(merged)

        model = keras.models.Model(inputs=inputs, outputs=output)
        return model

    def compile_model(self, model, optimizer, loss, metrics):
        # Compile the model
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)
        return model

    def fit_model(self, model, train_x, train_y, valid_x, valid_y, num_epochs=100, batch_size=32, callback_list=[]):
        model.fit(train_x,
                  train_y,
                  validation_data=(valid_x, valid_y),
                  epochs=num_epochs,
                  batch_size=batch_size,
                  callbacks=callback_list)

    def partition_data(self, data, train_percent=0.6, valid_percent=0.2, seed=42):
        # Partition the data into the provided segments.
        # return a tuple containing [trainX, trainY, validX, validY, testX, testY]
        numpy.random.seed(seed)
        nrow = data.shape[0]
        ranges = range(0, nrow - 1)
        valid_percent = 1.0 - valid_percent
        train, validate, test = numpy.split(data, [int(train_percent * len(data)),
                                                   int(valid_percent * len(data))])
        return train, validate, test
