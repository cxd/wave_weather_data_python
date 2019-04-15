import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import MSE
import numpy as numpy


class NetworkModel:

    def __init__(self):
        None


    def model_dense(self, input_size, output_size, hidden_units=[64], hidden_activation='relu', output_activation='linear'):
        # Define the structure of the model.
        model = Sequential()
        model.add(Dense(units=input_size,
                        activation='linear'))
        for size in hidden_units:
            model.add(Dense(units=size,
                            activation=hidden_activation))

        model.add(Dense(units=output_size, activation=output_activation))
        return model

    def compile_model(self, model, optimizer, loss, metrics):
        # Compile the model
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)
        return model

    def fit_model(self, model, train_x, train_y, valid_x, valid_y, num_epochs=100, batch_size=32):
        model.fit(train_x,
                  train_y,
                  validation_data=(valid_x, valid_y),
                  epochs=num_epochs,
                  batch_size=batch_size)


    def partition_data(self, data, train_percent=0.6, valid_percent=0.2, seed=42):
        # Partition the data into the provided segments.
        # return a tuple containing [trainX, trainY, validX, validY, testX, testY]
        numpy.random.seed(seed)
        nrow = data.shape[0]
        ranges = range(0,nrow-1)
        valid_percent = 1.0 - valid_percent
        train, validate, test = numpy.split(data, [int(train_percent*len(data)),
                                                   int(valid_percent*len(data))])
        return train, validate, test


