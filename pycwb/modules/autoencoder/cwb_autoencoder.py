# -*- coding: utf-8 -*-
#
# Copyright (C) 2021  Sophie Bini, Gabriele Vedovato
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# author: Sophie Bini


import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # INFO and WARNING and ERROR tensorflow messages are not printed
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from tensorflow.python.keras.layers import Flatten, Reshape, AveragePooling1D, Input, Dense, Conv1D, MaxPooling1D, \
    UpSampling1D
from tensorflow.python.keras.models import Model


# functions:
# 1. timeseries windowing and normalization
# 2. autoencoder neural network
# 3. Mean square error between autoencoder input and output

# 1. timeseries windowing and normalization ---------------------------------------------------------------- #

def timeseries_processing(data, window):
    """
    timeseries windowing and normalization

    Parameters
    ----------
    data: pycbc.timeseries.TimeSeries
        timeseries
    window: int
        window length

    Returns
    -------
    data_mod: numpy.ndarray
        windowed timeseries centered around the absolute maximum, normalized between [0,1], offset = 0.5
    """

    data = np.array(data)
    maxim_len = len(data)
    data_min = data.min()
    data_min = np.abs(data_min)
    data_max = data.max()

    if data_max > data_min:

        half = data.argmax()
        if ((half < window + 1) or (
                half > maxim_len - window)):  # if the maximum is close to the timeseries edge, padding with the overall mean value
            data = np.pad(data, (window + 2,), 'mean')
            half = half + window + 2

        data_c = data[int(half - window): (int(half + window))]  # cropping the timeseries
        norm = data_c.max()  # normalization
        data_mod = data_c / (2 * norm)

    else:  # same as before
        half = data.argmin()

        if ((half < window + 1) or (half > maxim_len - window)):
            data = np.pad(data, (window + 2,), 'mean')
            half = half + window + 2

        data_c = data[int(half - window): (int(half + window))]
        data_min = data_c.min()
        data_min = np.abs(data_min)
        norm = data_min
        data_mod = data_c / (2 * norm)

    data_mod = data_mod + 0.5  # offset
    data_mod = np.reshape(data_mod, (window * 2, 1))

    return data_mod


# autoencoder neural network ---------------------------------------------------------------------------------------------  #

def autoencoder(dim1):
    """
    autoencoder neural network

    Parameters
    ----------
    dim1: int
        timeseries length after preprocessing ( = 2*window )

    Returns
    -------
    autoencoder: tensorflow.python.keras.engine.training.Model
        autoencoder model
    """
    input_sig = Input(shape=(dim1, 1))

    x = Conv1D(64 * 2, 3, activation='relu', padding='same')(input_sig)
    x1 = MaxPooling1D(2)(x)
    x2 = Conv1D(16, 3, activation='relu', padding='same')(x1)
    x3 = MaxPooling1D(2)(x2)
    x4 = Conv1D(16, 3, activation='relu', padding='same')(x3)
    x5 = AveragePooling1D()(x4)
    flat = Flatten()(x5)
    encoded = Dense(200)(flat)

    d1 = Dense(x5.shape[1] * x5.shape[2])(encoded)
    d2 = Reshape((x5.shape[1], x5.shape[2]))(d1)
    d3 = Conv1D(16, 3, strides=1, activation='relu', padding='same')(d2)
    d4 = UpSampling1D(2)(d3)
    d5 = Conv1D(16, 3, strides=1, activation='relu', padding='same')(d4)
    d6 = UpSampling1D(2)(d5)
    d7 = Conv1D(64 * 2, 3, strides=1, activation='relu', padding='same')(d6)
    d8 = UpSampling1D(2)(d7)
    decoded = Conv1D(1, 3, strides=1, activation='relu', padding='same')(d8)

    return Model(inputs=input_sig, outputs=decoded, name='AE')


# mean square error (MSE)  --------------------------------- #

def MSE(original, reconstructed):
    """
    Mean square error between autoencoder input and output

    Parameters
    ----------
    original: numpy.ndarray
        original timeseries
    reconstructed: numpy.ndarray
        reconstructed timeseries

    Returns
    -------
    mse: numpy.ndarray
        mean square error between original and reconstructed timeseries
    """
    mse_1 = (original - reconstructed) ** (2)  # mean square error
    mse = mse_1.mean(axis=1)
    return mse


# --------------------------------------------------------- #
class AutoEncoder:

    def __init__(self):
        self.rate = 0
        self.glness = 0
        self.weights = ''  # r''
        self.window = 0
        self.ae_net = None
        print('init autoencoder')

    def set_weights(self, weights):
        if not os.path.exists(weights): print("\nerror, file ", weights, " not exist\n"); exit(1)
        print('load autoencoder weights = ', weights)
        self.window = 208
        self.ae_net = autoencoder(self.window * 2)  # load the autoencoder model
        # self.ae_net.summary()                                       #print the autoencoder structure
        self.ae_net.load_weights(weights)  # load the autoencoder weights for BLIP glitches
        # print(self.ae_net.get_weights())
        return 0

    def get_glness(self, data, rate):
        # TODO: rate is not used?
        if (self.ae_net == None):
            print("\nerror, autoencoder weights not loaded\n");
            exit(1)

        ##  glitchiness
        data_norm = timeseries_processing(data, self.window)  # timeseries windowing and normalization
        data_norm = np.reshape(data_norm, (1, data_norm.shape[0], 1))  # reshape according to autoencoder needs
        reconstruction = self.ae_net.predict(data_norm)  # the autoencoder recosntructed the timeseries
        glness = MSE(reconstruction, data_norm)
        return glness
