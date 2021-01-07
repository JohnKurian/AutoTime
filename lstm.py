# univariate multi-step vector-output stacked lstm example
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

import pandas as pd

import numpy as np

from metrics import calculate_metrics



# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def lstm_model(dataset_location, predictor, forecasting_horizon, cfg):
    df = pd.read_csv(dataset_location)
    raw_seq = list(df[predictor].values)
    raw_seq = (raw_seq - np.min(raw_seq)) / np.ptp(raw_seq)
    df[predictor] = raw_seq

    # choose a number of time steps
    n_steps_in, n_steps_out = cfg['n_steps_in'], forecasting_horizon
    # split into samples
    X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # define model
    model = Sequential()
    model.add(LSTM(cfg['hidden_layer_1_neurons'], activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(cfg['hidden_layer_2_neurons'], activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')

    # # define model
    # model = Sequential()
    # model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
    # model.add(RepeatVector(n_steps_out))
    # model.add(LSTM(100, activation='relu', return_sequences=True))
    # model.add(TimeDistributed(Dense(1)))
    # model.compile(optimizer='adam', loss='mse')

    # model = Sequential()
    # model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
    # model.add(Conv1D(filters=128, kernel_size=5, padding='causal', activation='relu', input_shape=[None, 1]))
    # model.add(LSTM(100, activation='relu', return_sequences=True))
    # model.add(TimeDistributed(Dense(1)))
    # model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(X, y, epochs=20, verbose=1)

    # demonstrate prediction
    x_input = X[-1]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)

    rmse, r2 = calculate_metrics(yhat[0], df[predictor].values[-forecasting_horizon:], forecasting_horizon)

    return r2, rmse, df[predictor].values[-forecasting_horizon:], yhat

