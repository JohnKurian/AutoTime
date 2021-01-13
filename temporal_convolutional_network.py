from tensorflow.keras import Input, Model

from tcn import TCN, tcn_full_summary
import pandas as pd

from numpy import array
from tensorflow.keras.layers import Dense
import numpy as np

from metrics import calculate_metrics



def lag_days(col, x):
    df_new = pd.concat([col.shift(i) for i in range(x + 1)], axis=1)
    col_names = ['t-' + str(i) for i in range(x + 1)]
    df_new.columns = col_names
    df_new.columns = list(df_new.columns)[::-1]
    return df_new.dropna()



# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def temporal_convolutional_network(dataset_location, predictor, forecasting_horizon, cfg):
    df = pd.read_csv(dataset_location)
    raw_seq = list(df[predictor].values)
    raw_seq = (raw_seq - np.min(raw_seq)) / np.ptp(raw_seq)
    df[predictor] = raw_seq
    df = lag_days(df[predictor], cfg['lag_days'])
    cols = list(df.columns)
    cols.remove('t-0')
    col_arr = [df[col].values for col in cols] + [df['t-0'].values]
    dataset = np.stack(col_arr, axis=1)

    # choose a number of time steps
    n_steps_in, n_steps_out = forecasting_horizon, forecasting_horizon
    # covert into input/output
    x, y = split_sequences(dataset, n_steps_in, n_steps_out)
    # the dataset knows the number of features, e.g. 2
    n_features = x.shape[2]

    x = np.asarray(x).astype('float32')
    y = np.asarray(y).astype('float32')

    timesteps = cfg['lag_days']
    batch_size = None
    i = Input(batch_shape=(batch_size, n_steps_in, n_features))

    o = TCN(return_sequences=False)(i)  # The TCN layers are here.
    o = Dense(n_steps_out)(o)

    m = Model(inputs=[i], outputs=[o])
    m.compile(optimizer='adam', loss='mse')

    tcn_full_summary(m, expand_residual_blocks=False)

    m.fit(x, y, epochs=100, validation_split=0.2, shuffle=False, verbose=1)


    # demonstrate prediction
    x_input = x[-1]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = m.predict(x_input, verbose=0)

    rmse, r2 = calculate_metrics(yhat[0], df['t-0'].values[-forecasting_horizon:], forecasting_horizon)
    return r2, rmse, df['t-0'].values[-forecasting_horizon:], yhat