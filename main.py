
import streamlit as st
import pandas as pd
import numpy as np
import hiplot as hip
import os
from functools import partial

from machine_learning  import rf_regression, kn_regression, theta_forecaster
from lstm import lstm_model, multivariate_lstm_model
from nbeats_model import nbeats_model
from temporal_convolutional_network  import temporal_convolutional_network

results = {
    'best_r2': 0
}


dataset_location = ''
predictor = ''

chart_data = pd.DataFrame(columns=['r2'])
chart = st.line_chart()

my_slot1 = st.empty()


my_slot2 = st.empty()




# import graphviz as graphviz
# graph = graphviz.Digraph()
# graph.edge('run', 'intr')
# graph.edge('intr', 'runbl')
# graph.edge('run', 'kernel')
#
# st.graphviz_chart(graph)


# data = [{'dropout':0.1, 'lr': 0.001, 'loss': 10.0, 'optimizer': 'SGD'},
#         {'dropout':0.15, 'lr': 0.01, 'loss': 3.5, 'optimizer': 'Adam'},
#         {'dropout':0.3, 'lr': 0.1, 'loss': 4.5, 'optimizer': 'Adam'}]
#
# hip.Experiment.from_iterable(data).display_st()


def initiate_run(chart_data, chart, results, my_slot1, dataset_location, predictor, forecasting_horizon, mode, selected_features, selected_algos):

    import math
    from hyperopt import fmin, tpe, hp, Trials
    import pickle

    # global selected_algos, mode

    algo_space_map = {
        'TCN': {'algo': 'tcn',
             'lag_days': hp.uniformint('lag_days', 2, 50)
             },
        'LSTM': {'algo': 'lstm',
             'n_steps_in': hp.uniformint('n_steps_in', 2, 50),
             'hidden_layer_1_neurons': hp.uniformint('hidden_layer_1_neurons', 3, 100),
             'hidden_layer_2_neurons': hp.uniformint('hidden_layer_2_neurons', 3, 100)
             },
        'Sktime-RandomForest': {'algo': 'sk_random_forest',
             'window_length': hp.uniformint('sk_random_forest_window_length', 3, 100)
             },
        'Sktime-KNN': {'algo': 'sk_knn',
             'neighbours': hp.uniformint('sk_knn_neighbours', 1, 50),
             'window_length': hp.uniformint('sk_knn_window_length', 3, 100)
             },
        'Sktime-ThetaForecaster': {'algo': 'theta_forecaster',
             'sp': hp.uniformint('th_sp', 1, 50)
             },
        'NBeats': {'algo': 'nbeats'
             }
    }

    algos = [algo_space_map[selected_algo] for selected_algo in selected_algos]

    space = {
        'algo': hp.choice('algo', algos)
    }


    trials = None
    try:
        print('loading from local..')
        trials = pickle.load(open("trials.txt", "rb"))
    except:
        try:
            print('loading from mlflow..')
            from mlflow.tracking import MlflowClient

            # Download artifacts
            client = MlflowClient()
            # print(run.info.run_id)
            from mlflow.tracking.artifact_utils import _download_artifact_from_uri

            # artifact_path = mlflow.get_artifact_uri() + '/trials.txt'
            print('the artifact path:', artifact_path)
            _download_artifact_from_uri(artifact_path, '.')
            print('success')
        except:
            print('starting a new trial..')
            trials = Trials()
    Trials = None

    best = 0

    data =  {
        'chart_data': chart_data,
        'chart': chart,
        'results': results,
        'my_slot1': my_slot1,
        'dataset_location': dataset_location,
        'predictor': predictor,
        'forecasting_horizon': forecasting_horizon,
        'mode': mode,
        'selected_features': selected_features
    }

    # fmin_objective = partial(run_pipeline, chart_data, chart, results, my_slot1, dataset_location, predictor, forecasting_horizon, mode, selected_features)

    fmin_objective = partial(run_pipeline, data=data)

    if Trials:
        print('here')
        best = fmin(fmin_objective, space, trials=trials, algo=tpe.suggest, max_evals=10000, trials_save_file='trials.txt')
    else:
        best = fmin(fmin_objective, space, trials=trials, algo=tpe.suggest, max_evals=10000)

    # mlflow.get_artifact_uri()
    # mlflow.end_run()


    return 0



best_r2 = -9999

def run_pipeline(cfg, data):
    print('cfg:', cfg)
    print('data:', data)


    chart_data = data['chart_data']
    chart = data['chart']
    results = data['results']
    my_slot1 = data['my_slot1']
    dataset_location = data['dataset_location']
    predictor = data['predictor']
    forecasting_horizon = data['forecasting_horizon']
    mode = data['mode']
    selected_features = data['selected_features']



    x = input('press to continue')
    global best_r2
    # global chart_data, chart, results, my_slot1, dataset_location, predictor, forecasting_horizon
    # global mode, selected_features
    print('#####', cfg)



    cfg = cfg['algo']

    r2 = 0
    rmse = 0
    y_test = []
    y_pred = []


    if mode == 'univariate':
        if cfg['algo'] == 'sk_random_forest':
            r2, rmse, y_test, y_pred = rf_regression(dataset_location, predictor, forecasting_horizon, cfg)
        elif cfg['algo'] == 'sk_knn':
            r2, rmse, y_test, y_pred = kn_regression(dataset_location, predictor, forecasting_horizon, cfg)
        elif cfg['algo'] == 'theta_forecaster':
            r2, rmse, y_test, y_pred = theta_forecaster(dataset_location, predictor, forecasting_horizon, cfg)
        elif cfg['algo'] == 'lstm':
            r2, rmse, y_test, y_pred = lstm_model(dataset_location, predictor, forecasting_horizon, cfg)
        elif cfg['algo'] == 'tcn':
            r2, rmse, y_test, y_pred = temporal_convolutional_network(dataset_location, predictor, forecasting_horizon, cfg)
        elif cfg['algo'] == 'nbeats':
            r2, rmse, y_test, y_pred = nbeats_model(dataset_location, predictor, forecasting_horizon, cfg)

    else:
        if cfg['algo'] == 'lstm':
            r2, rmse, y_test, y_pred = multivariate_lstm_model(dataset_location, predictor, selected_features, forecasting_horizon, cfg)


    print('here:', r2)

    if r2 > best_r2:
        best_r2 = r2
        results['best_r2'] = r2
        results['best_params'] = cfg
        my_slot1.text(results)

        print(np.vstack((y_test,y_pred)).T.shape)




        chart_data = pd.DataFrame(
        np.vstack((y_test,y_pred)).T,
        columns = ['y_test', 'y_pred'])
        my_slot2.line_chart(chart_data)


    if r2 > 0:
        chart.add_rows(np.array([r2]))
    # mlflow.log_metric("r2", r2)
    # mlflow.log_artifact('trials.txt')


    return rmse

# uploaded_file = st.file_uploader("Upload the dataset")



def run_ui():
    datasets = os.listdir('datasets')
    dataset = st.selectbox(
        'Pick the dataset',
        datasets)
    dataframe = pd.read_csv('datasets/' + dataset)
    dataset_location = 'datasets/' +dataset
    st.write(dataframe)
    st.text('dataset has {} rows'.format(len(dataframe)))

    dataframe = dataframe.apply(lambda col: pd.to_datetime(col, errors='ignore')
                  if col.dtypes == object
                  else col,
                  axis=0)

    date_cols = [col for col in list(dataframe.columns) if (dataframe[col].dtype == np.dtype('datetime64[ns]'))]
    st.text('Date columns found: {}'.format(date_cols))

    date_index = st.selectbox(
        'Pick the date index',
        date_cols)


    forecasting_horizon = int(st.text_input("input the forecasting horizon", 49))


    options = list(dataframe.columns)
    predictor = st.selectbox(
        'Pick the predictor column',
        options)

    st.write('You selected:', predictor)

    options = ['univariate', 'multivariate']
    mode = st.selectbox(
        'Select the mode',
        options)

    st.write('You selected:', mode)



    if mode == 'multivariate':
        available_algos = ['LSTM']
    else:
        available_algos = ['TCN', 'LSTM', 'Sktime-RandomForest', 'Sktime-KNN', 'Sktime-ThetaForecaster', 'NBeats']

    selected_features = None

    if mode == 'multivariate':
        available_features = list(dataframe.columns)
        available_features.remove(predictor)

        selected_features = st.multiselect(
            'Selected features',
             available_features,
             available_features)



    selected_algos = st.multiselect(
        'Selected algorithms',
         available_algos,
         ['LSTM'])

    if predictor:
        st.line_chart(dataframe[predictor])

    if st.button('initiate automl'):
        initiate_run(chart_data, chart, results, my_slot1, dataset_location, predictor, forecasting_horizon, mode, selected_features, selected_algos)


run_ui()





