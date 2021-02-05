from flask import Flask
from flask_cors import CORS
import mlflow

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from flask import request



import streamlit as st
import pandas as pd
import numpy as np
import hiplot as hip
import os
from functools import partial

from machine_learning  import rf_regression, kn_regression, theta_forecaster
from lstm import lstm_model, multivariate_lstm_model
# from nbeats_model import nbeats_model
from temporal_convolutional_network  import temporal_convolutional_network

import SessionState

from datetime import datetime



from automl import initiate_run

mlflow.set_tracking_uri("http://localhost:5000")


client = MlflowClient()





app = Flask(__name__)
CORS(app)

@app.route('/hello')
def say_hello_world():
    return {'result': "Hello World"}



@app.route('/create_experiment', methods=['POST'])
def create_experiment():
    print('here')
    req_data = request.get_json(force=True)
    print(req_data)
    experiment = mlflow.create_experiment(req_data['exp_name'])
    client.set_experiment_tag(experiment,"dataset_location", req_data['dataset_location'])
    client.set_experiment_tag(experiment,"notes", req_data['notes'])
    client.set_experiment_tag(experiment, "forecasting_horizon", req_data['forecasting_horizon'])
    client.set_experiment_tag(experiment, "mode", req_data['mode'])
    client.set_experiment_tag(experiment, "predictor_column", req_data['predictor_column'])
    client.set_experiment_tag(experiment, "selected_algos", req_data['selected_algos'])
    return {'experiment_id': experiment}


@app.route('/start_run')
def start_run():
    client.get_run('c526b082f55e488fab7234b55ab561d0')
    return {'result': "Hello experiment"}


from threading import Thread
from time import sleep

def threaded_function(arg):
    for i in range(arg):
        print("running")
        sleep(1)

def run_mlflow_parallel(exp_name, experiment_id):
    # do something
    mlflow.set_experiment(exp_name)
    exp = client.get_experiment(experiment_id)
    dataset_location = 'datasets/' + exp.tags['dataset_location']

    forecasting_horizon = int(exp.tags['forecasting_horizon'])
    mode = exp.tags['mode']
    predictor_column = exp.tags['predictor_column']
    selected_algos = exp.tags['selected_algos']

    import ast
    selected_algos = ast.literal_eval(selected_algos)

    active_run = mlflow.active_run()
    if active_run is None:
        with mlflow.start_run() as run:
            initiate_run(exp_name,
                         dataset_location,
                         predictor_column,
                        forecasting_horizon,
                         mode,
                         [predictor_column],
                         selected_algos)
    else:
        print('ending previous run')
        mlflow.end_run()
        print('starting new run')
        with mlflow.start_run() as run:
            initiate_run(exp_name,
                         dataset_location,
                         predictor_column,
                        forecasting_horizon,
                         mode,
                         [predictor_column],
                         selected_algos)




from multiprocessing.pool import ThreadPool

@app.route('/create_run', methods=['POST'])
def create_run():
    experiment_id = request.args.get('experiment_id')

    exp = client.get_experiment(experiment_id)
    exp_name = exp.name

    run_mlflow_parallel(exp_name, experiment_id)
    # thread = Thread(target = run_mlflow_parallel, args = (exp_name, experiment_id ))
    # thread.start()

    # pool = ThreadPool(processes=100)
    # pool.map(run_mlflow_parallel, [exp_name])


    return {'result': "Hello experiment"}


@app.route('/get_experiment')
def get_experiment():
    return {'result': "Hello experiment"}



@app.route('/get_runs')
def get_runs():
    experiment_id = request.args.get('experiment_id')

    runs = [{'run_id': run.run_id} for run in client.list_run_infos(experiment_id)]

    return {'result': runs}





@app.route('/datasets')
def get_datasets():
    datasets = os.listdir('datasets')
    return {'datasets': datasets}

import pandas as pd

@app.route('/datasets/info')
def get_dataset_info():
    data = request.args.get('data')
    df = pd.read_csv('datasets/'+data)

    dataset_columns = list(df.columns)

    return {'dataset_columns': dataset_columns}


@app.route('/dataset/upload', methods = ['POST'])
def upload_file():
    file = request.files['file']
    print(file)
    return "done"


@app.route('/runs')
def get_run():
    run_id = request.args.get('run_id')
    run_obj = client.get_run(run_id)
    run = [{'metrics': run_obj.data.metrics,
           'params': run_obj.data.params,
           'tags': run_obj.data.tags
           }]

    print(run)
    return {'result': run}








@app.route('/experiments')
def experiments():
    exps = [{'experiment_id': e.experiment_id,
             'name': e.name}
            for e in client.list_experiments(view_type=ViewType.ALL)]

    return {'result': exps}