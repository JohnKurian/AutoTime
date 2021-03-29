import os
import sys
import inspect
import time

from hyperopt import fmin, tpe, hp, Trials
import pickle


#mlflow stuff
# import mlflow
# # import mlflow.tensorflow
# mlflow.set_tracking_uri("http://sim-mlflow.kn.crc.de.abb.com") # this is optional - if you don't specify, all mlflow runs will be logged locally
# # mlflow.keras.autolog()# this automatically logs data relevant for your model e.g. hyperparameters, evaluation metrics
# mlflow.set_experiment("john_temp_run_sktime")
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['AWS_SECRET_ACCESS_KEY'] = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
# os.environ['AWS_ACCESS_KEY_ID'] = "AKIAIOSFODNN7EXAMPLE"
# os.environ['MLFLOW_S3_ENDPOINT_URL']='http://sim-store.kn.crc.de.abb.com'
#
# #start mlflow
# # run = mlflow.start_run(run_id='dd4a2fcda08649fcbb5b14450c831a60')
# run = mlflow.start_run()

# currentdir = os.path.dirname(os.path.abspath(
#     inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

import os
import sys
import inspect

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

from autokeras import StructuredDataRegressor





# currentdir = os.path.dirname(os.path.abspath(
#     inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)


import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Activation
import math

from sklearn.metrics import r2_score


from warnings import simplefilter

import numpy as np
import pandas as pd

from sktime.datasets import load_airline
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    EnsembleForecaster,
    ReducedRegressionForecaster,
    TransformedTargetForecaster,
)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import sMAPE, smape_loss
from sktime.transformers.series.detrend import Deseasonalizer, Detrender
# from sktime.transformers.series.detrend._deseasonalize import Deseasonalizer
# from sktime.transformers.series.detrend._detrend import Detrender
from sktime.utils.plotting import plot_series

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.ets import AutoETS





def calculate_loss(yhat, test_y):
    smape = smape_loss(yhat, test_y)
    return smape




def load_data():
    dataset = pd.read_csv('../notebooks/datasets/monthly-sunspots.csv')
    y = dataset['Sunspots']
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    return y_train, y_test, fh


class BaseModel:
    def __init__(self, load_data, calculate_loss):
        self.calculate_loss = calculate_loss
        self.load_data = load_data
        self.y_train, self.y_test, self.fh = self.load_data()

    def fit_model(self, features={}):
        pass

    def predict(self, input):
        pass

    def save_model(self):
        pass



class KNNRegressor(BaseModel):
    def fit_model(self, features={}):
        y_train, y_test, fh = load_data()
        regressor = RandomForestRegressor()
        forecaster = ReducedRegressionForecaster(regressor, window_length=window_length)
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh)

        loss= calculate_loss(y_test, y_pred)
        return loss



def rf_regression(window_length):
    y_train, y_test, fh = load_data()
    regressor = RandomForestRegressor()
    forecaster = ReducedRegressionForecaster(regressor, window_length=window_length)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)

    loss= calculate_loss(y_test, y_pred)
    return loss





def kn_regression(neighbours,window_length):
    y_train, y_test, fh = load_data()
    regressor = KNeighborsRegressor(n_neighbors=neighbours)
    forecaster = ReducedRegressionForecaster(
        regressor=regressor, window_length=window_length, strategy="recursive"
    )
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    loss = calculate_loss(y_test, y_pred)
    return loss


def theta_forecaster(sp):
    y_train, y_test, fh = load_data()
    forecaster = ThetaForecaster(sp=sp)  # monthly seasonal periodicity
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    loss = calculate_loss(y_test, y_pred)
    return loss




# model_functions = [rf_regression, kn_regression, theta_forecaster]


import pickle, dill


def run_pipeline(cfg, config_space, param_indices, model_functions):

    cfg = np.rint(cfg).astype(int)

    loss = 0

    selected_model_index = cfg[0]
    selected_params = param_indices[selected_model_index]
    hyperparams = [cfg[i] for i in selected_params]


    try:
        loss, model = model_functions[selected_model_index](*hyperparams)
        with open('model.dat', 'wb') as fh:
            dill.dump(model, fh)
    except ValueError:
        print('error')
        loss = 99999

    print('loss:', loss)
    return loss



from mosaic.external.ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

cs = ConfigurationSpace()

#defining the boundaries (search space)
models = CategoricalHyperparameter("model", ["random_forest", "k_nearest_neighbor", "theta"])
rf_window_length = UniformIntegerHyperparameter("rf_window_length", 40, 70, default_value=55)
knn_neighours = UniformIntegerHyperparameter("knn_neighours", 20, 120, default_value=50)
knn_window_length = UniformIntegerHyperparameter("knn_window_length", 20, 120, default_value=50)
theta_sp = UniformIntegerHyperparameter("theta_sp", 20, 120, default_value=50)

#adding the models
cs.add_hyperparameters([models, rf_window_length, knn_neighours, knn_window_length, theta_sp])

#adding the conditionalities
cs.add_condition(InCondition(child=rf_window_length, parent=models, values=["random_forest"]))
cs.add_condition(InCondition(child=knn_neighours, parent=models, values=["k_nearest_neighbor"]))
cs.add_condition(InCondition(child=knn_window_length, parent=models, values=["k_nearest_neighbor"]))
cs.add_condition(InCondition(child=theta_sp, parent=models, values=["theta"]))









config_obj = {
    'random_forest' : {
        'window_length': [3,100,'int']
    },

    'k_nearest_neighbor' : {
            'neighbors': [1,25, 'int'],
            'window_length': [3,40,'int']
        },

    'theta' : {
            'th_sp': [1,50,'int']
        }
}



def initiate_run():




    space = {
        'algo': hp.choice('algo', [
            {'algo': 'sk_random_forest',
             'window_length': hp.uniformint('sk_random_forest_window_length', 3, 100)
             },
            {'algo': 'sk_knn',
             'neighbours': hp.uniformint('sk_knn_neighbours', 1, 50),
             'window_length': hp.uniformint('sk_knn_window_length', 3, 100)
             },
            {'algo': 'theta_forecaster',
             'sp': hp.uniformint('th_sp', 1, 50)
             }
        ])
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
            print(run.info.run_id)
            from mlflow.tracking.artifact_utils import _download_artifact_from_uri

            artifact_path = mlflow.get_artifact_uri() + '/trials.txt'
            print('the artifact path:', artifact_path)
            _download_artifact_from_uri(artifact_path, '')
            print('success')
        except:
            print('starting a new trial..')
            trials = Trials()
    Trials = None

    best = 0
    if Trials:
        print('here')
        best = fmin(run_pipeline, space, trials=trials, algo=tpe.suggest, max_evals=10000, trials_save_file='trials.txt')
    else:
        best = fmin(run_pipeline, space, trials=trials, algo=tpe.suggest, max_evals=10000)

    mlflow.get_artifact_uri()
    mlflow.end_run()


    return {'time': time.time()}








