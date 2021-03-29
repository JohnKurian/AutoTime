from functions.functions import *
from functions.mujoco_functions import *
from lamcts import MCTS


import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    ReducedRegressionForecaster
)
from sktime.forecasting.model_selection import (
    temporal_train_test_split,
)
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import sMAPE, smape_loss
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor



def run_automl_search(config_space, model_functions):
    f = AutoML(config_space, model_functions)

    agent = MCTS(
        lb=f.lb,  # the lower bound of each problem dimensions
        ub=f.ub,  # the upper bound of each problem dimensions
        dims=f.dims,  # the problem dimensions
        ninits=f.ninits,  # the number of random samples used in initializations
        func=f,  # function object to be optimized
        Cp=f.Cp,  # Cp for MCTS
        # categories=f.categories,
        # dependencies=f.dependencies,
        leaf_size=f.leaf_size,  # tree leaf size
        kernel_type=f.kernel_type,  # SVM configruation
        gamma_type=f.gamma_type,  # SVM configruation

    )

    x = agent.search(iterations=1000)
    return x






def calculate_loss(yhat, test_y):
    smape = smape_loss(yhat, test_y)
    return smape



def load_data():
    dataset = pd.read_csv('datasets/monthly-sunspots.csv')
    y = dataset['Sunspots']
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    return y_train, y_test, fh


def rf_regression(window_length):

    #load the dataset
    y_train, y_test, fh = load_data()

    #define and fit the model
    regressor = RandomForestRegressor()
    forecaster = ReducedRegressionForecaster(regressor, window_length=window_length)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)


    #calculate the loss
    loss= calculate_loss(y_test, y_pred)

    return loss, forecaster


def kn_regression(neighbours,window_length):
    y_train, y_test, fh = load_data()
    regressor = KNeighborsRegressor(n_neighbors=neighbours)
    forecaster = ReducedRegressionForecaster(
        regressor=regressor, window_length=window_length, strategy="recursive"
    )
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    loss = calculate_loss(y_test, y_pred)
    return loss, forecaster


def theta_forecaster(sp):
    y_train, y_test, fh = load_data()
    forecaster = ThetaForecaster(sp=sp)  # monthly seasonal periodicity
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    loss = calculate_loss(y_test, y_pred)
    return loss, forecaster


model_functions = [rf_regression, kn_regression, theta_forecaster]

config_space = {
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


best_configuration = run_automl_search(config_space, model_functions)