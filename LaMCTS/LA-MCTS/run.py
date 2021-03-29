# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
from functions.functions import *
from functions.mujoco_functions import *
from lamcts import MCTS
import argparse
from machine_learning import run_pipeline


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






parser = argparse.ArgumentParser(description='Process inputs')
parser.add_argument('--func', help='specify the test function')
parser.add_argument('--dims', type=int, help='specify the problem dimensions')
parser.add_argument('--iterations', type=int, help='specify the iterations to collect in the search')


args = parser.parse_args()

f = None
iteration = 0
if args.func == 'ackley':
    assert args.dims > 0
    f = Ackley(dims =args.dims)
elif args.func == 'levy':
    assert args.dims > 0
    f = Levy(dims = args.dims)
elif args.func == 'lunar': 
    f = Lunarlanding()
elif args.func == 'swimmer':
    f = Swimmer()
elif args.func == 'hopper':
    f = Hopper()
elif args.func == 'automl':
    f = AutoML()
else:
    print('function not defined')
    os._exit(1)

assert f is not None
assert args.iterations > 0


f = Ackley(dims =args.dims)


# f = Ackley(dims = 10)
# f = Levy(dims = 10)
# f = Swimmer()
# f = Hopper()
# f = Lunarlanding()

import numpy as np
# define an objective function



# define a search space
from hyperopt import hp

space = {
    'algo': hp.choice('algo', [
        {'algo0': 'cat0',
         'cat0': hp.choice('algo0', [
            {'algo': [0,0],
            'cat000': hp.uniform('cat000', -5, 10),
            'cat001': hp.uniform('cat001', -5, 10),
            'cat002': hp.uniform('cat002', -5, 10),
            'cat003': hp.uniform('cat003', -5, 10),
            'cat004': hp.uniform('cat004', -5, 10),
            'cat005': hp.uniform('cat005', -5, 10),
            'cat006': hp.uniform('cat006', -5, 10),
            'cat007': hp.uniform('cat007', -5, 10)
             },
             {'algo': [0,1],
              'cat010': hp.uniform('cat010', -5, 10),
              'cat011': hp.uniform('cat011', -5, 10),
              'cat012': hp.uniform('cat012', -5, 10),
              'cat013': hp.uniform('cat013', -5, 10),
              'cat014': hp.uniform('cat014', -5, 10),
              'cat015': hp.uniform('cat015', -5, 10),
              'cat016': hp.uniform('cat016', -5, 10),
              'cat017': hp.uniform('cat017', -5, 10)
              },
             {'algo': [0,2],
              'cat020': hp.uniform('cat020', -5, 10),
              'cat021': hp.uniform('cat021', -5, 10),
              'cat022': hp.uniform('cat022', -5, 10),
              'cat023': hp.uniform('cat023', -5, 10),
              'cat024': hp.uniform('cat024', -5, 10),
              'cat025': hp.uniform('cat025', -5, 10),
              'cat026': hp.uniform('cat026', -5, 10),
              'cat027': hp.uniform('cat027', -5, 10)
              }

         ])

         },
        {'algo1': 'cat1',
                 'cat1': hp.choice('algo1', [
                    {'algo': [1,0],
                    'cat100': hp.uniform('cat100', -5, 10),
                    'cat101': hp.uniform('cat101', -5, 10),
                    'cat102': hp.uniform('cat102', -5, 10),
                    'cat103': hp.uniform('cat103', -5, 10),
                    'cat104': hp.uniform('cat104', -5, 10),
                    'cat105': hp.uniform('cat105', -5, 10),
                    'cat106': hp.uniform('cat106', -5, 10),
                    'cat107': hp.uniform('cat107', -5, 10)
                     },
                     {'algo': [1,1],
                      'cat110': hp.uniform('cat110', -5, 10),
                      'cat111': hp.uniform('cat111', -5, 10),
                      'cat112': hp.uniform('cat112', -5, 10),
                      'cat113': hp.uniform('cat113', -5, 10),
                      'cat114': hp.uniform('cat114', -5, 10),
                      'cat115': hp.uniform('cat115', -5, 10),
                      'cat116': hp.uniform('cat116', -5, 10),
                      'cat117': hp.uniform('cat117', -5, 10)
                      },
                     {'algo': [1,2],
                      'cat120': hp.uniform('cat120', -5, 10),
                      'cat121': hp.uniform('cat121', -5, 10),
                      'cat122': hp.uniform('cat122', -5, 10),
                      'cat123': hp.uniform('cat123', -5, 10),
                      'cat124': hp.uniform('cat124', -5, 10),
                      'cat125': hp.uniform('cat125', -5, 10),
                      'cat126': hp.uniform('cat126', -5, 10),
                      'cat127': hp.uniform('cat127', -5, 10)
                      }

                 ])

                 },
        {'algo1': 'cat2',
                 'cat2': hp.choice('algo2', [
                    {'algo': [2,0],
                    'cat200': hp.uniform('cat200', -5, 10),
                    'cat201': hp.uniform('cat201', -5, 10),
                    'cat202': hp.uniform('cat202', -5, 10),
                    'cat203': hp.uniform('cat203', -5, 10),
                    'cat204': hp.uniform('cat204', -5, 10),
                    'cat205': hp.uniform('cat205', -5, 10),
                    'cat206': hp.uniform('cat206', -5, 10),
                    'cat207': hp.uniform('cat207', -5, 10)
                     },
                     {'algo': [2,1],
                      'cat210': hp.uniform('cat210', -5, 10),
                      'cat211': hp.uniform('cat211', -5, 10),
                      'cat212': hp.uniform('cat212', -5, 10),
                      'cat213': hp.uniform('cat213', -5, 10),
                      'cat214': hp.uniform('cat214', -5, 10),
                      'cat215': hp.uniform('cat215', -5, 10),
                      'cat216': hp.uniform('cat216', -5, 10),
                      'cat217': hp.uniform('cat217', -5, 10)
                      },
                     {'algo': [2,2],
                      'cat220': hp.uniform('cat220', -5, 10),
                      'cat221': hp.uniform('cat221', -5, 10),
                      'cat222': hp.uniform('cat222', -5, 10),
                      'cat223': hp.uniform('cat223', -5, 10),
                      'cat224': hp.uniform('cat224', -5, 10),
                      'cat225': hp.uniform('cat225', -5, 10),
                      'cat226': hp.uniform('cat226', -5, 10),
                      'cat227': hp.uniform('cat227', -5, 10)
                      }

                 ])

                 },
        {'algo3': 'cat3',
                 'cat3': hp.choice('algo3', [
                    {'algo': [3,0],
                    'cat300': hp.uniform('cat300', -5, 10),
                    'cat301': hp.uniform('cat301', -5, 10),
                    'cat302': hp.uniform('cat302', -5, 10),
                    'cat303': hp.uniform('cat303', -5, 10),
                    'cat304': hp.uniform('cat304', -5, 10),
                    'cat305': hp.uniform('cat305', -5, 10),
                    'cat306': hp.uniform('cat306', -5, 10),
                    'cat307': hp.uniform('cat307', -5, 10)
                     },
                     {'algo': [3,1],
                      'cat310': hp.uniform('cat310', -5, 10),
                      'cat311': hp.uniform('cat311', -5, 10),
                      'cat312': hp.uniform('cat312', -5, 10),
                      'cat313': hp.uniform('cat313', -5, 10),
                      'cat314': hp.uniform('cat314', -5, 10),
                      'cat315': hp.uniform('cat315', -5, 10),
                      'cat316': hp.uniform('cat316', -5, 10),
                      'cat317': hp.uniform('cat317', -5, 10)
                      },
                     {'algo': [3,2],
                      'cat320': hp.uniform('cat320', -5, 10),
                      'cat321': hp.uniform('cat321', -5, 10),
                      'cat322': hp.uniform('cat322', -5, 10),
                      'cat323': hp.uniform('cat323', -5, 10),
                      'cat324': hp.uniform('cat324', -5, 10),
                      'cat325': hp.uniform('cat325', -5, 10),
                      'cat326': hp.uniform('cat326', -5, 10),
                      'cat327': hp.uniform('cat327', -5, 10)
                      }

                 ])

                 }
    ])
}

automl_space = {
    'algo': hp.choice('algo', [
        {'algo': 'sk_random_forest',
         'window_length': hp.uniformint('sk_random_forest_window_length', 3, 100)
         },
        {'algo': 'sk_knn',
         'neighbours': hp.uniformint('sk_knn_neighbours', 1, 25),
         'window_length': hp.uniformint('sk_knn_window_length', 3, 40)
         },
        {'algo': 'theta_forecaster',
         'sp': hp.uniformint('th_sp', 1, 50)
         }
    ])
}





def hyperopt_objective(args):
    # print(args)
    x = np.array(list(args['algo'].values())[1:])[0]
    print(x)

    category1 = x['algo'][0]
    category2 = x['algo'][1]

    x = np.array(list(x.values())[1:])
    print(x)
    result = None

    if category1 == 0:

        if category2 == 0:
            result = 8 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)

            return result

        elif category2 == 1:
            result = 5 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)

            return result

        elif category2 == 2:
            result = 10 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)

            return result

    elif category1 == 1:

        if category2 == 0:
            result = 4 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)

            return result

        elif category2 == 1:
            result = 8 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)

            return result

        elif category2 == 2:
            result = 17 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)

            return result

    elif category1 == 2:

        if category2 == 0:
            result = 23 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)

            return result

        elif category2 == 1:
            result = 15 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)

            return result

        elif category2 == 2:
            result = 20 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)

            return result

    elif category1 == 3:

        if category2 == 0:
            result = 50 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)

            return result

        elif category2 == 1:
            result = 60 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)

            return result

        elif category2 == 2:
            result = (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)

            return result








# # minimize the objective over the space
# from hyperopt import fmin, tpe
# best = fmin(hyperopt_objective, space, algo=tpe.suggest, max_evals=1000)


import optuna







# study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
# study.optimize(optuna_objective, n_trials=100)
# #best: 4.73

# study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
# study.optimize(optuna_objective, n_trials=100)
#best: 9.5
#
# study = optuna.create_study(sampler=optuna.samplers.TPESampler())
# study.optimize(optuna_objective, n_trials=100)
#best: 6.5
#



# study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), pruner=optuna.pruners.MedianPruner())
# study.optimize(optuna_objective, n_trials=100)
# #best: 4.25

# study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.MedianPruner())
# study.optimize(optuna_objective, n_trials=100)
#best: 10.5
#
# study = optuna.create_study(sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
# study.optimize(optuna_objective, n_trials=100)
#best: 7.42





# study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), pruner=optuna.pruners.SuccessiveHalvingPruner())
# study.optimize(optuna_objective, n_trials=100)
# #best: 4.13

# study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.SuccessiveHalvingPruner())
# study.optimize(optuna_objective, n_trials=100)
#best: 9.8
#
# study = optuna.create_study(sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.SuccessiveHalvingPruner())
# study.optimize(optuna_objective, n_trials=100)
#best: 8.44





# study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), pruner=optuna.pruners.HyperbandPruner())
# study.optimize(optuna_ackley_2level_objective, n_trials=5000)
# #best: 4.38

# study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.HyperbandPruner())
# study.optimize(optuna_objective, n_trials=1000)
#best: 9.128
#
# study = optuna.create_study(sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
# study.optimize(optuna_ackley_2level_objective, n_trials=5000)
#best: 6.04
#best: 11.05


# f = Cat1Ackley(dims = 1)
#
# from mosaic.external.ConfigSpace.configuration_space import ConfigurationSpace
# from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
#     UniformFloatHyperparameter, UniformIntegerHyperparameter
# from ConfigSpace.conditions import InCondition
#
#
# cs = ConfigurationSpace()
# model = CategoricalHyperparameter("model", ["random_forest", "k_nearest_neighbor", "theta"])

def calculate_loss(yhat, test_y):
    smape = smape_loss(yhat, test_y)
    return smape



def load_data():
    dataset = pd.read_csv('../notebooks/datasets/monthly-sunspots.csv')
    y = dataset['Sunspots']
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    return y_train, y_test, fh

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




model_functions = [rf_regression, kn_regression, theta_forecaster]



f = AutoML(config_space, model_functions)

agent = MCTS(
             lb = f.lb,              # the lower bound of each problem dimensions
             ub = f.ub,              # the upper bound of each problem dimensions
             dims = f.dims,          # the problem dimensions
             ninits = f.ninits,      # the number of random samples used in initializations
             func = f,               # function object to be optimized
             Cp = f.Cp,              # Cp for MCTS
             # categories=f.categories,
             # dependencies=f.dependencies,
             leaf_size = f.leaf_size, # tree leaf size
             kernel_type = f.kernel_type, #SVM configruation
             gamma_type = f.gamma_type,    #SVM configruation

             )

x = agent.search(iterations = 1000)




print('this is:', x)




# for i in range(100):
#     f = Custom2LevelAckleyNew(dims = 50)
#
#
#
#     agent = MCTS(
#                  lb = f.lb,              # the lower bound of each problem dimensions
#                  ub = f.ub,              # the upper bound of each problem dimensions
#                  dims = f.dims,          # the problem dimensions
#                  ninits = f.ninits,      # the number of random samples used in initializations
#                  func = f,               # function object to be optimized
#                  Cp = f.Cp,              # Cp for MCTS
#                  # categories=f.categories,
#                  # dependencies=f.dependencies,
#                  leaf_size = f.leaf_size, # tree leaf size
#                  kernel_type = f.kernel_type, #SVM configruation
#                  gamma_type = f.gamma_type,    #SVM configruation
#
#                  )
#
#     # agent.load_agent()
#     import pickle
#     agent = None
#     with open('mcts-agent_root.dat', 'rb') as json_data:
#         agent = pickle.load(json_data)
#         print("=====>loads:", len(agent.samples)," samples" )
#     agent.sample_counter = 0
#     x = agent.search(iterations = 1)
#     print('hey:', x)
#     input()
# print('this is:', x)
#best: 3.147










###########################################################################
############################ LEVY #########################################
###########################################################################

# f = Levy(dims = 10)
#
# agent = MCTS(
#              lb = f.lb,              # the lower bound of each problem dimensions
#              ub = f.ub,              # the upper bound of each problem dimensions
#              dims = f.dims,          # the problem dimensions
#              ninits = f.ninits,      # the number of random samples used in initializations
#              func = f,               # function object to be optimized
#              Cp = f.Cp,              # Cp for MCTS
#              # categories=f.categories,
#              # dependencies=f.dependencies,
#              leaf_size = f.leaf_size, # tree leaf size
#              kernel_type = f.kernel_type, #SVM configruation
#              gamma_type = f.gamma_type,    #SVM configruation
#
#              )
#
# x = agent.search(iterations = 100)
# print('this is:', x)


# study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
# study.optimize(optuna_levy_objective, n_trials=400)
#best: 4.73




# f = Custom2LevelAckleyNew(dims = 50)



# f = AutoML()
#
#
# agent = MCTS(
#              lb = f.lb,              # the lower bound of each problem dimensions
#              ub = f.ub,              # the upper bound of each problem dimensions
#              dims = f.dims,          # the problem dimensions
#              ninits = f.ninits,      # the number of random samples used in initializations
#              func = f,               # function object to be optimized
#              Cp = f.Cp,              # Cp for MCTS
#              # categories=f.categories,
#              # dependencies=f.dependencies,
#              leaf_size = f.leaf_size, # tree leaf size
#              kernel_type = f.kernel_type, #SVM configruation
#              gamma_type = f.gamma_type,    #SVM configruation
#
#              )
#
# x = agent.search(iterations = 100)

#
# study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), pruner=optuna.pruners.HyperbandPruner())
# study.optimize(optuna_automl_objective, n_trials=5000)

# search_space = {"x1": [0, 2], "x2": [3, 100], "x3": [1, 25], "x4": [3, 40], "x5": [3, 50]}
# study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
# study.optimize(optuna_automl_objective, n_trials=5000)

# study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.HyperbandPruner())
# study.optimize(optuna_automl_objective, n_trials=100)
#
# study = optuna.create_study(sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.SuccessiveHalvingPruner())
# study.optimize(optuna_automl_objective, n_trials=1000)


# from hyperopt import fmin, tpe
# best = fmin(hyperopt_automl_objective, automl_space, algo=tpe.suggest, max_evals=1000)

from os.path import exists


class AutoML_rf:
    def __init__(self, dims=1):
        self.dims = dims
        self.lb = np.array([3])
        self.ub = np.array([100])
        self.categories = [0]
        self.dependencies = [(1,), (2, 3), (4,)]
        self.counter = 0
        # self.tracker = tracker('AutoML' + str(dims))

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 1
        self.leaf_size = 10
        self.ninits = 40
        self.kernel_type = "rbf"
        self.gamma_type = "auto"

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        x_hat = [0,0,0,0,0]
        x_hat[0] = 0
        x_hat[1] = x[0]
        result = run_pipeline(x_hat)
        # self.tracker.track(result)

        return result


class AutoML_knn:
    def __init__(self, dims=2):
        self.dims = dims
        self.lb = np.array([1, 3])
        self.ub = np.array([25, 40])
        self.categories = [0]
        self.dependencies = [(1,), (2, 3), (4,)]
        self.counter = 0
        # self.tracker = tracker('AutoML' + str(dims))

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 1
        self.leaf_size = 10
        self.ninits = 40
        self.kernel_type = "rbf"
        self.gamma_type = "auto"

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        x_hat = [0,0,0,0,0]
        x_hat[0] = 1
        x_hat[2] = x[0]
        x_hat[3] = x[1]
        result = run_pipeline(x_hat)
        # self.tracker.track(result)

        return result


class AutoML_theta:
    def __init__(self, dims=1):
        self.dims = dims
        self.lb = np.array([1])
        self.ub = np.array([50])
        self.categories = [0]
        self.dependencies = [(1,), (2, 3), (4,)]
        self.counter = 0
        # self.tracker = tracker('AutoML' + str(dims))

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 1
        self.leaf_size = 10
        self.ninits = 40
        self.kernel_type = "rbf"
        self.gamma_type = "auto"

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        x_hat = [0, 0, 0, 0, 0]
        x_hat[0] = 2
        x_hat[4] = x[0]
        result = run_pipeline(x_hat)
        # self.tracker.track(result)

        return result

class AutoML_root:
    def __init__(self, dims=1):
        self.dims = dims
        self.lb = np.array([0])
        self.ub = np.array([2])
        self.categories = [0]
        self.dependencies = [(1,), (2, 3), (4,)]
        self.counter = 0
        # self.tracker = tracker('AutoML' + str(dims))

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 1
        self.leaf_size = 10
        self.ninits = 40
        self.kernel_type = "rbf"
        self.gamma_type = "auto"

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        result = None

        # x = [0, 0, 0, 0, 0]

        if np.rint(x[0]).astype(int) == 0:
            f = AutoML_rf(dims=1)

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
                name='automl_rf'

            )

            # agent.load_agent()
            import pickle
            # agent = None

            if exists('automl_rf.dat'):
                with open('automl_rf.dat', 'rb') as json_data:
                    agent = pickle.load(json_data)
                    print("=====>loads:", len(agent.samples), " samples")
                agent.sample_counter = 0
            result = agent.search(iterations=1)


        if np.rint(x[0]).astype(int) == 1:
            f = AutoML_knn(dims=2)

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
                name='automl_knn'

            )

            # agent.load_agent()
            import pickle
            # agent = None

            if exists('automl_knn.dat'):
                with open('automl_knn.dat', 'rb') as json_data:
                    agent = pickle.load(json_data)
                    print("=====>loads:", len(agent.samples), " samples")
                agent.sample_counter = 0
            result = agent.search(iterations=1)



        if np.rint(x[0]).astype(int) == 2:
            f = AutoML_theta(dims=1)

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
                gamma_type=f.gamma_type,
                name='automl_theta'# SVM configruation

            )

            # agent.load_agent()
            import pickle
            # agent = None

            if exists('automl_theta.dat'):
                with open('automl_theta.dat', 'rb') as json_data:
                    agent = pickle.load(json_data)
                    print("=====>loads:", len(agent.samples), " samples")
                agent.sample_counter = 0
            result = agent.search(iterations=1)

        print('returing result:', result)

        return result



        # result = run_pipeline(x)
        # self.tracker.track(result)

        # return result





# f = AutoML(dims=5)
#
#
# agent = MCTS(
#              lb = f.lb,              # the lower bound of each problem dimensions
#              ub = f.ub,              # the upper bound of each problem dimensions
#              dims = f.dims,          # the problem dimensions
#              ninits = f.ninits,      # the number of random samples used in initializations
#              func = f,               # function object to be optimized
#              Cp = f.Cp,              # Cp for MCTS
#              # categories=f.categories,
#              # dependencies=f.dependencies,
#              leaf_size = f.leaf_size, # tree leaf size
#              kernel_type = f.kernel_type, #SVM configruation
#              gamma_type = f.gamma_type,    #SVM configruation
#
#              )
#
# x = agent.search(iterations = 100)
"""
FAQ:

1. How to retrieve every f(x) during the search?

During the optimization, the function will create a folder to store the f(x) trace; and
the name of the folder is in the format of function name + function dimensions, e.g. Ackley10.

Every 100 samples, the function will write a row to a file named results + total samples, e.g. result100 
mean f(x) trace in the first 100 samples.

Each last row of result file contains the f(x) trace starting from 1th sample -> the current sample.
Results of previous rows are from previous experiments, as we always append the results from a new experiment
to the last row.

Here is an example to interpret a row of f(x) trace.
[5, 3.2, 2.1, ..., 1.1]
The first sampled f(x) is 5, the second sampled f(x) is 3.2, and the last sampled f(x) is 1.1 

2. How to improve the performance?
Tune Cp, leaf_size, and improve BO sampler with others.

"""
