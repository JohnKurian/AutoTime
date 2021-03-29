# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import numpy as np
import gym
import json
import os
from machine_learning import run_pipeline
import imageio


class tracker:
    def __init__(self, foldername):
        self.counter   = 0
        self.results   = []
        self.curt_best = float("inf")
        self.foldername = foldername
        try:
            os.mkdir(foldername)
        except OSError:
            print ("Creation of the directory %s failed" % foldername)
        else:
            print ("Successfully created the directory %s " % foldername)
        
    def dump_trace(self):
        trace_path = self.foldername + '/result' + str(len( self.results) )
        final_results_str = json.dumps(self.results)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')
            
    def track(self, result):
        if result < self.curt_best:
            self.curt_best = result
        self.results.append(self.curt_best)
        if len(self.results) % 100 == 0:
            self.dump_trace()

class Levy:
    def __init__(self, dims=10):
        self.dims        = dims
        self.lb          = -10 * np.ones(dims)
        self.ub          =  10 * np.ones(dims)
        self.tracker     = tracker('Levy'+str(dims))
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp          = 10
        self.leaf_size   = 8
        self.kernel_type = "poly"
        self.ninits      = 40
        self.gamma_type   = "auto"
        print("initialize levy at dims:", self.dims)
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        w = []
        for idx in range(0, len(x)):
            w.append( 1 + (x[idx] - 1) / 4 )
        w = np.array(w)
        
        term1 = ( np.sin( np.pi*w[0] ) )**2;
        
        term3 = ( w[-1] - 1 )**2 * ( 1 + ( np.sin( 2 * np.pi * w[-1] ) )**2 );
        
        
        term2 = 0;
        for idx in range(1, len(w) ):
            wi  = w[idx]
            new = (wi-1)**2 * ( 1 + 10 * ( np.sin( np.pi* wi + 1 ) )**2)
            term2 = term2 + new
        
        result = term1 + term2 + term3
        self.tracker.track( result )

        return result

class Ackley:
    def __init__(self, dims=10):
        self.dims      = dims
        self.lb        = -5 * np.ones(dims)
        self.ub        =  10 * np.ones(dims)
        self.counter   = 0
        self.tracker   = tracker('Ackley'+str(dims) )
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp        = 1
        self.leaf_size = 10
        self.ninits    = 40
        self.kernel_type = "rbf"
        self.gamma_type  = "auto"
        self.categories = []
        self.dependencies = []
        
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
        self.tracker.track( result )
                
        return result


class Custom1LevelAckley:
    def __init__(self, dims=10):
        self.dims = dims
        self.lb = np.array([0, -5, -5, -5, -5, -5, -5, -5, -5, -5])
        self.ub = np.array([3, 10, 10, 10, 10, 10, 10, 10, 10, 10])

        self.counter = 0
        self.tracker = tracker('Ackley' + str(dims))

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 1
        self.leaf_size = 10
        self.ninits = 40
        self.kernel_type = "rbf"
        self.gamma_type = "auto"
        self.categories = []
        self.dependencies = []

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        category = int(round(x[0]))

        x = x[1:]

        if category == 0:

            result = 4 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
            self.tracker.track(result)

            return result

        elif category == 1:
            result = 3 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
            self.tracker.track(result)

            return result

        elif category == 2:
            result = 2 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
            self.tracker.track(result)

            return result

        elif category == 3:
            result = (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
            self.tracker.track(result)

            return result




class Custom2LevelAckley:
    def __init__(self, dims=10):
        self.dims = dims
        self.lb = np.array([0, 0, -5, -5, -5, -5, -5, -5, -5, -5])
        self.ub = np.array([3, 2, 10, 10, 10, 10, 10, 10, 10, 10])

        self.counter = 0
        self.tracker = tracker('Ackley' + str(dims))

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 1
        self.leaf_size = 10
        self.ninits = 40
        self.kernel_type = "rbf"
        self.gamma_type = "auto"
        self.categories = []
        self.dependencies = []

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        category1 = int(round(x[0]))
        category2 = int(round(x[1]))

        x = x[2:]

        if category1 == 0:

            if category2 == 0:
                result = 8 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 1:
                result = 5 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 2:
                result = 10 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

        elif category1 == 1:

            if category2 == 0:
                result = 4 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 1:
                result = 8 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 2:
                result = 17 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

        elif category1 == 2:

            if category2 == 0:
                result = 23 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 1:
                result = 15 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 2:
                result = 20 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

        elif category1 == 3:

            if category2 == 0:
                result = 50 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 1:
                result = 60 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 2:
                result = (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result



class Cat1Ackley:
    def __init__(self, dims=1):
        self.dims = dims
        self.lb = np.array([0])
        self.ub = np.array([3])

        self.counter = 0
        self.tracker = tracker('AckleyCat1' + str(dims))

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 1
        self.leaf_size = 10
        self.ninits = 40
        self.kernel_type = "rbf"
        self.gamma_type = "auto"
        self.categories = []
        self.dependencies = []

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        category1 = int(round(x[0]))

        f = Cat2LevelAckleyNew(dims=50)

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

        # agent.load_agent()
        import pickle
        agent = None
        with open('mcts-agent.dat', 'rb') as json_data:
            agent = pickle.load(json_data)
            print("=====>loads:", len(agent.samples), " samples")
        agent.sample_counter = 0
        x = agent.search(iterations=100)




class Custom2LevelAckleyNew:
    def __init__(self, dims=10):
        self.dims = dims
        self.lb = np.append([0,0], np.repeat(-5, 48))
        self.ub = np.append([3,2], np.repeat(10, 48))

        self.counter = 0
        self.tracker = tracker('Ackley' + str(dims))

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 1
        self.leaf_size = 10
        self.ninits = 40
        self.kernel_type = "rbf"
        self.gamma_type = "auto"
        self.categories = []
        self.dependencies = []

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        category1 = int(round(x[0]))
        category2 = int(round(x[1]))

        x = x[2:]

        if category1 == 0:

            if category2 == 0:
                x = x[(2 + 12*category1 + 4*category2):(2 + 12*category1 + 4*category2 +4)]
                result = 8 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 1:
                x = x[(2 + 12*category1 + 4*category2):(2 + 12*category1 + 4*category2 +4)]
                result = 5 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 2:
                x = x[(2 + 12*category1 + 4*category2):(2 + 12*category1 + 4*category2 +4)]
                result = 10 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

        elif category1 == 1:

            if category2 == 0:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 4 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 1:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 8 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 2:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 17 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

        elif category1 == 2:

            if category2 == 0:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 23 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 1:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 15 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 2:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 20 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

        elif category1 == 3:

            if category2 == 0:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 50 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 1:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 60 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 2:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result






class Custom2LevelAckleyNew:
    def __init__(self, dims=10):
        self.dims = dims
        self.lb = np.append([0,0], np.repeat(-5, 48))
        self.ub = np.append([3,2], np.repeat(10, 48))

        self.counter = 0
        self.tracker = tracker('Ackley' + str(dims))

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 1
        self.leaf_size = 10
        self.ninits = 40
        self.kernel_type = "rbf"
        self.gamma_type = "auto"
        self.categories = []
        self.dependencies = []

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        category1 = int(round(x[0]))
        category2 = int(round(x[1]))

        x = x[2:]

        if category1 == 0:

            if category2 == 0:
                x = x[(2 + 12*category1 + 4*category2):(2 + 12*category1 + 4*category2 +4)]
                result = 8 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 1:
                x = x[(2 + 12*category1 + 4*category2):(2 + 12*category1 + 4*category2 +4)]
                result = 5 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 2:
                x = x[(2 + 12*category1 + 4*category2):(2 + 12*category1 + 4*category2 +4)]
                result = 10 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

        elif category1 == 1:

            if category2 == 0:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 4 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 1:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 8 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 2:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 17 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

        elif category1 == 2:

            if category2 == 0:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 23 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 1:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 15 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 2:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 20 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

        elif category1 == 3:

            if category2 == 0:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 50 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 1:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = 60 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result

            elif category2 == 2:
                x = x[(2 + 12 * category1 + 4 * category2):(2 + 12 * category1 + 4 * category2 + 4)]
                result = (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
                    np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
                self.tracker.track(result)

                return result


def get_bounds(config_space):
    total_bound_dim = sum([len(i) for i in config_space.values()]) + 1
    lower_bounds = [0]
    upper_bounds = [len(config_space.keys()) - 1]
    param_indices = []
    curr_param_index = 0
    for key in config_space.keys():
        param_idx = []
        for hp in config_space[key].keys():
            curr_param_index = curr_param_index + 1
            lower_bounds.append(config_space[key][hp][0])
            upper_bounds.append(config_space[key][hp][1])
            param_idx.append(curr_param_index)
        param_indices.append(param_idx)

    return np.array(lower_bounds), np.array(upper_bounds), param_indices



class AutoML:
    def __init__(self, config_space=None, model_functions=[]):
        self.model_functions = model_functions

        self.dims = sum([len(i) for i in config_space.values()]) + 1
        self.config_space = config_space


        self.lb, self.ub, self.param_indices = get_bounds(config_space)


        self.categories = [0]
        self.dependencies = [(1,), (2,3), (4,)]
        self.counter = 0
        self.tracker = tracker('AutoML' + str(self.dims))

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
        result = run_pipeline(x, self.config_space, self.param_indices, self.model_functions)
        self.tracker.track(result)

        return result







def optuna_automl_objective(trial):
    x1 = trial.suggest_int('x1', 0, 2)
    x2 = trial.suggest_int('x2', 3, 100)
    x3 = trial.suggest_int('x3', 1, 25) #25
    x4 = trial.suggest_int('x4', 3, 40) #40
    x5 = trial.suggest_int('x5', 1, 50)

    x = [x1,x2,x3,x4,x5]
    result = run_pipeline(x)
    return result


def hyperopt_automl_objective(args):
    print(args['algo'])

    x = [0, 0, 0, 0, 0]
    result = None
    if args['algo']['algo'] == 'sk_random_forest':
        x[0] = 0
        x[1] = args['algo']['window_length']
        result = run_pipeline(x)
    elif args['algo']['algo'] == 'sk_knn':
        x[0] = 1
        x[2] = args['algo']['neighbours']
        x[3] = args['algo']['window_length']
        result = run_pipeline(x)
    elif args['algo']['algo'] == 'theta_forecaster':
        x[0] = 2
        x[4] = args['algo']['sp']
        result = run_pipeline(x)

    return result


        
class Lunarlanding:
    def __init__(self):
        self.dims = 12
        self.lb   = np.zeros(12)
        self.ub   = 2 * np.ones(12)
        self.counter = 0
        self.env = gym.make('LunarLander-v2')
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp          = 50
        self.leaf_size   = 10
        self.kernel_type = "poly"
        self.ninits      = 40
        self.gamma_type  = "scale"
        
        self.render      = False
        
        
    def heuristic_Controller(self, s, w):
        angle_targ = s[0] * w[0] + s[2] * w[1]
        if angle_targ > w[2]:
            angle_targ = w[2]
        if angle_targ < -w[2]:
            angle_targ = -w[2]
        hover_targ = w[3] * np.abs(s[0])

        angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
        hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

        if s[6] or s[7]:
            angle_todo = w[8]
            hover_todo = -(s[3]) * w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
        return a
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
    
        total_rewards = []
        for i in range(0, 3): # controls the number of episode/plays per trial
            state = self.env.reset()
            rewards_for_episode = []
            num_steps = 2000
        
            for step in range(num_steps):
                if self.render:
                    self.env.render()
                received_action = self.heuristic_Controller(state, x)
                next_state, reward, done, info = self.env.step(received_action)
                rewards_for_episode.append( reward )
                state = next_state
                if done:
                     break
                        
            rewards_for_episode = np.array(rewards_for_episode)
            total_rewards.append( np.sum(rewards_for_episode) )
        total_rewards = np.array(total_rewards)
        mean_rewards = np.mean( total_rewards )
        
        return mean_rewards*-1

def optuna_ackley_2level_objective(trial):
    x1 = trial.suggest_int('x1', 0, 4)
    category2 = trial.suggest_int('x2', 0, 3)

    category1 = x1

    subcattrials = [trial.suggest_uniform('x' + str(i), -5, 10) for i in range(3, 51)]

    x = np.array(subcattrials[(12*category1 + 4*category2):(12*category1 + 4*category2 +4)])
    # category2 = x2

    # if category1 == 2:
    #     print(category1, category2)

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
    print(result)


def optuna_objective(trial):
    x1 = trial.suggest_int('x1', -5, 10)
    x2 = trial.suggest_uniform('x2', -5, 10)
    x3 = trial.suggest_uniform('x3', -5, 10)
    x4 = trial.suggest_uniform('x4', -5, 10)
    x5 = trial.suggest_uniform('x5', -5, 10)

    x6 = trial.suggest_uniform('x6', -5, 10)
    x7 = trial.suggest_uniform('x7', -5, 10)
    x8 = trial.suggest_uniform('x9', -5, 10)
    x9 = trial.suggest_uniform('x10', -5, 10)
    x10 = trial.suggest_uniform('x', -5, 10)

    x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]

    x = np.array(x)

    return (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
        np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)




def optuna_levy_objective(trial):
    x1 = trial.suggest_int('x1', -5, 10)
    x2 = trial.suggest_uniform('x2', -5, 10)
    x3 = trial.suggest_uniform('x3', -5, 10)
    x4 = trial.suggest_uniform('x4', -5, 10)
    x5 = trial.suggest_uniform('x5', -5, 10)

    x6 = trial.suggest_uniform('x6', -5, 10)
    x7 = trial.suggest_uniform('x7', -5, 10)
    x8 = trial.suggest_uniform('x9', -5, 10)
    x9 = trial.suggest_uniform('x10', -5, 10)
    x10 = trial.suggest_uniform('x', -5, 10)

    x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]

    x = np.array(x)

    w = []
    for idx in range(0, len(x)):
        w.append(1 + (x[idx] - 1) / 4)
    w = np.array(w)

    term1 = (np.sin(np.pi * w[0])) ** 2;

    term3 = (w[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[-1])) ** 2);

    term2 = 0;
    for idx in range(1, len(w)):
        wi = w[idx]
        new = (wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2)
        term2 = term2 + new

    result = term1 + term2 + term3

    return result
    
    
    
    
    
# def hyperopt_levy_objective(args):
#     # print(args)
#     x = np.array(list(args['algo'].values())[1:])[0]
#     print(x)
#
#     category1 = x['algo'][0]
#     category2 = x['algo'][1]
#
#     x = np.array(list(x.values())[1:])
#     print(x)
#     result = None
#
#     if category1 == 0:
#
#         if category2 == 0:
#             result = 8 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
#                 np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
#
#             return result
#
#         elif category2 == 1:
#             result = 5 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
#                 np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
#
#             return result
#
#         elif category2 == 2:
#             result = 10 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
#                 np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
#
#             return result
#
#     elif category1 == 1:
#
#         if category2 == 0:
#             result = 4 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
#                 np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
#
#             return result
#
#         elif category2 == 1:
#             result = 8 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
#                 np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
#
#             return result
#
#         elif category2 == 2:
#             result = 17 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
#                 np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
#
#             return result
#
#     elif category1 == 2:
#
#         if category2 == 0:
#             result = 23 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
#                 np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
#
#             return result
#
#         elif category2 == 1:
#             result = 15 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
#                 np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
#
#             return result
#
#         elif category2 == 2:
#             result = 20 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
#                 np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
#
#             return result
#
#     elif category1 == 3:
#
#         if category2 == 0:
#             result = 50 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
#                 np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
#
#             return result
#
#         elif category2 == 1:
#             result = 60 + (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
#                 np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
#
#             return result
#
#         elif category2 == 2:
#             result = (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
#                 np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
#
#             return result
#
    
    
    
