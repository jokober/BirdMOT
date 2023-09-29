import lightgbm as lgb

import skopt.plots

skopt.plots.plot_convergence(results)

SPACE = [
   skopt.space.Real(0.01, 0.5, name='quality_level', prior='log-uniform'),
   skopt.space.Integer(1, 30, name='max_corners'),
   skopt.space.Integer(1, 20, name='min_distance'),
   skopt.space.Integer(1, 20, name='block_size'),

   skopt.space.Integer(1, 60, name='square_win_size'),
   skopt.space.Integer(1, 10, name='maxLevel'),

   skopt.space.Real(0.01, 0.5, name='criteria_epsilon', prior='log-uniform'),
   skopt.space.Integer(1, 20, name='criteria_epsilon', prior='log-uniform'),
   ]

# Bounded region of parameter space
pbounds = {'quality_level': (0.01, 0.5),
           'max_corners': (1, 30),
           'min_distance': (),
           '':(),
           }

params = {
    # Parameters for Shi-Tomasi corner detection
    "feature_params": {
        "max_corners": 150,
        "quality_level": 0.03,
        "min_distance": 5,
        "block_size": 50
    },

def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1

from bayes_opt import BayesianOptimization



optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=3,
)

print(optimizer.max)