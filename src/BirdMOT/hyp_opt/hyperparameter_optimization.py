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

params = {
    # Parameters for Shi-Tomasi corner detection
    "feature_params": {
        "max_corners": 150,
        "quality_level": 0.03,
        "min_distance": 5,
        "block_size": 50
    },
    # Parameters for Lucas-Kanade optical flow
    "lk_params": {
        "winSize": [50, 50],
        "maxLevel": 6,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
}

import lightgbm as lgb
from sklearn.model_selection import train_test_split

NUM_BOOST_ROUND = 300
EARLY_STOPPING_ROUNDS = 30


def train_evaluate(X, y, params):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=0.2,
                                                          random_state=1234)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model = lgb.train(params, train_data,
                      num_boost_round=NUM_BOOST_ROUND,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      valid_sets=[valid_data],
                      valid_names=['valid'])

    score = model.best_score['valid']['auc']
    return score


@skopt.utils.use_named_args(SPACE)
def objective(**params):
    all_params = {**params, **STATIC_PARAMS}
    return -1.0 * train_evaluate(X, y, all_params)


results = skopt.forest_minimize(objective, SPACE,
                                callback=[monitor], **HPO_PARAMS)

skopt.dump(results, 'artifacts/results.pkl')
