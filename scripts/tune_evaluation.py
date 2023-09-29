import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.lightgbm import TuneReportCheckpointCallback

from BirdMOT.detection.evaluate import EvaluationController


def optimize(one_experiment_config, assembly_configs, device):
    eval_controller = EvaluationController()
    results = eval_controller.find_or_create_evaluation(one_experiment_config, assembly_configs, device=device,
                                                        train_missing=True)


    session.report(
        {
            "mean_accuracy": sklearn.metrics.accuracy_score(test_y, pred_labels),
            "done": True,
        }
    )

    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
    train_set = lgb.Dataset(train_x, label=train_y)
    test_set = lgb.Dataset(test_x, label=test_y)
    gbm = lgb.train(
        config,
        train_set,
        valid_sets=[test_set],
        valid_names=["eval"],
        verbose_eval=False,
        callbacks=[
            TuneReportCheckpointCallback(
                {
                    "binary_error": "eval-binary_error",
                    "binary_logloss": "eval-binary_logloss",
                }
            )
        ],
    )
    preds = gbm.predict(test_x)
    pred_labels = np.rint(preds)
    session.report(
        {
            "mean_accuracy": sklearn.metrics.accuracy_score(test_y, pred_labels),
            "done": True,
        }
    )


if __name__ == "__main__":
    config = {
        "objective": "binary",
        "metric": ["binary_error", "binary_logloss"],
        "verbose": -1,
        "boosting_type": tune.grid_search(["gbdt", "dart"]),
        "num_leaves": tune.randint(10, 1000),
        "learning_rate": tune.loguniform(1e-8, 1e-1),
    }

    tuner = tune.Tuner(
        train_breast_cancer,
        tune_config=tune.TuneConfig(
            metric="binary_error",
            mode="min",
            scheduler=ASHAScheduler(),
            num_samples=2,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)