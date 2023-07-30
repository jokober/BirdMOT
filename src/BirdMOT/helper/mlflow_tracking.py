import json
from pathlib import Path

import mlflow


from collections.abc import MutableMapping

def flatten(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def log_evaluation(experiment, params: dict, results: dict, results_path:Path):


    mlflow.set_experiment(experiment)

    flattened_params = flatten(params)
    flattened_results = flatten(results)
    #flattened_results.pop('results_per_category_bbox_bird_mAP50_m')
    try:
        flattened_results.pop('bbox_mAP_copypaste')
    except:
        print("Warning: key 'bbox_mAP_copypaste' not found in results.")
    flattened_results = {key: float(value) for key, value in flattened_results.items()}

    print(f'params {json.dumps(flattened_params)}')
    print(f'results {json.dumps(flattened_results)}')

    with mlflow.start_run():
        mlflow.log_params(flattened_params)
        mlflow.log_metrics(flattened_results)
        mlflow.log_artifacts(results_path.as_posix())