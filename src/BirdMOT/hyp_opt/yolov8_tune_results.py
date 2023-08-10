import pickle
from argparse import ArgumentParser

from ray import tune
from ultralytics import YOLO

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_path", type=str, required=True)

    args = parser.parse_args()

    with open(f"{args.experiment_path}/tuner.pkl", "rb") as input_file:
            e = pickle.load(input_file)

    print(e)
    print(type(e))
    dir(e)
    print(dir(e))
    print(f"Loading results from {args.experiment_path}...")

    from inspect import getmembers, isfunction

    print(getmembers(e, isfunction))



    print("without pkl")

    restored_tuner = tune.Tuner.restore(args.experiment_path, trainable="_tune")
    result_grid = restored_tuner.get_results()

    for i, result in enumerate(result_grid):
        print(f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}")

    import matplotlib.pyplot as plt

    for result in result_grid:
        plt.plot(result.metrics_dataframe["training_iteration"], result.metrics_dataframe["mean_accuracy"], label=f"Trial {i}")

    plt.xlabel('Training Iterations')
    plt.ylabel('Mean Accuracy')
    plt.legend()
    plt.show()