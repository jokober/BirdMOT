from argparse import ArgumentParser

from ray import tune
from ultralytics import YOLO

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--gpu_per_trial", type=int, required=True)
    args = parser.parse_args()


    search_space = {"lr0": tune.uniform(1e-5, 1e-1),
                    "lrf": tune.uniform(0.01, 1.0),
                    "momentum": tune.uniform(0.6, 0.98),
                    "weight_decay": tune.uniform(0.0, 0.001),
                    "warmup_epochs": tune.uniform(0.0, 5.0),
                    "warmup_momentum": tune.uniform(0.0, 0.95),
                    "box": tune.uniform(0.02, 0.2),
                    "hsv_h": tune.uniform(0.0, 0.1),
                    "hsv_s": tune.uniform(0.0, 0.9),
                    "hsv_v": tune.uniform(0.0, 0.9),
                   # "degrees": tune.uniform(0.0, 45.0),
                    "translate": tune.uniform(0.0, 0.9),
                    "scale": tune.uniform(0.0, 0.9),
                    "shear": tune.uniform(0.0, 10.0),
                    "perspective": tune.uniform(0.0, 0.001),
                    #"flipud": tune.uniform(0.0, 1.0),  # Vertical flip augmentation probability
                    "fliplr": tune.uniform(0.0, 1.0),  # Horizontal flip augmentation probability
                    "mosaic": tune.uniform(0.0, 1.0),  # Mosaic augmentation probability
                    "mixup": tune.uniform(0.0, 1.0),  # Mixup augmentation probability
                    "copy_paste": tune.uniform(0.0, 1.0),  # Copy-Paste augmentation probability
                    }

    model = YOLO("yolov8n.pt")
    result_grid = model.tune(data=args.data, space=search_space, gpu_per_trial=args.gpu_per_trial, max_samples=60, epochs=40, project=args.out,
                             name="birdmot_yolov8n_tune", verbose=True)

