"""
Explore coco datasets with fiftyone recusively from a path.
"""
from argparse import ArgumentParser
from pathlib import Path

from sahi.utils.fiftyone import launch_fiftyone_app

from BirdMOT.data.dataset_tools import merge_coco_recursively_from_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    coco_paths = list(args.input_path.glob('**/*.json'))

    print("which dataset do you want to explore? (Enter the number of the dataset you want to explore)")
    for i, coco_path in enumerate(coco_paths):
        print(f"{i}: {coco_path}")
    input = input()

    session = launch_fiftyone_app(args.image_path, coco_paths[int(input)])