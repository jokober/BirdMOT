"""
The purpose of this script is to merge multiple COCO-formatted annotation files into one.

command:
python -m scripts.merge_coco_files --input_path ... --image_path ... --output_path ...
"""
from argparse import ArgumentParser
from pathlib import Path

from BirdMOT.data.dataset_tools import merge_coco_recursively_from_path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--categories", type=str, required=True)
    args = parser.parse_args()

    merge_coco_recursively_from_path(input_path=Path(args.input_path),
                                     image_path=args.image_path,
                                     output_path=Path(
                                         args.output_path),
                                     categories=Path(args.categories)
                                     )
