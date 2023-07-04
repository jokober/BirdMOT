"""
This module slices image and coco annotation data.
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from BirdMOT.data.SliceParams import SliceParams

os.getcwd()

from sahi.slicing import slice_coco
from sahi.utils.file import load_json

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt



def slice_dataset(coco_annotation_file_path: Path, image_dir: Path, output_dir: Path, output_coco_dir, slice_params: SliceParams):
    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=coco_annotation_file_path.as_posix(),
        image_dir=image_dir.as_posix(),
        output_coco_annotation_file_name=output_coco_dir.name, # The strip is required as sahi adds '_coco.json' to the output file name
        ignore_negative_samples=slice_params.ignore_negative_samples,
        output_dir=(output_dir / "images").as_posix(),
        slice_height=slice_params.height,
        slice_width=slice_params.width,
        overlap_height_ratio=slice_params.overlap_height_ratio,
        overlap_width_ratio=slice_params.overlap_width_ratio,
        min_area_ratio=slice_params.min_area_ratio,
        verbose=slice_params.verbose,
    )
    shutil.move(coco_path, output_coco_dir)

    return coco_dict, output_coco_dir
