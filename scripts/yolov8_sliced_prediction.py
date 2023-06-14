from argparse import ArgumentParser
from pathlib import Path

from BirdMOT.data.SliceParams import SliceParams
from BirdMOT.detection.yolov8 import sliced_yolov8_train, Yolov8TrainParams

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--coco_annotation_file_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()

    yolov8_train_params = Yolov8TrainParams(
        epochs=20,
        batch_size=20,
        model = 'YOLOv8n',
    )

    slice_params = SliceParams(
        height=640,
        width=640,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
        min_area_ratio=0.2,
        ignore_negative_samples=True
    )

    sliced_batch_predict(yolov8_train_params, slice_params, Path(a