from pathlib import Path

from PIL.Image import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, predict, get_sliced_prediction

yolov8_model_path = "/mnt/fids_data/BirdMOT/local_data/models/dataset_assembly2_rc_4good_tracks_in_val/YOLOv8n+FI+PO/weights/best_delete_suffix.pt"

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.000001,
    device="cpu", # or 'cuda:0'
)

images = [
    "/mnt/fids_data/BirdMOT/local_data/dataset/images/low_bitrate_09_2022/MOT-1654911941_zoex_0446964_0447233/img1/000006.jpg",
    "/mnt/fids_data/BirdMOT/local_data/dataset/images/low_bitrate_09_2022/MOT-1656471341_zoex_0149757_0149861/img1/000031.jpg",
    "/home/jo/coding_projects/fids/BirdMOT/src/000031_png.png",
    "/mnt/fids_data/BirdMOT/local_data/dataset/images/rainy_09_2021/1631281579043_0027025_0027126-0000090.png",
    "/mnt/fids_data/BirdMOT/local_data/dataset/images/rainy_09_2021/1631281579043_0027025_0027126-0000024.png",
    "/mnt/fids_data/BirdMOT/local_data/dataset/images/good_04_2021/C0054_783015_783046/C0054_783015_783046-783018.png",
    ]

for img in images:
    result = get_sliced_prediction(
        img,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.5,
        overlap_width_ratio=0.5
    )

    result.export_visuals(export_dir="demo_data/", file_name=Path(img).stem)
    print(result)


    object_prediction_list = result.object_prediction_list

#    object_prediction_list[0]

model_type = "yolov8"
model_path = yolov8_model_path
model_device = "cpu" # or 'cuda:0'
model_confidence_threshold = 0.4

slice_height = 256
slice_width = 256
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "demo_data/"


