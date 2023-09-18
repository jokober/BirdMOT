import yaml
from pathlib import Path

filtered_yolo_params = ['epochs', 'batch', 'imgsz', 'dropout', 'iou', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate',
                    'scale', 'shear', 'perspective', 'mosaic', 'mixup']
def load_yolo_args_yaml(path: Path) -> dict:
    args = yaml.safe_load(path.read_text())
    return args


def load_filtered_yolo_args_yaml(path: Path) -> dict:
    args = load_yolo_args_yaml(path)
    return {f"yolo_conf_{key}": value for key, value in args.items() if
            key in filtered_yolo_params}
