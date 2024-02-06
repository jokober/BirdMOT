from BirdMOT.helper.yolov_args_reader import load_filtered_yolo_args_yaml, filtered_yolo_params
from fixtures.fixtures import yolov8_test_model

def test_load_filtered_yolo_args_yaml():
        filtered_yolo_args = load_filtered_yolo_args_yaml(yolov8_test_model / 'args.yaml')

        assert len(filtered_yolo_args) == len(filtered_yolo_params)
        for param in filtered_yolo_params:
            assert f"yolo_conf_{param}" in filtered_yolo_args