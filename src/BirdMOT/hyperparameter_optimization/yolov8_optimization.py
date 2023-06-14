from ray import tune

slice_param_space = {
    'size': tune.qrandint(300, 1000, 50),
    'overlap_ratio': tune.quniform(0, 0.8, 0.05),
    'min_area_ratio': tune.quniform(0, 0.8, 0.05),
}

yolov8_param_space = {
    'epochs': tune.qrandint(1, 100, 1),
    'batch_size': tune.qrandint(1, 100, 1),
    'model': tune.choice(["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"]),
}