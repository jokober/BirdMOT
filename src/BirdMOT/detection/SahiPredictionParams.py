from dataclasses import dataclass


@dataclass
class SahiPredictionParams:
    """

    Attributes:
        source: str
            Folder directory that contains images or path of the image to be predicted. Also video to be predicted.
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'NMM', 'GRREDYNMM' or 'NMS'. Default is 'GRREDYNMM'
        postprocess_match_metric str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        export_pickle: bool
            Export predictions as .pickle
        dataset_json_path: str
            If coco file path is provided, detection results will be exported in coco json format.
        project: str
            Save results to project/name.

    """
    model_path: str # path to model weight file
    model_type: str # one of yolov5', 'mmdet', 'detectron2'
    model_confidence_threshold: float = 0.2
    source: str = None
    model_device: str = 'cpu' # 'cpu' or 'cuda:0'
    no_standard_prediction: bool = True
    no_sliced_prediction: bool = False
    slice_height: int = 640
    slice_width: int = 640
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2
    export_crop: bool = False
    #model_config_path = None # for detectron2 and mmdet models
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    export_pickle: bool = True,
    dataset_json_path: bool = True,
    project: str = "runs/predict",
    name: str = None,
    return_dict: bool = True,

