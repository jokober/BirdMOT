from pathlib import Path

import fiftyone.utils.coco as fouc
import fiftyone.utils.coco as fouc
import numpy as np
import fiftyone as fo


def fiftyone_evaluation(image_path: Path, labels_path: Path, predictions_path: Path, iou=0.1, iou_thresholds=None):
    # Load COCO formatted dataset
    coco_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=image_path.as_posix(),
        labels_path=labels_path.as_posix(),
        include_id=True
    )
    classes = ["bird"]
    fouc.add_coco_labels(coco_dataset, "ground_truth", labels_path.as_posix(), classes)

    # load predictions
    fouc.add_coco_labels(coco_dataset, "predictions",
                         predictions_path.as_posix(),
                         classes)

    if iou_thresholds is None:
        iou_thresholds = np.arange(0.05, 0.95, 0.05).tolist()

    for iou in iou_thresholds:
        # make evaluations
        result = coco_dataset.evaluate_detections(
            "predictions",
            gt_field="detections",
            method="coco",
            eval_key=f"eval{str(iou).replace('.', '')}",
            classwise=False,
            iou=iou
            # use_boxes=True
        )
    #asd = coco_dataset.field_names
    field_schema= coco_dataset.get_field_schema()

    print(coco_dataset)
    tp_iou_results = [coco_dataset.sum(f"eval{str(iou).replace('.', '')}_tp") for iou in iou_thresholds]
    fn_iou_results = [coco_dataset.sum(f"eval{str(iou).replace('.', '')}_fn") for iou in iou_thresholds]
    fp_iou_results = [coco_dataset.sum(f"eval{str(iou).replace('.', '')}_fp") for iou in iou_thresholds]
    # tn_iou_results = [coco_dataset.sum(f"eval{str(iou).replace('.', '')}_tn") for iou in iou_thresholds]
    recall = [tp / (tp + fn) for tp, fn in zip(tp_iou_results, fn_iou_results)]
    precision = [tp / (tp + fp) for tp, fp in zip(tp_iou_results, fp_iou_results)]
    # fpr =  [fp / (fp + tn) for fp, tn in zip(fp_iou_results, tn_iou_results)]
    fnr = [fn / (fn + tp) for tp, fn, fp in zip(tp_iou_results, fn_iou_results, fp_iou_results)]
    tpr = [tp / (tp + fn) for tp, fn, fp in zip(tp_iou_results, fn_iou_results, fp_iou_results)]

    results = coco_dataset.evaluate_detections(
        "predictions",
        gt_field="detections",
        method="coco",
        eval_key=f"eval",
        classwise=False,
        compute_mAP=True,
        iou=iou,
        # use_boxes=True
    )

    fiftyone_eval_res = {}
    fiftyone_eval_res["mAP"] = results.mAP()
    fiftyone_eval_res["iou_thresholds"] = iou_thresholds
    fiftyone_eval_res["tp_iou_results"] = tp_iou_results
    fiftyone_eval_res["fn_iou_results"] = fn_iou_results
    fiftyone_eval_res["fp_iou_results"] = fp_iou_results
    fiftyone_eval_res["recall"] = recall
    fiftyone_eval_res["precision"] = precision
    # fiftyone_eval_res["fpr"] = fpr
    fiftyone_eval_res["fnr"] = fnr
    fiftyone_eval_res["tpr"] = tpr

    return fiftyone_eval_res
