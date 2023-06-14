from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

coco_ground_truth = COCO(annotation_file="coco_dataset.json")
coco_predictions = coco_ground_truth.loadRes("coco_predictions.json")

coco_evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()