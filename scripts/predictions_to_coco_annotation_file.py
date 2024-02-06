import json
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.coco import Coco, CocoImage, CocoAnnotation
from sahi.utils.file import save_json


def create_coco_annotation_file_from_path(image_path: Path, output_path: Path, coco_categories_path: Path,
                                          model_path: str, image_path_prefix: str = ""):
    # Init Sahi Coco object
    coco = Coco(image_dir=image_path.parent.as_posix())
    # Add categories
    with open(coco_categories_path) as json_file:
        categories = json.load(json_file)

    coco.add_categories_from_coco_category_list(categories['categories'])

    # Loop through all the image files in the input directory
    for filename in (p.resolve() for p in Path(image_path).glob("**/*") if p.suffix in {".png", ".jpg"}):
        im = Image.open(image_path / filename)
        width, height = im.size
        print(filename.as_posix())

        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=0.5,
            device="cuda:0",  # or 'cuda:0'
        )

        sahi_result = get_sliced_prediction(
            filename.as_posix(),
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        prediction_list = sahi_result.to_coco_annotations()

        # Add Coco image including the predicted annotations
        coco_image = CocoImage(file_name=image_path_prefix + filename.name, height=height, width=width)
        for pred in prediction_list:
            coco_image.add_annotation(
                CocoAnnotation(
                    bbox=pred['bbox'],
                    category_id=1,
                    category_name="bird",
                )
            )
        coco.add_image(coco_image)

    # Convert coco object to coco json:
    coco_json = coco.json
    save_json(coco_json, output_path.as_posix())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--coco_categories_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path_prefix", type=str, required=True)
    args = parser.parse_args()

    create_coco_annotation_file_from_path(Path(args.image_path), Path(args.output_path),
                                          Path(args.coco_categories_path), model_path=args.model_path,
                                          image_path_prefix=args.image_path_prefix)
