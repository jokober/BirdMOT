import json
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
from sahi.utils.coco import Coco


def verify_jpeg_image(file_path):
    try:
        img = Image.open(file_path)
        print(img.format)
        print(file_path.suffix)
        # img.getdata()[0]
        img.verify()
    except OSError:
        return False
    if file_path.suffix == '.jpg':
        assert img.format == 'JPEG'
    elif file_path.suffix == '.png':
        assert img.format == 'PNG'
    return True


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--assembly_config", type=Path, required=True)
    parser.add_argument("--image_path", type=Path, required=True)
    args = parser.parse_args()

    assert args.assembly_config.exists(), f"Assembly config {args.assembly_config} does not exist"
    with open(args.assembly_config) as json_file:
        assembly_config = json.load(json_file)

    corrupt_images = {}
    for coco_file in assembly_config['dataset_config']:
        corrupt_images[coco_file['name']] = []
        coco = (Coco.from_coco_dict_or_path((args.assembly_config.parents[2] / "dataset/coco_files" / coco_file[
            'coco_annotation_file_path']).as_posix(), image_dir=args.image_path.as_posix()))
        for image in coco.images:
            img_path = args.image_path / image.file_name
            img_path.exists(), f"Image {img_path} does not exist"
            if not verify_jpeg_image(img_path):
                print(img_path)
                corrupt_images[coco_file['name']].append(image.file_name)

    with open('corrupt_data.json', 'w') as f:
        json.dump(corrupt_images, f)
