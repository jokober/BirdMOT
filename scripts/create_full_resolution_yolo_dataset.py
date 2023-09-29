from sahi.utils.coco import export_coco_as_yolov5, Coco

"""
The purpose of this script is to create a yolov5 dataset for full resolution training without any slicing
"""
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--train_coco", type=str, required=True)
    parser.add_argument("--val_coco", type=str, required=False)
    args = parser.parse_args()

    train_coco = Coco.from_coco_dict_or_path(args.train_coco, image_dir=args.image_dir)
    val_coco = Coco.from_coco_dict_or_path(args.val_coco, image_dir=args.image_dir)

    data_yml_path = export_coco_as_yolov5(
        output_dir=args.output_dir,
        train_coco=train_coco,
        numpy_seed=0,
        disable_symlink=True,
    )
