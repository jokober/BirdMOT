import json
import os
import shutil
from pathlib import Path
from typing import List

from sahi.utils.coco import Coco, CocoVid, CocoVideo, CocoVidImage, CocoVidAnnotation
from sahi.utils.file import save_json
from sahi.utils.coco import Coco, export_coco_as_yolov5


from BirdMOT.data.slice_data import SliceParams, slice_dataset
from BirdMOT.helper.folder_structure import folder_structure_obj


def create_sliced_dataset(coco_annotation_file_path: Path, image_dir: Path, slice_params: SliceParams):

    dataset_path = folder_structure_obj.get_or_create_sliced_dataset_folder_path(slice_params)
    if not any((dataset_path / 'images').iterdir()):
        slice_dataset(coco_annotation_file_path, image_dir, dataset_path, slice_params)
        val_train_split(coco_path=dataset_path / "coco_files/sliced_coco.json", output_path=dataset_path / "coco_files")
    else:
        print("Sliced dataset already exists. If you want to create a new one, delete the old one first.")
    print(dataset_path.as_posix())
    return dataset_path

def val_train_split(coco_path: Path, output_path: Path, train_split_rate: float = 0.85):
    # init Coco object
    coco = Coco.from_coco_dict_or_path(coco_path.as_posix())

    # split COCO dataset
    result = coco.split_coco_as_train_val(
      train_split_rate=train_split_rate
    )

    # export train val split files
    save_json(result["train_coco"].json, (output_path / "train_split_coco.json").as_posix())
    save_json(result["val_coco"].json, (output_path / "val_split_coco.json").as_posix())

    assert (output_path / "").exists()
    assert (output_path / "").exists()


def merge_two_cocovid_files(coco_1_path: Path, coco_2_path: Path, output_path: Path = None) -> Coco:
    with open(coco_1_path) as json_file:
        coco1_dict = json.load(json_file)
    with open(coco_2_path) as json_file:
        coco2_dict = json.load(json_file)

    merged_coco_dict = {}

    # merge categories
    #if coco2_dict[categories]

    # merge videos

    # merge images

    # merge annotations

def merge_cocovid_recurively_from_path(original_coco_vid: CocoVid, input_path: Path, image_path: Path, output_path: Path = None) -> Coco:
        assert input_path.exists(), f"input_path {input_path} does not exist"
        assert image_path.exists(), f"image_path {image_path} does not exist"
        coco_paths = list(input_path.glob('**/*.json'))
        assert len(
            coco_paths) > 1, f"There should be more than one coco file in the fixture directory. Found {list(input_path.glob('**/*.json'))}"

        merge_cocovid_recurively_from_path(original_coco_vid=original_coco_vid, cocovid_json_path_list=coco_paths, categories_path=None, image_path=image_path, output_path=output_path)


def merge_cocovid_recurively_from_path(original_coco_vid: CocoVid, cocovid_json_path_list: List[Path], categories_path: Path, image_path: Path, output_path: Path = None) -> CocoVid:
    if original_coco_vid == None:
        original_coco_vid = CocoVid(name='BirdMOT', remapping_dict=None)

    if categories_path is not None:
        assert categories_path.exists(), f"categories_path {categories_path} does not exist"
        with open(categories_path) as json_file:
            categories = json.load(json_file)['categories']
            assert len(categories) != 0, f"categories file is empty"
        original_coco_vid.add_categories_from_coco_category_list(categories)


    for cocovid_path in cocovid_json_path_list:
        print(cocovid_path)
        with open(cocovid_path) as json_file:
            cocovid_dict = json.load(json_file)
        # merge categories
        if len(original_coco_vid.json_categories) == 0:
            original_coco_vid.add_categories_from_coco_category_list(cocovid_dict['categories'])
        elif len(original_coco_vid.json_categories) > 0:
            # Compare Category lists
            merging_cats = list(sorted(cocovid_dict['categories'], key=lambda d: d['id']))
            for i, cat in enumerate(sorted(original_coco_vid.json_categories, key=lambda d: d['id'])):
                if cat != merging_cats[i]:
                    raise Exception(f"Category lists are not the same. {cat} != {merging_cats[i]}. Merging different\
                     categories is not implemented yet. Its not your fault!")

        # Merge Videos
        for video in cocovid_dict['videos']:
            # Get height and width
            if 'height' not in video:
                height = set([img["height"] for img in cocovid_dict["images"] is img['video_id'] == video['id']])
                width = set([img["width"] for img in cocovid_dict["images"] is img['video_id'] == video['id']])
                assert len(height) == 1, 'multiple different heights for one video'
                assert len(width) == 1, 'multiple different width for one video'
            else:
                height = video['height']
                width = video['width']

            # Get fps
            if 'fps' not in video:
                fps = 29.97
            else:
                fps = video['fps']

            # Get video id
            video = CocoVideo(name=video['name'], video_id=1, frame_rate=fps, width=width, height=height)





    image = CocoVidImage(file_name='image_1.jpg', height=1, width =1, video_id=1, frame_id=1,)
    annotation = CocoVidAnnotation(bbox=[1, 1, 1, 1], category_id=1, category_name = '', image_id=1, instance_id=1)
    image.add_annotation()
    video.add_cocovidimage()
    original_coco_vid.add_video(video)


def merge_coco_recursively_from_path(input_path: Path, image_path: Path, output_path: Path = None, categories: Path = None) -> Coco:
    """

    Args:
        input_path:
        image_path:
        output_path:
        categories: if path to coco formatted category json file is given. The

    Returns:

    """
    # ToDo: this does not keep instance_ids and videos. Implement this yourself!
    assert  input_path.exists(), f"input_path {input_path} does not exist"
    assert  image_path.exists(), f"image_path {image_path} does not exist"
    coco_paths = list(input_path.glob('**/*.json'))
    assert len(coco_paths) > 1, f"There should be more than one coco file in the fixture directory. Found {list(input_path.glob('**/*.json'))}"


    with open(categories) as json_file:
        categories = json.load(json_file)

    [print(cat) for cat in categories['categories']]
    remapping_dict = {i:i+1 for i, _ in enumerate(categories['categories'])}
    print(remapping_dict)
    name2id_dict = {cat['name']: cat['id'] for cat in categories['categories']}
    coco_1 = Coco(image_dir=image_path.as_posix())
    coco_1.add_categories_from_coco_category_list(categories['categories'])


    #coco_1 = Coco.from_coco_dict_or_path(coco_paths[0].as_posix(), image_dir=image_path.as_posix())
    for coco_path in coco_paths:
        print(coco_path.as_posix())
        coco_2 = Coco.from_coco_dict_or_path(coco_path.as_posix(), image_dir=image_path.as_posix())
        coco_1.merge(coco_2, desired_name2id=name2id_dict)

    coco_1.add_categories_from_coco_category_list(categories['categories'])

    if output_path is not None:
        save_json(coco_1.json, output_path)

    return coco_1



def coco2yolov5(dataset_path: Path, coco_images_dir: Path):
    # init Coco object
    train_coco = Coco.from_coco_dict_or_path((dataset_path / 'coco_files' / "train_split_coco.json").as_posix(), image_dir=coco_images_dir.as_posix())
    val_coco = Coco.from_coco_dict_or_path((dataset_path / 'coco_files' / "val_split_coco.json").as_posix(), image_dir=coco_images_dir.as_posix())

    # export converted YoloV5 formatted dataset into given output_dir with given train/val split
    data_yml_path = export_coco_as_yolov5(
        output_dir=(dataset_path / 'yolov5_files').as_posix(),
        train_coco=train_coco,
        val_coco=val_coco
    )

    shutil.copy((dataset_path / 'yolov5_files' / "data.yml").as_posix(),( dataset_path / 'yolov5_files' / "data.yaml").as_posix())


