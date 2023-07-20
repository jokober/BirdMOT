import json
from pathlib import Path

from BirdMOT.data.dataset_tools import find_correct_image_path, rapair_absolute_image_paths
from fixtures.fixtures import primary_val_images_path_fixture, coco_images_fixture_path, \
    primary_coco_images_fixture_path, train_coco_fixture_path


def test_find_correct_image_path():
    wrong_image_path = Path("/some/wrong/path") / 'images' / 'good_04_2021' / 'C0085_125820_125828'
    new_image_path = find_correct_image_path(coco_images_fixture_path, wrong_image_path)

    assert  new_image_path.resolve() == primary_coco_images_fixture_path.resolve(), f"Wrong image path found. Path found: {new_image_path}. Correct path: {primary_coco_images_fixture_path}"

def test_rapair_absolute_image_paths():
    coco_dict = rapair_absolute_image_paths(train_coco_fixture_path, coco_images_fixture_path, overwrite_file=False)

    for image_element in coco_dict['images']:
        print(image_element)
        assert Path(image_element['file_name']).exists(), f"""Image path {image_element['file_name']} does not exist.
                                                          Hence rapair_absolute_image_paths has failed."""