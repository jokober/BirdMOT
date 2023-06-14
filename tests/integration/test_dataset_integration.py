from sahi.utils.coco import Coco

from BirdMOT.data.dataset_stats import get_basic_dataset_detection_stats
from BirdMOT.data.dataset_tools import merge_coco_recursively_from_path
from fixtures.fixtures import secondary_coco_annotations_fixture_path, primary_coco_annotations_fixture_path, \
    coco_images_fixture_path, coco_annotations_fixture_dir


def test_merge_dataset_and_stats(tmp_path):
    coco_1 = Coco.from_coco_dict_or_path(primary_coco_annotations_fixture_path.as_posix(), image_dir=coco_images_fixture_path.as_posix())
    coco_2 = Coco.from_coco_dict_or_path(secondary_coco_annotations_fixture_path.as_posix(), image_dir=coco_images_fixture_path.as_posix())

    assert len(list(coco_annotations_fixture_dir.glob('**/*.json'))) > 1, f"There should be more than one coco file in the fixture directory. Found {list(coco_annotations_fixture_dir.glob('**/*.json'))}"

    merged_coco = merge_coco_recursively_from_path(coco_annotations_fixture_dir, coco_images_fixture_path)

    assert get_basic_dataset_detection_stats(merged_coco)['num_images'] == get_basic_dataset_detection_stats(coco_1)['num_images'] + get_basic_dataset_detection_stats(
        coco_2)['num_images']

