from BirdMOT.data.dataset_stats import calculate_dataset_stats, get_basic_dataset_tracking_stats
from fixtures.fixtures import primary_coco_annotations_fixture_path


def test_dataset_stats_runnable(tmp_path):
    stats_dict = calculate_dataset_stats(coco_path=primary_coco_annotations_fixture_path, save_path=tmp_path)
    print(stats_dict)

def test_get_basic_dataset_tracking_stats():
    dataset_tracking_stats = get_basic_dataset_tracking_stats(primary_coco_annotations_fixture_path)
    assert 'num_instances'in dataset_tracking_stats, "num_instances not in stats_dict"
    assert dataset_tracking_stats['num_instances'] == 1, "num_instances should be 1"
    print(dataset_tracking_stats)