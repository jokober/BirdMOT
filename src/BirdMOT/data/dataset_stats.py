import json
import math
from argparse import ArgumentParser
from pathlib import Path

from prettytable import PrettyTable
import numpy as np
import plotly.express as px
from plotly.offline import iplot
from sahi.utils.coco import Coco

from BirdMOT.helper.folder_structure import FolderStructure


def get_basic_dataset_tracking_stats(coco_path: Path, ) -> dict:
    """
    Get basic dataset stats for tracking datasets.

    Args:
        coco_path: Path to the coco annotation file.

    Returns:
        A dict containing the stats.

    """
    with open(coco_path) as json_file:
        data = json.load(json_file)
        assert "videos" in data

        instance_id_set =  set(annotation['instance_id'] for annotation in data['annotations'])
        num_instances = len(instance_id_set)

    return {
        'num_instances': num_instances,
    }

def get_basic_dataset_detection_stats(coco: Coco) -> dict:
    """
    Get dataset stats, save them to a file and return them as a dict.

    Args:
        coco: Path to the coco annotation file.

    Returns:
        A dict containing the stats.

    """

    return coco.stats

def save_stats_table(stats_dict, save_path):
    x = PrettyTable(border=True, header=False, padding_width=1)
    x.field_names = ["Key", "Value"]
    x.align["Key"] = "r"
    x.align["Value"] = "l"
    x.add_row(["Number of Images", stats_dict['num_images']])
    x.add_row(["Number of Annotations", stats_dict['num_annotations']])
    x.add_row(["Minimum Number of Annotations in Image", stats_dict['min_num_annotations_in_image']])
    x.add_row(["Maximum Number of Annotations in Image", stats_dict['max_num_annotations_in_image']])
    x.add_row(["Average Number of Annotations in Image", stats_dict['avg_num_annotations_in_image']])
    x.add_row(["Minimum Annotation Area", f"{stats_dict['min_annotation_area']} (approx. equivalent to {int(math.sqrt(stats_dict['min_annotation_area']))}\N{SUPERSCRIPT TWO})"])
    x.add_row(["Maximum Annotation Area", f"{stats_dict['max_annotation_area']} (approx. equivalent to {int(math.sqrt(stats_dict['max_annotation_area']))}\N{SUPERSCRIPT TWO})"])
    x.add_row(["Average Annotation Area", f"{stats_dict['avg_annotation_area']} (approx. equivalent to {int(math.sqrt(stats_dict['avg_annotation_area']))}\N{SUPERSCRIPT TWO})"])

    if 'num_instances' in stats_dict:
        x.add_row(["Num Tracking instances", f"{stats_dict['num_instances']}"])

    with open(save_path / 'basic_dataset_stats.txt', 'w') as w:
        w.write(x.get_string())

    with open(save_path / 'basic_dataset_stats.tex', 'w') as w:
        w.write(x.get_latex_string())

def create_bbox_size_histogram(coco: Coco, save_path: Path, end = 1000):
    bbox_areas = []

    for image in coco.images:
        #for annotation in filter(lambda annotation: annotation.area < 10000, image.annotations): # ToDo: Careful with this filter. Annotations might not go into stats
        for annotation in image.annotations:
            bbox_areas.append(math.sqrt(annotation.area))

    bbox_areas = np.array(bbox_areas)

    start = int(bbox_areas.min())
    size = 1

    # Making a histogram
    largest_value = bbox_areas.max()
    if largest_value > end:
        hist = np.histogram(bbox_areas, bins=list(range(start, end+size, size)) + [largest_value])
    else:
        hist = np.histogram(bbox_areas, bins=list(range(start, end+size, size)) + [end+size])

    # Adding labels to the chart
    labels = []
    for i, j in zip(hist[1][0::1], hist[1][1::1]):
        if j <= end:
            labels.append('{} - {}'.format(i, j))
        else:
            labels.append('> {}'.format(i))

    #fig = px.histogram(bbox_areas,x=labels,y=hist[0], nbins=20)
    fig = px.bar(x=labels,y=hist[0])
    fig.write_image((save_path / "ann_bbox_size_hist.png").as_posix())
    #fig.show() # ToDo: Save somewhere?


def calculate_dataset_stats(coco_path: Path, save_path: Path):
    print(save_path.as_posix())
    coco = Coco.from_coco_dict_or_path(coco_path.as_posix())
    create_bbox_size_histogram(coco, save_path)
    stats_dict = get_basic_dataset_detection_stats(coco)
    #stats_dict.update(get_basic_dataset_tracking_stats(coco_path))

    save_stats_table(stats_dict, save_path)

    return stats_dict

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--coco_path", type=str, required=True)
    args = parser.parse_args()

    calculate_dataset_stats(Path(args.coco_path), FolderStructure().dataset_stats_dir)

