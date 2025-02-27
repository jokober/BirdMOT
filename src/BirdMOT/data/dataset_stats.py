import json
import math
from argparse import ArgumentParser
from copy import deepcopy
from itertools import count
from pathlib import Path

from matplotlib import pyplot as plt
from prettytable import PrettyTable
import numpy as np
import plotly.express as px
from plotly.offline import iplot
from sahi.utils.coco import Coco
from sahi.utils.file import save_json


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
    x.add_row(["Minimum Object Area", f"{stats_dict['min_annotation_area']} (approx. equivalent to {int(math.sqrt(stats_dict['min_annotation_area']))}\N{SUPERSCRIPT TWO})"])
    x.add_row(["Maximum Object Area", f"{stats_dict['max_annotation_area']} (approx. equivalent to {int(math.sqrt(stats_dict['max_annotation_area']))}\N{SUPERSCRIPT TWO})"])
    x.add_row(["Average Object Area", f"{stats_dict['avg_annotation_area']} (approx. equivalent to {int(math.sqrt(stats_dict['avg_annotation_area']))}\N{SUPERSCRIPT TWO})"])
    x.add_row(["Average Object Width", f"{stats_dict['average_object_width']}"])
    x.add_row(["Number of Negatives", stats_dict['num_negatives']])
    x.add_row(["Small Object Count", stats_dict['small_object_count']])
    x.add_row(["Medium Object Count", stats_dict['medium_object_count']])
    x.add_row(["Large Object Count", stats_dict['large_object_count']])


    if 'num_instances' in stats_dict:
        x.add_row(["Num Tracking instances", f"{stats_dict['num_instances']}"])

    with open(save_path / 'basic_dataset_stats.txt', 'w') as w:
        table_string = x.get_string()
        w.write(table_string)

    with open(save_path / 'basic_dataset_stats.tex', 'w') as w:
        w.write(x.get_latex_string())


def save_dataset_comparison_stats_table(stats_list, save_path):
    x = PrettyTable(border=True, header=True, padding_width=1)
    x.field_names = ["Dataset"] + [it['name'] for it in stats_list]

    x.align = "l"
    x.align["Dataset"] = "r"

    x.add_row(["Number of Images"] + [it['stats']['num_images'] for it in stats_list])
    x.add_row(["Number of Annotations"] + [it['stats']['num_annotations'] for it in stats_list], divider=True)
    x.add_row(["Minimum Number of Annotations in Image"] + [it['stats']['min_num_annotations_in_image'] for it in stats_list])
    x.add_row(["Maximum Number of Annotations in Image"] + [it['stats']['max_num_annotations_in_image'] for it in stats_list])
    x.add_row(["Average Number of Annotations in Image"] + [f"{it['stats']['avg_num_annotations_in_image']:.2f}" for it in stats_list], divider=True)
    x.add_row(["Minimum Annotation Area"] + [f"{int(it['stats']['min_annotation_area'])} ({chr(8773)} {int(math.sqrt(it['stats']['min_annotation_area']))}\N{SUPERSCRIPT TWO})" if it['stats']['min_annotation_area'] != 10000000000 else 0 for it in stats_list])
    x.add_row(["Maximum Annotation Area"] + [f"{int(it['stats']['max_annotation_area'])} ({chr(8773)} {int(math.sqrt(it['stats']['max_annotation_area']))}\N{SUPERSCRIPT TWO})" if it['stats']['max_annotation_area'] != 0 else 0 for it in stats_list])
    x.add_row(["Average Annotation Area"] + [f"{int(it['stats']['avg_annotation_area'])} ({chr(8773)} {int(math.sqrt(it['stats']['avg_annotation_area']))}\N{SUPERSCRIPT TWO})" if it['stats']['avg_annotation_area'] != 0 else 0 for it in stats_list])
    x.add_row(["Number of Negatives"] + [it['stats']['num_negatives'] for it in stats_list])


    with open(save_path / 'basic_dataset_comparison_stats.txt', 'w') as w:
        w.write(x.get_string())

    with open(save_path / 'basic_dataset_comparison_stats.tex', 'w') as w:
        w.write(x.get_latex_string())

def create_bbox_size_histogram(coco: Coco, save_path: Path, end = 265):
    bbox_areas = []

    for image in coco.images:
        #for annotation in filter(lambda annotation: annotation.area < 10000, image.annotations): # ToDo: Careful with this filter. Annotations might not go into stats
        for annotation in image.annotations:
            bbox_areas.append(math.sqrt(annotation.area))

    if  len(bbox_areas) > 0:
        bbox_areas = np.array(bbox_areas)

        print(np.median(bbox_areas))
        print(((0 < bbox_areas) & (bbox_areas < 60)).sum())
        import time
        time.sleep(4)

        start = int(bbox_areas.min())
        size = 10

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
                labels.append('${}²$-${}²$'.format(int(i), int(j)))
            else:
                labels.append('>${}²$'.format(int(i)))

        #fig = px.histogram(bbox_areas,x=labels,y=hist[0], nbins=20)
        fig = px.bar(x=labels,y=hist[0])
        fig.write_image((save_path / "ann_bbox_size_hist.png").as_posix())
        #fig.show() # ToDo: Save somewhere?

        plt.bar(x = labels, height = hist[0])
        plt.xticks(rotation='vertical')
        plt.xlabel("Object area in pixels")
        plt.ylabel("Number of Objects")
        plt.subplots_adjust(bottom=0.23, top=0.93)
        plt.savefig((save_path / "ann_bbox_size_hist_mathplotlib.png").as_posix())

def create_size_histogram(coco: Coco, save_path: Path):
    bbox_areas = []

    for image in coco.images:
        #for annotation in filter(lambda annotation: annotation.area < 10000, image.annotations): # ToDo: Careful with this filter. Annotations might not go into stats
        for annotation in image.annotations:
            bbox_areas.append(math.sqrt(annotation.area))

    if  len(bbox_areas) > 0:
        bbox_areas = np.array(bbox_areas)

        largest_value = bbox_areas.max()
        hist = np.histogram(bbox_areas, bins=[0,32,96, largest_value])

        plt.clf()
        plt.bar(x=('Small', 'Medium', 'Large'), height=hist[0])
        plt.xlabel("Object Size Category")
        plt.ylabel("Number of Objects")
        #plt.subplots_adjust(bottom=0.23, top=0.93)
        plt.savefig((save_path / "size_hist_mathplotlib.png").as_posix())

    return hist[0]

def calculate_dataset_stats(coco_path: Path, save_path: Path):
    print(save_path.as_posix())
    if not save_path.exists():
        print("Dataset Stats save path does not exist. Creating folder")
        save_path.mkdir(parents=True, exist_ok=True)
    assert save_path.is_dir()

    coco = Coco.from_coco_dict_or_path(coco_path.as_posix())
    create_bbox_size_histogram(coco, save_path)
    bird_size_count = create_size_histogram(coco, save_path)
    stats_dict = get_basic_dataset_detection_stats(coco)
    stats_dict['num_negatives'] = len([coco_image for coco_image in coco.images if len(coco_image.annotations) == 0])
    stats_dict['small_object_count'] = bird_size_count[0]
    stats_dict['medium_object_count'] = bird_size_count[1]
    stats_dict['large_object_count'] = bird_size_count[2]
    #stats_dict.update(get_basic_dataset_tracking_stats(coco_path))

    # Calculate average width
    bbox_widths = []
    for image in coco.images:
        for annotation in image.annotations:
            bbox_widths.append(math.sqrt(annotation.area))
    stats_dict['average_object_width'] = np.mean(bbox_widths)

    save_stats_table(stats_dict, save_path)
    return stats_dict

def datasets_stats_from_config(config):
    with open(config) as json_file:
        config = json.load(json_file)

    datasets_stats = []
    for dataset in config['dataset_config']:
        coco_path = Path(dataset['coco_annotation_file_path'])
        from BirdMOT.data.DatasetCreator import DatasetCreator
        datasets_stats.append(
            {
                'name': dataset['name'],
                'stats': calculate_dataset_stats(DatasetCreator().coco_files_dir / coco_path, DatasetCreator().dataset_stats_dir)
             })
    print(datasets_stats)
    save_dataset_comparison_stats_table(datasets_stats, DatasetCreator().dataset_stats_dir)
    return datasets_stats

if __name__ == "__main__":
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(help='commands')

    from_coco_parser = subparsers.add_parser('from_coco', help='Create stats from coco files')
    from_coco_parser.add_argument('train_path', type=Path, help='Path to train coco')
    from_coco_parser.add_argument('val_path', type=Path, help='Path to val coco')
    from_coco_parser.add_argument('image_path', type=Path, help='Path to images')
    from_coco_parser.add_argument('save_path', type=Path, help='Path save location')

    from_config_parser = subparsers.add_parser('from_config', help='Create stats from config')
    from_config_parser.add_argument('config_path', type=Path, help='Path to config')


    args = parser.parse_args()

    print(args)

    if 'train_path' in args and 'val_path' in args:
        train_coco = Coco.from_coco_dict_or_path(args.train_path.as_posix(), image_dir=args.image_path.as_posix())
        val_coco = Coco.from_coco_dict_or_path(args.val_path.as_posix(), image_dir=args.image_path.as_posix())

        train_coco.merge(val_coco)

        merged_coco_path = args.save_path / 'merged_coco.json'
        save_json(train_coco.json, merged_coco_path)

        calculate_dataset_stats(merged_coco_path, args.save_path)
        #calculate_dataset_stats(Path(args.coco_path), DatasetCreator().dataset_stats_dir)
        pass
    elif 'config' in args:
        datasets_stats_from_config(args.config)


