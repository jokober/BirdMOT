import pathlib
from argparse import ArgumentParser

from BirdMOT.data.dataset_stats import save_dataset_comparison_stats_table
from BirdMOT.data.dataset_tools import assemble_dataset_from_config

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("assembly_name", type=str)
    parser.add_argument("--coco_annotations_dir", type=pathlib.Path, required=True)
    parser.add_argument("--output_path", type=pathlib.Path, required=False)
    parser.add_argument("--categories_path", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    args = parser.parse_args()

    # Create dataset assembled by the config
    dataset_assembly_results = assemble_dataset_from_config(args.assembly_name, args.config,
                                                            args.coco_annotations_dir, args.output_path,
                                                            args.categories_path)

    # Create stats and save them
    datasets_stats = [{
        'name': args.assembly_name + '_train',
        'stats': dataset_assembly_results['train']['coco'].stats
    },{
        'name': args.assembly_name + '_val',
        'stats': dataset_assembly_results['val']['coco'].stats
    }
    ]
    save_dataset_comparison_stats_table(datasets_stats, args.output_path / args.assembly_name)
