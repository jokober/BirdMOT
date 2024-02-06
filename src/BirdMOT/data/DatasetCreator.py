import math
import os
import pickle
import shutil
import time
from copy import deepcopy
from pathlib import Path

from deepdiff import DeepHash
from sahi.slicing import slice_coco
from sahi.utils.coco import export_coco_as_yolov5, Coco
from sahi.utils.file import save_json

from BirdMOT.data.dataset_stats import calculate_dataset_stats
from BirdMOT.data.dataset_tools import assemble_dataset, \
    merge_coco_datasets
from BirdMOT.helper.config import get_local_data_path


class DatasetCreator:
    def __init__(self):

        self.local_data_path = get_local_data_path()

        self.dataset_dir: Path = self.local_data_path / 'dataset'
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.coco_files_dir: Path = self.local_data_path / 'dataset' / 'coco_files'
        self.coco_files_dir.mkdir(parents=True, exist_ok=True)

        self.coco_categories_path: Path = self.local_data_path / 'configs' / 'categories'
        self.coco_categories_path.mkdir(parents=True, exist_ok=True)

        self.images_dir: Path = self.local_data_path / 'dataset' / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_stats_dir: Path = self.local_data_path / 'dataset' / 'stats'
        self.dataset_stats_dir.mkdir(parents=True, exist_ok=True)

        # Temporary directory for storing intermediate results
        self.tmp_dir_path: Path = self.local_data_path / 'tmp_dir'
        self.tmp_dir_path.mkdir(parents=True, exist_ok=True)

        self.tmp_state_path: Path = self.tmp_dir_path / 'tmp_state.pkl'

        self.tmp_assemblies_path: Path = self.tmp_dir_path / 'dataset_assemblies'
        self.tmp_assemblies_path.mkdir(parents=True, exist_ok=True)

        self.tmp_fine_tuning_datasets_dir: Path = self.tmp_dir_path / 'fine_tuning_datasets'
        self.tmp_fine_tuning_datasets_dir.mkdir(parents=True, exist_ok=True)

        self.tmp_yolov5_fine_tuning_datasets_dir: Path = self.tmp_dir_path / 'yolov5_fine_tuning_datasets'
        self.tmp_yolov5_fine_tuning_datasets_dir.mkdir(parents=True, exist_ok=True)

        self.sliced_datasets_dir: Path = self.tmp_dir_path / 'sliced_datasets'
        self.sliced_datasets_dir.mkdir(parents=True, exist_ok=True)

        self.val_images_dir: Path = self.local_data_path / 'validation_images'  # ToDo: Refactor
        self.val_images_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir: Path = self.local_data_path / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_dir: Path = self.local_data_path / 'predictions'
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        self.state = None
        self.load_state()

    def write_state(self):
        with open(self.tmp_state_path, 'wb') as handle:
            pickle.dump(self.state, handle)

    def load_state(self):
        if self.tmp_state_path.exists():
            with open(self.tmp_state_path, 'rb') as handle:
                self.state = pickle.load(handle)
        else:
            print("tmp_state_path does not exist. Creating new state.")
            self.create_new_datasets_state()
            self.write_state()

    def update_state(self, type, key, value):
        self.load_state()
        if type == 'append':
            self.state[key].append(value)
        else:
            raise NotImplementedError("The type is not implemented.")

        self.write_state()

    def create_new_datasets_state(self):
        self.state = {
            'assemblies': [],
            'sliced_datasets': [],
            'fine_tuning_datasets': [],
            'yolov5_fine_tuning_datasets': [],
        }

    def find_or_create_dataset_assembly(self, assembly_config: dict) -> dict:
        assembly_config = deepcopy(assembly_config)
        assembly_hash = DeepHash(assembly_config)[assembly_config]

        assembly_in_state = [ass_conf for ass_conf in self.state['assemblies'] if ass_conf["hash"] == assembly_hash]
        if len(assembly_in_state) == 0 or not assembly_in_state[0]['data']['train']['path'].exists():
            print("Assembly hash not found in existing assemblies. Creating new dataset assembly.")
            assembly_config['hash'] = assembly_hash

            dataset_assembly_results = assemble_dataset(output_path=self.tmp_assemblies_path,
                                                        assembly_config=assembly_config,
                                                        coco_files_path=self.coco_files_dir,
                                                        image_path=self.images_dir,
                                                        categories_path=self.coco_categories_path)

            # Assert that there are no images with a width value of 3640. This is only required as there where a faulty
            # Datsaset #ToDO: Remove this assertion as soon as the faulty dataset is removed
            try:
                is_faulty_dimension_list = [image.width == 3640 for image in
                                            dataset_assembly_results['train']['coco'].images]
                assert not any(is_faulty_dimension_list)
            except:
                raise AssertionError(
                    f"There are images with a width value of 3640. Check faulty images: {[b for a, b in zip(is_faulty_dimension_list, dataset_assembly_results['train']['coco'].images) if a]}")

            del dataset_assembly_results['train']['coco']  # ToDo: Implement if needed
            del dataset_assembly_results['val']['coco']  # ToDo: Implement if needed

            train_stats = calculate_dataset_stats(dataset_assembly_results['train']['path'],
                                                  self.tmp_assemblies_path / assembly_hash / "stats" / 'train')
            val_stats = calculate_dataset_stats(dataset_assembly_results['val']['path'],
                                                self.tmp_assemblies_path / assembly_hash / "stats" / 'val')

            assert train_stats['num_negatives'] / train_stats[
                'num_images'] < 0.22, f"The sliced dataset contains too many negative samples.{train_stats['num_negatives'] / train_stats['num_images']}"
            assert val_stats['num_negatives'] / val_stats[
                'num_images'] < 0.22, f"The sliced dataset contains too many negative samples.{val_stats['num_negatives'] / val_stats['num_images']}"

            assembly_config['data'] = dataset_assembly_results
            self.update_state(type="append", key='assemblies', value=assembly_config)

            return assembly_config

        else:
            return assembly_in_state[0]

    def find_or_create_sliced_dataset(self, assembly_config: dict, one_sliced_dataset_config: dict) -> dict:
        """
        Creates a sliced dataset from Full Res Train/Val sets created in dataset assembly.

        """
        assembly_config = deepcopy(assembly_config)
        one_sliced_dataset_config = deepcopy(one_sliced_dataset_config)

        dataset_assembly = self.find_or_create_dataset_assembly(assembly_config)
        one_sliced_dataset_config["assembly_hash"] = dataset_assembly["hash"]

        deephash_exclude_paths = [
            "root['one_experiment_config']['model_config']",
            "root['one_experiment_config']['sahi_prediction_params']",
            "root['one_experiment_config']['evaluation_config']",
            "root['one_experiment_config']['hash']",
            "root['data']",
            "root['hash']",
        ]
        sliced_dataset_hash = DeepHash(one_sliced_dataset_config, exclude_paths=deephash_exclude_paths)[
            one_sliced_dataset_config]

        if sliced_dataset_hash not in [sliced_dat["hash"] for sliced_dat in self.state['sliced_datasets']]:
            print("Sliced Dataset hash not found in existing assemblies. Creating new sliced dataset.")

            # Relative paths from sliced dataset folder to image folders
            rel_sl_img_path_positives = Path(
                f"./images/{one_sliced_dataset_config['height']}x{one_sliced_dataset_config['width']}_{one_sliced_dataset_config['overlap_height_ratio']}_{one_sliced_dataset_config['overlap_width_ratio']}") / 'positives'
            abs_sl_img_path_positives = self.sliced_datasets_dir / sliced_dataset_hash / rel_sl_img_path_positives

            # Do the splitting of the positive samples
            one_sliced_dataset_config["data"] = {}
            for split in ("train", "val"):
                coco_dict, coco_path = slice_coco(
                    coco_annotation_file_path=dataset_assembly['data'][split]['path'].as_posix(),
                    image_dir=self.images_dir.as_posix(),
                    output_coco_annotation_file_name=f"sliced_{split}",
                    ignore_negative_samples=False,
                    output_dir=abs_sl_img_path_positives.as_posix(),
                    slice_height=one_sliced_dataset_config['height'],
                    slice_width=one_sliced_dataset_config['width'],
                    overlap_height_ratio=one_sliced_dataset_config['overlap_height_ratio'],
                    overlap_width_ratio=one_sliced_dataset_config['overlap_width_ratio'],
                    min_area_ratio=one_sliced_dataset_config['min_area_ratio'],
                    out_ext=".png",
                    verbose=False,
                )

                # Subsample in order to limit amount of negative images
                coco = Coco.from_coco_dict_or_path(coco_dict)
                ratio = len([coco_image for coco_image in coco.images if len(coco_image.annotations) == 0]) / (
                            len([coco_image for coco_image in coco.images if len(coco_image.annotations) != 0]) * 1 /
                            assembly_config['max_negatives_ratio'])
                coco = coco.get_subsampled_coco(
                    subsample_ratio=math.ceil(ratio), category_id=-1)
                save_json(coco.json, coco_path.as_posix())

                stats = calculate_dataset_stats(coco_path,
                                                self.sliced_datasets_dir / sliced_dataset_hash / "stats" / split)
                assert stats['num_negatives'] / stats[
                    'num_images'] < 0.26, f"The sliced dataset contains too many negative samples.{stats['num_negatives'] / stats['num_images']}"

                # Move coco file
                coco_path = Path(shutil.move(coco_path, self.sliced_datasets_dir / sliced_dataset_hash))
                # Change file_name to match folder structure
                for split_subfolder in range(int(len(coco.images) / 5000) + 1):
                    (abs_sl_img_path_positives / str(split_subfolder)).mkdir(parents=True, exist_ok=True)
                for c, image in enumerate(coco.images):
                    split_subfolder = int(c / 5000)
                    # Apparently there was a race condition. This is a quick fix
                    timer_counter = 0
                    while not (abs_sl_img_path_positives / image.file_name).exists():
                        time.sleep(1)
                        timer_counter += 1
                        if timer_counter > 60:
                            raise TimeoutError("The sliced dataset images were not created in time.")
                    shutil.move(abs_sl_img_path_positives / image.file_name,
                                abs_sl_img_path_positives / str(split_subfolder) / image.file_name)
                    image.file_name = (
                            rel_sl_img_path_positives / str(split_subfolder) / image.file_name).as_posix()
                coco_dict = coco.json
                save_json(coco_dict, coco_path.as_posix())

                # Remove unused slices
                types = ('.png', '.jpg')
                files_grabbed = []
                for suffix in types:
                    [os.remove(file) for file in abs_sl_img_path_positives.glob('*' + suffix)]

                stats = calculate_dataset_stats(coco_path,
                                                self.sliced_datasets_dir / sliced_dataset_hash / "stats" / split)
                assert stats['num_negatives'] / stats[
                    'num_images'] < 0.26, f"The sliced dataset contains too many negative samples.{stats['num_negatives'] / stats['num_images']}"

                one_sliced_dataset_config["data"][split] = {
                    "path": coco_path,
                    "dict": coco_dict,
                }

            one_sliced_dataset_config["hash"] = sliced_dataset_hash
            self.update_state(type="append", key='sliced_datasets', value=one_sliced_dataset_config)

            return one_sliced_dataset_config
        else:
            return \
                [sliced_dat for sliced_dat in self.state['sliced_datasets'] if
                 sliced_dat["hash"] == sliced_dataset_hash][0]

    def find_or_create_fine_tuning_dataset(self, assembly_config, sliced_dataset_configs):
        """
        Merges all relevant sliced datasets into one fine tuning dataset.
        Args:
            assembly_config:
            sliced_dataset_configs:

        Returns:

        """

        assembly_config = deepcopy(assembly_config)
        sliced_dataset_configs = deepcopy(sliced_dataset_configs)

        dataset_assembly = self.find_or_create_dataset_assembly(assembly_config)
        sliced_datasets = []
        for one_sliced_dataset_config in sliced_dataset_configs:
            one_sliced_dataset_config["assembly_hash"] = dataset_assembly["hash"]
            sliced_datasets.append(self.find_or_create_sliced_dataset(dataset_assembly, one_sliced_dataset_config))

        fine_tuning_dataset = {
            'assembly': dataset_assembly,
            'sliced_datasets': sliced_datasets,
        }
        deephash_exclude_paths = [
            "root['one_experiment_config']['model_config']",
            "root['one_experiment_config']['sahi_prediction_params']",
            "root['one_experiment_config']['evaluation_config']",
            "root['one_experiment_config']['hash']",
            "root['data']",
            "root['hash']",
        ]

        ftd_hash = DeepHash(fine_tuning_dataset, exclude_paths=deephash_exclude_paths)[fine_tuning_dataset]

        if ftd_hash not in [one_fine_tuning_dataset["hash"] for one_fine_tuning_dataset in
                            self.state['fine_tuning_datasets']]:
            print("Sliced Dataset hash not found in existing assemblies. Creating new sliced dataset.")

            ft_dataset_path = self.tmp_fine_tuning_datasets_dir / ftd_hash
            ft_dataset_path.mkdir(parents=True, exist_ok=True)

            fine_tuning_dataset['data'] = {
                "dataset_path": ft_dataset_path,
            }
            fine_tuning_dataset["hash"] = ftd_hash

            # Create symlinks to sliced datasets images
            for one_sliced_dataset in sliced_datasets:
                sliced_datasets_path = one_sliced_dataset["data"]["train"]["path"].parent
                types = ('.png', '.jpg')
                files_grabbed = []
                for suffix in types:
                    files_grabbed.extend((sliced_datasets_path / 'images').glob('**/*' + suffix))

                for orig_file in files_grabbed:
                    rel_img_path = os.path.relpath(orig_file, sliced_datasets_path)
                    img_dir = (ft_dataset_path / rel_img_path).resolve().parent
                    print(img_dir)
                    img_dir.mkdir(parents=True, exist_ok=True)
                    symlink_path = img_dir / orig_file.name
                    assert not symlink_path.exists(), f"""Symlink path {symlink_path} already exists. This might be due
                    to a failed previous attempt to create the fine tuning dataset. Try to remove the assoziated fine
                    tuning folder and try again.
                    """
                    symlink_path.symlink_to(orig_file)

            # Merge all sliced datasets coco files
            for split in ("train", "val"):
                coco_files = [sl_dataset["data"][split]["path"] for sl_dataset in sliced_datasets]
                coco_path = ft_dataset_path / f"{split}.json"
                merged_coco = merge_coco_datasets(coco_json_paths=coco_files,
                                                  image_paths=ft_dataset_path,
                                                  categories=self.coco_categories_path / assembly_config[
                                                      'coco_formatted_categories'],
                                                  output_path=coco_path)

                stats = calculate_dataset_stats(coco_path, ft_dataset_path / "stats" / split)
                assert stats['num_negatives'] / stats[
                    'num_images'] < 0.26, f"The sliced dataset contains too many negative samples.{stats['num_negatives'] / stats['num_images']}"

                fine_tuning_dataset['data'][split] = {
                    "path": coco_path,
                    "coco": merged_coco,
                }

            self.update_state(type="append", key='fine_tuning_datasets', value=fine_tuning_dataset)
            return fine_tuning_dataset.copy()

        else:
            return [one_fine_tuning_dataset for one_fine_tuning_dataset in self.state['fine_tuning_datasets'] if
                    one_fine_tuning_dataset["hash"] == ftd_hash][0]

    def find_or_create_yolov5_dataset(self, assembly_config, sliced_dataset_configs):
        assembly_config = deepcopy(assembly_config)
        sliced_dataset_configs = deepcopy(sliced_dataset_configs)

        fine_tuning_dataset = self.find_or_create_fine_tuning_dataset(assembly_config, sliced_dataset_configs)
        fine_tuning_dataset_hash = fine_tuning_dataset['hash']

        if fine_tuning_dataset_hash not in [ft_dataset["fine_tuning_dataset_hash"] for ft_dataset in
                                            self.state['yolov5_fine_tuning_datasets']]:
            print("Yolov5 fine tuning dataset has not been found. Creating new yolov5 fine tuning dataset.")

            dataset_path = self.tmp_yolov5_fine_tuning_datasets_dir / fine_tuning_dataset_hash

            # export converted YoloV5 formatted dataset into given output_dir with given train/val split
            data_yml_path = export_coco_as_yolov5(
                output_dir=dataset_path.as_posix(),
                train_coco=fine_tuning_dataset['data']['train']['coco'],
                val_coco=fine_tuning_dataset['data']['val']['coco'],
                numpy_seed=0,
            )

            one_yolov5_fine_tuning_dataset = {
                'fine_tuning_dataset_hash': fine_tuning_dataset_hash,
                'data_yml_path': Path(data_yml_path)
            }
            self.update_state(type="append", key='yolov5_fine_tuning_datasets', value=one_yolov5_fine_tuning_dataset)
            return one_yolov5_fine_tuning_dataset
        else:
            return [one_yolov5_fine_tuning_datasets for one_yolov5_fine_tuning_datasets in
                    self.state['yolov5_fine_tuning_datasets'] if
                    one_yolov5_fine_tuning_datasets["fine_tuning_dataset_hash"] == fine_tuning_dataset_hash][0]
