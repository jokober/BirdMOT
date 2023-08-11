import glob
import pickle
import shutil
from copy import deepcopy
from pathlib import Path

from sahi.slicing import slice_coco
from sahi.utils.coco import Coco, export_coco_as_yolov5

from BirdMOT.data import SliceParams
from BirdMOT.data.dataset_tools import rapair_absolute_image_paths, assemble_dataset_from_config, assemble_dataset, \
    merge_coco_datasets
from BirdMOT.helper.config import get_local_data_path

from deepdiff import DeepHash


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

        if assembly_hash not in [ass_conf["hash"] for ass_conf in self.state['assemblies']]:
            print("Assembly hash not found in existing assemblies. Creating new dataset assembly.")
            assembly_config['hash'] = assembly_hash

            dataset_assembly_results = assemble_dataset(output_path=self.tmp_assemblies_path,
                                                        assembly_config=assembly_config,
                                                        coco_files_path=self.coco_files_dir,
                                                        image_path=self.images_dir,
                                                        categories_path=self.coco_categories_path)
            del dataset_assembly_results['train']['coco']  # ToDo: Implement if needed
            del dataset_assembly_results['val']['coco']  # ToDo: Implement if needed
            assembly_config['data'] = dataset_assembly_results
            self.update_state(type="append", key='assemblies', value=assembly_config)

            return assembly_config

        else:
            return [ass_conf for ass_conf in self.state['assemblies'] if ass_conf["hash"] == assembly_hash][0]

    def find_or_create_sliced_dataset(self, assembly_config: dict, one_sliced_dataset_config: dict) -> dict:
        assembly_config = deepcopy(assembly_config)
        one_sliced_dataset_config = deepcopy(one_sliced_dataset_config)

        dataset_assembly = self.find_or_create_dataset_assembly(assembly_config)
        one_sliced_dataset_config["assembly_hash"] = dataset_assembly["hash"]
        sliced_dataset_hash = DeepHash(one_sliced_dataset_config)[one_sliced_dataset_config]

        if sliced_dataset_hash not in [sliced_dat["hash"] for sliced_dat in self.state['sliced_datasets']]:
            print("Sliced Dataset hash not found in existing assemblies. Creating new sliced dataset.")
            print(self.images_dir)
            one_sliced_dataset_config["data"] = {}
            for split in ("train", "val"):
                coco_dict, coco_path = slice_coco(
                    coco_annotation_file_path=dataset_assembly['data'][split]['path'].as_posix(),
                    image_dir=self.images_dir.as_posix(),
                    output_coco_annotation_file_name=f"sliced_{split}",
                    ignore_negative_samples=assembly_config['ignore_negative_samples'],
                    output_dir=(self.sliced_datasets_dir / sliced_dataset_hash).as_posix(),
                    slice_height=one_sliced_dataset_config['height'],
                    slice_width=one_sliced_dataset_config['width'],
                    overlap_height_ratio=one_sliced_dataset_config['overlap_height_ratio'],
                    overlap_width_ratio=one_sliced_dataset_config['overlap_width_ratio'],
                    min_area_ratio=one_sliced_dataset_config['min_area_ratio'],
                    verbose=False,
                )
                # Merge negative samples with sliced dataset
                if assembly_config['ignore_negative_samples'] == True:
                    neg_coco_dict, neg_coco_path = slice_coco(
                        coco_annotation_file_path=dataset_assembly['data']['negatives_' + split]['path'].as_posix(),
                        image_dir=self.images_dir.as_posix(),
                        output_coco_annotation_file_name=f"negatives_sliced_{split}",
                        ignore_negative_samples=False,
                        output_dir=(self.sliced_datasets_dir / sliced_dataset_hash).as_posix(),
                        slice_height=one_sliced_dataset_config['height'],
                        slice_width=one_sliced_dataset_config['width'],
                        overlap_height_ratio=one_sliced_dataset_config['overlap_height_ratio'],
                        overlap_width_ratio=one_sliced_dataset_config['overlap_width_ratio'],
                        min_area_ratio=one_sliced_dataset_config['min_area_ratio'],
                        verbose=False,
                    )
                    merge_coco_datasets([coco_path, neg_coco_path], self.sliced_datasets_dir / sliced_dataset_hash, self.coco_categories_path / assembly_config[
                                                      'coco_formatted_categories'], coco_path)

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
        ftd_hash = DeepHash(fine_tuning_dataset, ignore_string_type_changes=True, ignore_repetition=True,
                            ignore_numeric_type_changes=True)[fine_tuning_dataset]

        if ftd_hash not in [one_fine_tuning_dataset["hash"] for one_fine_tuning_dataset in
                            self.state['fine_tuning_datasets']]:
            print("Sliced Dataset hash not found in existing assemblies. Creating new sliced dataset.")

            dataset_path = self.tmp_fine_tuning_datasets_dir / ftd_hash
            dataset_path.mkdir(parents=True, exist_ok=True)

            fine_tuning_dataset['data'] = {
                "dataset_path": dataset_path,
            }
            fine_tuning_dataset["hash"] = ftd_hash

            # Create symlinks to sliced datasets images
            for one_sliced_dataset in sliced_datasets:
                types = ('**/*.png', '**/*.jpg')
                files_grabbed = []
                for files in types:
                    files_grabbed.extend(one_sliced_dataset["data"]["train"]["path"].parent.glob(files))

                for orig_file in files_grabbed:
                    symlink_path = Path(dataset_path / orig_file.name)
                    assert not symlink_path.exists(), f"Symlink path {symlink_path} already exists"
                    symlink_path.symlink_to(orig_file)

            # Merge all sliced datasets coco files
            for split in ("train", "val"):
                coco_files = [sl_dataset["data"][split]["path"] for sl_dataset in sliced_datasets]
                merged_coco = merge_coco_datasets(coco_json_paths=coco_files,
                                                  image_paths=dataset_path,
                                                  categories=self.coco_categories_path / assembly_config[
                                                      'coco_formatted_categories'],
                                                  output_path=dataset_path / f"{split}.json")
                fine_tuning_dataset['data'][split] = {
                    "path": dataset_path / f"{split}.json",
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

    def createYolov5Dataset(self, parent_dataset_path: Path, train_coco_path: Path, val_coco_path,
                            overwrite_existing: bool = True):
        sliced_images_dir = parent_dataset_path / "images"
        yolov5_dataset_path = parent_dataset_path / "yolov5_files"

        # Make sure the absolute paths in coco files are correct
        rapair_absolute_image_paths(train_coco_path, sliced_images_dir, overwrite_file=True)
        rapair_absolute_image_paths(val_coco_path, sliced_images_dir, overwrite_file=True)

        # Check if file indicating successful dataset creation exists
        check_finished = (yolov5_dataset_path / 'dataset_creation_finished.txt').exists()

        if not check_finished or (overwrite_existing and yolov5_dataset_path.exists()):
            shutil.rmtree(yolov5_dataset_path)
        if not yolov5_dataset_path.exists():
            (parent_dataset_path / "yolov5_files").mkdir(parents=True)

            # init Coco object
            train_coco = Coco.from_coco_dict_or_path(train_coco_path.as_posix(), image_dir=sliced_images_dir.as_posix())
            val_coco = Coco.from_coco_dict_or_path(val_coco_path.as_posix(), image_dir=sliced_images_dir.as_posix())

            # export converted YoloV5 formatted dataset into given output_dir with given train/val split
            data_yml_path = export_coco_as_yolov5(
                output_dir=(parent_dataset_path / 'yolov5_files').as_posix(),
                train_coco=train_coco,
                val_coco=val_coco,
                numpy_seed=0,
            )

            shutil.copy((parent_dataset_path / 'yolov5_files' / "data.yml").as_posix(),
                        (parent_dataset_path / 'yolov5_files' / "data.yaml").as_posix())

            with open(yolov5_dataset_path / 'dataset_creation_finished.txt', 'w') as fp:
                pass

        return yolov5_dataset_path
