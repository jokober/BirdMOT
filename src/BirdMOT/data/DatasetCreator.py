import shutil
from pathlib import Path

from sahi.utils.coco import Coco, export_coco_as_yolov5

from BirdMOT.data import SliceParams
from BirdMOT.data.dataset_tools import rapair_absolute_image_paths
from BirdMOT.data.slice_data import slice_dataset
from BirdMOT.helper.config import get_local_data_path


class DatasetCreator:
    def __init__(self):
        self.local_data_path = get_local_data_path()

        self.dataset_dir: Path = self.local_data_path / 'dataset'
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.coco_files_dir: Path = self.local_data_path / 'dataset' / 'coco_files'
        self.coco_files_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir: Path = self.local_data_path / 'dataset' / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_assemblies_dir: Path = self.local_data_path / 'dataset' / 'coco_files' / 'dataset_assemblies'
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_stats_dir: Path = self.local_data_path / 'dataset' / 'stats'
        self.dataset_stats_dir.mkdir(parents=True, exist_ok=True)

        self.val_images_dir: Path = self.local_data_path / 'validation_images'
        self.dataset_assemblies_dir.mkdir(parents=True, exist_ok=True)

        self.sliced_datasets_dir: Path = self.local_data_path / 'sliced_datasets'
        self.sliced_datasets_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir: Path = self.local_data_path / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_dir: Path = self.local_data_path / 'predictions'
        self.predictions_dir.mkdir(parents=True, exist_ok=True)


    def createSlicedDataset(self, train_coco_path: Path, val_coco_path: Path, image_dir: Path,
                            slice_params: SliceParams,
                            overwrite_existing: bool = False):

        # Make sure the absolute paths in coco files are correct
        rapair_absolute_image_paths(train_coco_path, image_dir, overwrite_file=True)
        rapair_absolute_image_paths(val_coco_path, image_dir, overwrite_file=True)

        dataset_path = self.sliced_datasets_dir / f"{train_coco_path.stem}_{val_coco_path.stem}_{slice_params.height}_{slice_params.width}_{slice_params.overlap_height_ratio}_{slice_params.overlap_width_ratio}_{slice_params.min_area_ratio}_{slice_params.ignore_negative_samples}".replace(
            ".", "_")

        # Check if file indicating successful dataset creation exists
        check_finished = (dataset_path / 'dataset_creation_finished.txt').exists()

        if (dataset_path.exists() and not check_finished) or (dataset_path.exists() and overwrite_existing):
            shutil.rmtree(dataset_path)

        if not dataset_path.exists():
            (dataset_path / "coco_files" / 'annotations').mkdir(parents=True)
            (dataset_path / "yolov5_files").mkdir(parents=True)
            (dataset_path / "images").mkdir(parents=True)
            (dataset_path / "results").mkdir(parents=True)



            train_coco_dict, train_coco_path = slice_dataset(train_coco_path, image_dir=image_dir,
                                                             output_dir=dataset_path,
                                                             output_coco_dir=dataset_path / "coco_files" / 'sliced_train_coco.json',
                                                             slice_params=slice_params)
            val_coco_dict, val_coco_path = slice_dataset(val_coco_path, image_dir=image_dir,
                                                         output_dir=dataset_path,
                                                         output_coco_dir=dataset_path / "coco_files" / 'sliced_val_coco.json',
                                                         slice_params=slice_params)

            # Write an empty file to indicate that the dataset creation is finished without errors
            with open(dataset_path / 'dataset_creation_finished.txt', 'w') as fp:
                pass
        else:
            print(
                "Sliced dataset already exists. If you want to overwrite existing file use overwrite_existing = True.")

        return dataset_path, train_coco_path, val_coco_path

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
