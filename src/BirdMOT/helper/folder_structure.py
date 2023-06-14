from pathlib import Path

from BirdMOT.data import SliceParams
from BirdMOT.helper.config import get_local_data_path


class FolderStructure:
    def __init__(self):
        self.local_data_path = get_local_data_path()

        self.dataset_dir: Path          = self.local_data_path / 'dataset'
        self.coco_files_dir: Path       = self.local_data_path / 'dataset' / 'coco_files'
        self.images_dir: Path           = self.local_data_path / 'dataset' / 'images'
        self.dataset_stats_dir: Path    = self.local_data_path / 'dataset' / 'stats'

        self.sliced_datasets_dir: Path  = self.local_data_path / 'sliced_datasets'
        self.models_dir: Path           = self.local_data_path / 'models'
        self.predictions_dir: Path      = self.local_data_path / 'predictions'

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.coco_files_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_stats_dir.mkdir(parents=True, exist_ok=True)

        self.sliced_datasets_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create_sliced_dataset_folder_path(self, slice_params: SliceParams) -> Path:
        dataset_path = self.sliced_datasets_dir / f"{slice_params.height}_{slice_params.width}_{slice_params.overlap_height_ratio}_{slice_params.overlap_width_ratio}_{slice_params.min_area_ratio}".replace(
            ".", "_")
        if not dataset_path.exists():
            (dataset_path / "coco_files" / 'annotations').mkdir(parents=True)
            (dataset_path / "yolov5_files").mkdir(parents=True)
            (dataset_path / "images").mkdir(parents=True)
            (dataset_path / "results").mkdir(parents=True)

        return dataset_path


folder_structure_obj = FolderStructure()