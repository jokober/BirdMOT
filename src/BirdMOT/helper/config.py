import os
from pathlib import Path
from typing import List


def get_local_data_path() -> Path:
    assert os.environ.get(
        'BirdMOT_DATA_PATH') is not None, "BirdMOT_DATA_PATH environment variable not set. Please select a target path, where your data will be stored."
    return Path(os.environ.get("BirdMOT_DATA_PATH"))


# Get lists of available config files

def get_list_of_dataset_assemblies() -> List[str]:
    return [path.name for path in (get_local_data_path() / 'configs' / 'dataset_assembly').glob('*.json')]


def get_list_of_experiments() -> List[str]:
    return [path.name for path in (get_local_data_path() / 'configs' / 'experiments').glob('*.json')]


# Get individual config files by name

def get_dataset_assembly_by_name(assembly_name: str) -> Path:
    return get_local_data_path() / 'configs' / 'dataset_assembly' / assembly_name


def get_experiment_by_name(experiment_name: str) -> Path:
    return get_local_data_path() / 'configs' / 'experiments' / experiment_name
