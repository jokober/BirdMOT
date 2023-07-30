import configparser
import os
from pathlib import Path

def get_local_data_path() -> Path:
    assert os.environ.get('BirdMOT_DATA_PATH') is not None, "BirdMOT_DATA_PATH environment variable not set. Please select a target path, where your data will be stored."
    return Path(os.environ.get("BirdMOT_DATA_PATH"))

