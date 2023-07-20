import configparser
import os
from pathlib import Path

config = configparser.ConfigParser()
config.sections()
project_root = Path(__file__).parent.parent.parent.parent
assert (project_root / 'birdmot_config.ini').exists()
config.read(project_root / 'birdmot_config.ini')

def get_local_data_path() -> Path:
    assert os.environ.get('BirdMOT_DATA_PATH') is not None, "BirdMOT_DATA_PATH environment variable not set. Please select a target path, where your data will be stored."
    return Path(os.environ.get("BirdMOT_DATA_PATH"))

