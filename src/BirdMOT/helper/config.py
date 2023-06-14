import configparser
from pathlib import Path

config = configparser.ConfigParser()
config.sections()
project_root = Path(__file__).parent.parent.parent.parent
assert (project_root / 'birdmot_config.ini').exists()
config.read(project_root / 'birdmot_config.ini')

def get_local_data_path() -> Path:
    return Path(config['DEFAULT']['local_data_path'])

def get_mlflow_tracking_uri():
    return Path(config['MLFLOW']['mlflow_tracking_uri'])