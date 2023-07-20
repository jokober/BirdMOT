import os

from BirdMOT.helper.config import get_local_data_path


def test_get_local_data_path():
    os.environ.get("BirdMOT_DATA_PATH")

    print(get_local_data_path)
    assert get_local_data_path() is not None, "BirdMOT_DATA_PATH environment variable not set. Please select a target path, where your data will be stored."
