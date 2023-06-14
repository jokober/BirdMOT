from pathlib import Path
from unittest.mock import patch

import pytest
from _pytest.tmpdir import tmp_path
from BirdMOT.helper.config import get_local_data_path
from fixtures.fixtures import local_data_path_fixture


@patch('BirdMOT.helper.config.get_local_data_path', side_effect=local_data_path_fixture)
def test_get_local_data_path(mocker):
    print(get_local_data_path)
    assert get_local_data_path() == local_data_path_fixture, "Value should be mocked"
