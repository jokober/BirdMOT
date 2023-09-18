from BirdMOT.helper.sahi_utils import create_sahi_setup_name, get_model_type
from fixtures.fixtures import one_experiment_config_fixture

def test_create_sahi_setup_name(one_experiment_config_fixture):
    name1 = create_sahi_setup_name(one_experiment_config_fixture)
    assert name1 == "yolov8n_SF_FI"

    one_experiment_conf_po = one_experiment_config_fixture
    one_experiment_conf_po['sahi_prediction_params']['overlap_height_ratio'] = 0
    one_experiment_conf_po['sahi_prediction_params']['overlap_width_ratio'] = 0
    name_po = create_sahi_setup_name(one_experiment_config_fixture)
    assert name_po == "yolov8n_SF_FI_PO"

def test_get_model_type(one_experiment_config_fixture):
    assert get_model_type(one_experiment_config_fixture, include_version=True) == 'yolov8n'
    assert get_model_type(one_experiment_config_fixture, include_version=False) == 'yolov8'