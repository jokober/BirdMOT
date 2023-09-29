from BirdMOT.helper.sahi_utils import get_setup_name, get_model_type, get_setup_name
from fixtures.fixtures import one_experiment_config_fixture


def test_get_setup_name(one_experiment_config_fixture):
    name1 = get_setup_name(one_experiment_config_fixture)
    assert name1 == "yolov8n_SF_FI_PO"

    one_experiment_conf_po = one_experiment_config_fixture
    one_experiment_conf_po['sahi_prediction_params']['overlap_height_ratio'] = 0
    one_experiment_conf_po['sahi_prediction_params']['overlap_width_ratio'] = 0
    name_po = get_setup_name(one_experiment_conf_po)
    assert name_po == "yolov8n_SF_FI"

    one_experiment_conf_sahi = one_experiment_config_fixture
    one_experiment_conf_sahi['sahi_prediction_params']['slice_height'] = 400
    one_experiment_conf_sahi['sahi_prediction_params']["no_sliced_prediction"] = False
    name_po = get_setup_name(one_experiment_conf_sahi)
    assert name_po == "yolov8n_SF_SAHI_FI"

    one_experiment_conf = one_experiment_config_fixture
    one_experiment_conf['sliced_datasets'] = [{
        "height": 640,
        "width": 640,
        "overlap_height_ratio": 0.1,
        "overlap_width_ratio": 0.1,
        "min_area_ratio": 0.2,
        "ignore_negative_samples": False
    }]

    name_po = get_setup_name(one_experiment_conf)
    assert name_po == "yolov8n_SAHI_FI"


def test_get_model_type(one_experiment_config_fixture):
    assert get_model_type(one_experiment_config_fixture, include_version=True) == 'yolov8n'
    assert get_model_type(one_experiment_config_fixture, include_version=False) == 'yolov8'
