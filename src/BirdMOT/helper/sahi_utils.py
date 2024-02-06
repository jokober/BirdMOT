def create_sahi_setup_name(one_experiment_config: dict):
    sahi = (not one_experiment_config['sahi_prediction_params']["no_sliced_prediction"] and
            (one_experiment_config['sahi_prediction_params']['slice_height'] < one_experiment_config['model_config'][
                'imgsz'])
            or one_experiment_config['sahi_prediction_params']['slice_width'] < one_experiment_config['model_config'][
                'imgsz'])
    sf = any(
        aug_dataset['height'] < one_experiment_config['model_config']['imgsz'] for aug_dataset
        in one_experiment_config['sliced_datasets'])
    fi = not one_experiment_config['sahi_prediction_params']['no_standard_prediction']
    po = one_experiment_config['sahi_prediction_params']['overlap_height_ratio'] != 0 and \
         one_experiment_config['sahi_prediction_params']['overlap_width_ratio'] != 0

    sahi_setup_name_elements = [get_model_type(one_experiment_config, include_version=True)] + [key for key, value in (
        {'SF': sf, 'SAHI': sahi, 'FI': fi, 'PO': po}).items() if value == True]
    setup_name = '_'.join(sahi_setup_name_elements)
    # assert setup_name != 'yolov8n', f"The setup name should not be empty. {one_experiment_config}"
    return setup_name


def create_full_resolution_name(one_experiment_config: dict):
    setup_name_elements = [get_model_type(one_experiment_config, include_version=True), 'full_res']
    setup_name = '_'.join(setup_name_elements)
    return setup_name


def get_setup_name(one_experiment_config: dict):
    if "rect" in one_experiment_config['model_config'] and one_experiment_config['model_config']['rect']:
        setup_name = create_full_resolution_name(one_experiment_config)
    else:
        setup_name = create_sahi_setup_name(one_experiment_config)
    return setup_name


def get_model_type(one_experiment_config: dict, include_version: bool):
    if include_version:
        model_type = one_experiment_config['model_config']['model']
        model_type = model_type.removesuffix('.pt')
        model_type = model_type.removesuffix('.json')
    elif not include_version:
        model_type = 'yolov8' if 'yolov8' in one_experiment_config['model_config']['model'] else None
    else:
        raise NotImplementedError("The type is not implemented.")

    return model_type
