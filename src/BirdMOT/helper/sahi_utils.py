def create_sahi_setup_name(one_experiment_config: dict):
    sahi = not "no_sliced_prediction" and one_experiment_config['sahi_prediction_params']['slice_height'] < \
           one_experiment_config['model_config']['imgsz']
    sf = any(
        aug_dataset['height'] < one_experiment_config['model_config']['imgsz'] for aug_dataset
        in one_experiment_config['sliced_datasets'])
    fi = not one_experiment_config['sahi_prediction_params']['no_standard_prediction']
    po = one_experiment_config['sahi_prediction_params']['overlap_height_ratio'] == 0 and \
         one_experiment_config['sahi_prediction_params']['overlap_width_ratio'] == 0

    sahi_setup_name_elements = [get_model_type(one_experiment_config, include_version= True)] + [key for key, value in ({'SF':sf,'SAHI': sahi,'FI':fi, 'PO':po}).items() if value == True]
    setup_name = '_'.join(sahi_setup_name_elements)
    return setup_name
def get_model_type(one_experiment_config: dict, include_version: bool):
    if include_version:
        model_type = 'yolov8n' if 'yolov8n' in one_experiment_config['model_config']['model'] else None
    elif not include_version:
        model_type = 'yolov8' if 'yolov8' in one_experiment_config['model_config']['model'] else None
    else:
        raise NotImplementedError("The type is not implemented.")

    return model_type