import pickle
from copy import deepcopy
from pathlib import Path

import pandas
from deepdiff import DeepHash

from BirdMOT.data.DatasetCreator import DatasetCreator
from BirdMOT.detection.yolov8 import sliced_yolov8_train_2
from BirdMOT.helper.config import get_local_data_path


class TrainingController:
    def __init__(self):
        self.local_data_path = get_local_data_path()
        self.tmp_train_dir_path = self.local_data_path / 'tmp_train_dir'
        self.tmp_train_dir_path.mkdir(parents=True, exist_ok=True)
        self.tmp_train_state_path: Path = self.tmp_train_dir_path / 'tmp_train_state.pkl'
        self.tmp_model_path: Path = self.tmp_train_dir_path / 'models'
        self.tmp_model_path.mkdir(parents=True, exist_ok=True)

        self.state = None
        self.load_state()

    def write_state(self):
        with open(self.tmp_train_state_path, 'wb') as handle:
            pickle.dump(self.state, handle)

    def load_state(self):
        if self.tmp_train_state_path.exists():
            with open(self.tmp_train_state_path, 'rb') as handle:
                self.state = pickle.load(handle)
        else:
            print("tmp_state_path does not exist. Creating new state.")
            self.create_new_eval_state()
            self.write_state()

    def create_new_eval_state(self):
        self.state = {
            'models': [],
        }
        self.write_state()

    def update_state(self, type, key, value):
        self.load_state()
        if type == 'append':
            self.state[key].append(value)
        elif type == 'delete':
            self.state[key] = [item["hash"] for item in self.state[key] if item != value["hash"]]
        else:
            raise NotImplementedError("The type is not implemented.")
        self.write_state()

    def find_or_train_model(self, one_experiment_config: dict, assembly_config=None, device='cpu', train_missing=True):
        one_experiment_config = deepcopy(one_experiment_config)
        assembly_config = deepcopy(assembly_config)
        assembly = DatasetCreator().find_or_create_dataset_assembly(assembly_config)

        model = {
            "one_experiment_config": one_experiment_config,
            "assembly_config": assembly_config,
            "data": {},
            "hash": None
        }

        deephash_exclude_paths = [
            "root['one_experiment_config']['model_config']['project']",
            "root['one_experiment_config']['model_config']['name']",
            "root['one_experiment_config']['model_config']['device']",
            "root['one_experiment_config']['model_config']['exists_ok']",
            "root['one_experiment_config']['model_config']['conf']",
            "root['one_experiment_config']['sahi_prediction_params']",
            "root['one_experiment_config']['evaluation_config']",
            "root['one_experiment_config']['hash']",
            "root['data']",
            "root['hash']",
        ]
        model_hash = DeepHash(model, exclude_paths=deephash_exclude_paths)[model]

        model_path = self.tmp_model_path / assembly_config['dataset_assembly_id'] / model_hash
        if (model_hash not in [model_conf["hash"] for model_conf in self.state['models']]) or (
                model_hash not in [model_conf["hash"] for model_conf in self.state['models']]):
            if (self.tmp_model_path / assembly_config['dataset_assembly_id'] / model_hash).exists():
                model_path = self.tmp_model_path / assembly_config['dataset_assembly_id'] / model_hash
            else:
                if train_missing:
                    one_experiment_config["model_config"]["project"] = self.tmp_model_path / assembly_config[
                        'dataset_assembly_id']
                    one_experiment_config["model_config"]["name"] = model_hash
                    training_run = sliced_yolov8_train_2(
                        assembly_configs=assembly_config,
                        sliced_dataset_configs=one_experiment_config["sliced_datasets"],
                        yolo_train_params=one_experiment_config["model_config"],
                        device=device)
                    # model['data']['result_dict'] = training_run.result_dict()
                    # model['data']['speed'] = training_run.speed
                    model_path = Path(training_run['save_dir'])
                else:
                    print("state models")
                    print(self.state['models'])
                    print("model")
                    print(model)
                    raise NotImplementedError(f"""The model does not exist and train_missing is False. Searching for a model 
                                              in following directory also failed:
                                              {(self.tmp_model_path / assembly_config['dataset_assembly_id'] / one_experiment_config['model_config']['name'])}
                                              """)

            model['data']['model_path'] = model_path
            print((Path(model['data']['model_path']) / 'results.csv').as_posix())
            model['data']['results_df'] = pandas.read_csv(model_path / 'results.csv')
            model['data']['weights_path'] = model_path / 'weights' / 'best.pt'
            # if not model['data']['weights_path'].exists():  # ToDo: Remove this if statement after problem of best.pt not generated is solved
            #    model['data']['weights_path'] = model['data']['weights_path'].parent / 'last.pt'
            #    assert model['data']['weights_path'].exists(), f"weights_path does not exist: {model['data']['weights_path']} Neither for best.pt nor for last.pt"
            model['hash'] = model_hash

            self.update_state(type="append", key='models', value=model)
        else:
            model = \
                [model for model in self.state['models'] if model["hash"] == model_hash or model["hash"] == model_hash][
                    0]

        return model
