import json
import shutil
from pathlib import Path

from sahi.utils.coco import Coco

from BirdMOT.data.DatasetCreator import DatasetCreator
from fixtures.fixtures import assembly_configs, sliced_dataset_configs


def test_find_or_create_dataset_assembly(assembly_configs):
    dataset_creator = DatasetCreator()
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_train_dir' / 'tmp_train_state.pkl', ignore_errors=True)
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_eval_dir', ignore_errors=True)
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_dir', ignore_errors=True)

    assert len(dataset_creator.state['assemblies']) == 0, "There should be no assemblies in the state"

    returned_assembly = dataset_creator.find_or_create_dataset_assembly(assembly_configs)
    assert 'hash' in returned_assembly, "The returned assembly should have a hash"
    assert returned_assembly['hash'] in [one_assembly['hash'] for one_assembly in dataset_creator.state[
        'assemblies']], "The returned assembly should be in the state"
    assert len(dataset_creator.state['assemblies']) == 1
    assert Path(returned_assembly['data']["train"]["path"]).exists()

    returned_assembly2 = dataset_creator.find_or_create_dataset_assembly(assembly_configs)
    assert len(dataset_creator.state['assemblies']) == 1, "There should still be only one assembly in the state"

    assembly_config2 = assembly_configs
    assembly_config2['dataset_config'][1]["train_split_rate"] = 0.3

    returned_assembly3 = dataset_creator.find_or_create_dataset_assembly(assembly_config2)
    assert returned_assembly3['hash'] in [one_assembly['hash'] for one_assembly in dataset_creator.state[
        'assemblies']], "The returned assembly should be in the state"
    assert len(dataset_creator.state['assemblies']) == 2, "There should be two assemblies in the state"


    # Assert images in split datasets are disjoint
    image_name_set_train = [image.file_name for image in Coco.from_coco_dict_or_path(returned_assembly3['data']['train']['path'].as_posix()).images]
    image_name_set_val = [image.file_name for image in Coco.from_coco_dict_or_path(returned_assembly3['data']['val']['path'].as_posix()).images]
    assert set(image_name_set_train).isdisjoint(image_name_set_val)

def test_find_or_create_sliced_dataset(assembly_configs,
                                       sliced_dataset_configs):
    dataset_creator = DatasetCreator()
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_train_dir' / 'tmp_train_state.pkl', ignore_errors=True)
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_eval_dir', ignore_errors=True)
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_dir', ignore_errors=True)
    dataset_creator = DatasetCreator()

    assert len(dataset_creator.state['sliced_datasets']) == 0, "There should be no sliced_datasets in the state"

    returned_sliced_dataset = dataset_creator.find_or_create_sliced_dataset(assembly_configs,
                                                                            sliced_dataset_configs[0])
    assert 'hash' in returned_sliced_dataset, "The returned sliced dataset should have a hash"
    assert returned_sliced_dataset['hash'] in [one_sliced_dataset['hash'] for one_sliced_dataset in
                                               dataset_creator.state[
                                                   'sliced_datasets']], "The returned sliced_dataset should be in the state"
    assert len(dataset_creator.state['sliced_datasets']) == 1

    returned_sliced_dataset2 = dataset_creator.find_or_create_sliced_dataset(assembly_configs,
                                                                             sliced_dataset_configs[0])
    assert len(
        [dataset_creator.state['sliced_datasets'] == 1]), "There should still be only one sliced_dataset in the state"

    sliced_dataset_config2 = sliced_dataset_configs[0]
    sliced_dataset_config2["width"] = 320

    returned_sliced_dataset3 = dataset_creator.find_or_create_sliced_dataset(assembly_configs,
                                                                             sliced_dataset_config2)
    assert returned_sliced_dataset3['hash'] in [one_sliced_dataset['hash'] for one_sliced_dataset in
                                                dataset_creator.state[
                                                    'sliced_datasets']], "The returned sliced_dataset should be in the state"
    assert len(dataset_creator.state['sliced_datasets']) == 2, "There should be two sliced_datasets in the state"

    assert returned_sliced_dataset['data']['train']['path'].exists()
    assert returned_sliced_dataset['data']['val']['path'].exists()

    assert returned_sliced_dataset2['data']['train']['path'].exists()
    assert returned_sliced_dataset2['data']['val']['path'].exists()

    assert returned_sliced_dataset3['data']['train']['path'].exists()
    assert returned_sliced_dataset3['data']['val']['path'].exists()


def test_find_or_create_fine_tuning_dataset(assembly_configs, sliced_dataset_configs):
    dataset_creator = DatasetCreator()
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_train_dir' / 'tmp_train_state.pkl', ignore_errors=True)
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_eval_dir', ignore_errors=True)
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_dir', ignore_errors=True)

    assert len(
        dataset_creator.state['fine_tuning_datasets']) == 0, "There should be no fine_tuning_dataset in the state"

    returned_fine_tuning_dataset = dataset_creator.find_or_create_fine_tuning_dataset(assembly_configs,
                                                                                      sliced_dataset_configs)
    assert 'hash' in returned_fine_tuning_dataset, "The returned fine_tuning_dataset should have a hash"
    assert returned_fine_tuning_dataset['hash'] in [one_fine_tuning_dataset['hash'] for one_fine_tuning_dataset in
                                                    dataset_creator.state[
                                                        'fine_tuning_datasets']], "The returned fine_tuning_dataset should be in the state"
    assert len(dataset_creator.state['fine_tuning_datasets']) == 1

    returned_fine_tuning_dataset2 = dataset_creator.find_or_create_fine_tuning_dataset(assembly_configs,
                                                                                       sliced_dataset_configs)

    assert len(dataset_creator.state[
                   'fine_tuning_datasets']) == 1, "There should still be only one fine_tuning_dataset in the state"

    assembly_config2 = assembly_configs
    assembly_config2["dataset_config"][1]["train_split_rate"] = 0.2

    returned_fine_tuning_dataset3 = dataset_creator.find_or_create_fine_tuning_dataset(assembly_config2,
                                                                                       sliced_dataset_configs)
    assert returned_fine_tuning_dataset3['hash'] in [one_fine_tuning_dataset['hash'] for one_fine_tuning_dataset in
                                                     dataset_creator.state[
                                                         'fine_tuning_datasets']], "The returned fine_tuning_dataset should be in the state"
    assert len(
        dataset_creator.state['fine_tuning_datasets']) == 2, "There should be two fine_tuning_datasets in the state"

    assert returned_fine_tuning_dataset['data']['train']['path'].exists()
    assert returned_fine_tuning_dataset['data']['val']['path'].exists()
    assert returned_fine_tuning_dataset2['data']['train']['path'].exists()
    assert returned_fine_tuning_dataset2['data']['val']['path'].exists()
    assert returned_fine_tuning_dataset3['data']['train']['path'].exists()
    assert returned_fine_tuning_dataset3['data']['val']['path'].exists()


def test_find_or_create_yolov5_fine_tuning_dataset(assembly_configs, sliced_dataset_configs):
    dataset_creator = DatasetCreator()
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_train_dir' / 'tmp_train_state.pkl', ignore_errors=True)
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_eval_dir', ignore_errors=True)
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_dir', ignore_errors=True)

    assert len(
        dataset_creator.state[
            'yolov5_fine_tuning_datasets']) == 0, "There should be no fine_tuning_dataset in the state"

    returned_yolov5_fine_tuning_dataset = dataset_creator.find_or_create_yolov5_dataset(assembly_configs,
                                                                                        sliced_dataset_configs)
    assert 'fine_tuning_dataset_hash' in returned_yolov5_fine_tuning_dataset, "The returned yolov5_fine_tuning_dataset should have a fine_tuning_dataset_hash"
    assert returned_yolov5_fine_tuning_dataset['fine_tuning_dataset_hash'] in [
        one_yolov5_fine_tuning_dataset['fine_tuning_dataset_hash'] for one_yolov5_fine_tuning_dataset in
        dataset_creator.state[
            'yolov5_fine_tuning_datasets']], "The returned yolov5_fine_tuning_datasets should be in the state"
    assert len(dataset_creator.state['yolov5_fine_tuning_datasets']) == 1

    returned_yolov5_fine_tuning_dataset2 = dataset_creator.find_or_create_yolov5_dataset(assembly_configs,
                                                                                         sliced_dataset_configs)

    assert len(dataset_creator.state[
                   'yolov5_fine_tuning_datasets']) == 1, "There should still be only one yolov5_fine_tuning_datasets in the state"

    assembly_config2 = assembly_configs
    assembly_config2["dataset_config"][1]["train_split_rate"] = 0.2

    returned_yolov5_fine_tuning_dataset3 = dataset_creator.find_or_create_yolov5_dataset(assembly_config2,
                                                                                         sliced_dataset_configs)
    assert returned_yolov5_fine_tuning_dataset3['fine_tuning_dataset_hash'] in [
        one_yolov5_fine_tuning_dataset['fine_tuning_dataset_hash'] for one_yolov5_fine_tuning_dataset in
        dataset_creator.state[
            'yolov5_fine_tuning_datasets']], "The returned yolov5_fine_tuning_dataset should be in the state"
    assert len(
        dataset_creator.state[
            'yolov5_fine_tuning_datasets']) == 2, "There should be two yolov5_fine_tuning_datasets in the state"

    assert returned_yolov5_fine_tuning_dataset['data_yml_path'].exists()
    assert returned_yolov5_fine_tuning_dataset2['data_yml_path'].exists()
    assert returned_yolov5_fine_tuning_dataset3['data_yml_path'].exists()



def test_plausibility_of_image_counts(assembly_configs, sliced_dataset_configs):
    # Check counts in assembly
    dataset_creator = DatasetCreator()
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_train_dir' / 'tmp_train_state.pkl', ignore_errors=True)
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_eval_dir', ignore_errors=True)
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_dir', ignore_errors=True)
    dataset_creator = DatasetCreator()

    returned_assembly = dataset_creator.find_or_create_dataset_assembly(assembly_configs)

    with open(returned_assembly['data']['train']['path'].as_posix()) as json_file:
        number_of_train_images_in_assembly = len(json.load(json_file)['images'])
    with open(returned_assembly['data']['val']['path'].as_posix()) as json_file:
        number_of_val_images_in_assembly = len(json.load(json_file)['images'])
    assert number_of_train_images_in_assembly + number_of_val_images_in_assembly == 14, "There should be 14 images in the assembly. According to the Test data"
    assert number_of_train_images_in_assembly > number_of_val_images_in_assembly, "There should be more train images"

    # Check counts in sliced dataset and fine tuning dataset
    returned_fine_tuning_dataset = dataset_creator.find_or_create_fine_tuning_dataset(assembly_configs,
                                                                                      sliced_dataset_configs)

    with open(returned_fine_tuning_dataset['data']['train']['path'].as_posix()) as json_file:
        number_of_train_images_in_fine_tuning = len(json.load(json_file)['images'])
    with open(returned_fine_tuning_dataset['data']['val']['path'].as_posix()) as json_file:
        number_of_val_images_in_fine_tuning = len(json.load(json_file)['images'])
    assert number_of_train_images_in_fine_tuning > number_of_val_images_in_fine_tuning, "There should be more train images than val images in the fine tuning dataset"

    # Get Slices per image for each slice config
    slices_per_image = []
    for sliced_dataset in returned_fine_tuning_dataset['sliced_datasets']:
        with open(sliced_dataset['data']['val']['path'].as_posix()) as json_file:
            slices_per_image.append(len(json.load(json_file)['images']) / number_of_val_images_in_assembly)

    # Comparing and testing expected_image_count_in_train_fine_tuning with actual number_of_train_images_in_fine_tuning
    # only works if negative samples are not ignored
    expected_image_count_in_val_fine_tuning = sum(
        [sl_per_image * number_of_val_images_in_assembly for sl_per_image in slices_per_image])
    expected_image_count_in_train_fine_tuning = sum(
        [sl_per_image * number_of_train_images_in_assembly for sl_per_image in slices_per_image])

    assert number_of_train_images_in_fine_tuning == expected_image_count_in_train_fine_tuning
    assert number_of_val_images_in_fine_tuning == expected_image_count_in_val_fine_tuning

    types = ('.png', '.jpg')
    files_grabbed = []
    for suffix in types:
        files_grabbed.extend(returned_fine_tuning_dataset["data"]["train"]["path"].parent.glob('**/*'+ suffix))
    assert len(
        files_grabbed) == number_of_train_images_in_fine_tuning + number_of_val_images_in_fine_tuning, "There should be as many images in the folder fine tuning folder as in the train and val dataset combined"

    # Check counts in yolov dataset
    returned_yolov5_fine_tuning_dataset = dataset_creator.find_or_create_yolov5_dataset(assembly_configs,
                                                                                        sliced_dataset_configs)
    files_grabbed = []
    for files in types:
        files_grabbed.extend((returned_yolov5_fine_tuning_dataset["data_yml_path"].parent / "train").glob(files))
    assert len(
        files_grabbed) == number_of_train_images_in_fine_tuning, "There should be as many images in the folder yolov train folder as in the train dataset"
    files_grabbed = []
    for files in types:
        files_grabbed.extend((returned_yolov5_fine_tuning_dataset["data_yml_path"].parent / "val").glob(files))
    assert len(
        files_grabbed) == number_of_val_images_in_fine_tuning, "There should be as many images in the folder yolov val folder as in the val dataset"


def test_negatives_handling_plausability(assembly_configs, sliced_dataset_configs):
    dataset_creator = DatasetCreator()
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_train_dir' / 'tmp_train_state.pkl', ignore_errors=True)
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_eval_dir', ignore_errors=True)
    shutil.rmtree(dataset_creator.tmp_dir_path.parent / 'tmp_dir', ignore_errors=True)
    dataset_creator = DatasetCreator()

    returned_fine_tuning_dataset = dataset_creator.find_or_create_fine_tuning_dataset(assembly_configs,
                                                                                      sliced_dataset_configs)

    assert assembly_configs['ignore_negative_samples'] == False
    assembly_configs['ignore_negative_samples'] = True
    returned_fine_tuning_dataset_wo_negatives = dataset_creator.find_or_create_fine_tuning_dataset(assembly_configs,
                                                                                                   sliced_dataset_configs)

    with open(returned_fine_tuning_dataset['data']['train']['path'].as_posix()) as json_file:
        number_of_train_images_in_fine_tuning = len(json.load(json_file)['images'])
    with open(returned_fine_tuning_dataset['data']['val']['path'].as_posix()) as json_file:
        number_of_val_images_in_fine_tuning = len(json.load(json_file)['images'])

    with open(returned_fine_tuning_dataset_wo_negatives['data']['train']['path'].as_posix()) as json_file:
        number_of_train_images_in_fine_tuning_wo_neg = len(json.load(json_file)['images'])
    with open(returned_fine_tuning_dataset_wo_negatives['data']['val']['path'].as_posix()) as json_file:
        number_of_val_images_in_fine_tuning_wo_neg = len(json.load(json_file)['images'])

    assert number_of_train_images_in_fine_tuning + number_of_val_images_in_fine_tuning > number_of_train_images_in_fine_tuning_wo_neg + number_of_val_images_in_fine_tuning_wo_neg
