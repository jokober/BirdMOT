# Install
## Install using conda and poetry
Conda
```bash
conda env create -f environment.yml
``` 
Activate Conda Environment
```bash
 conda activate BirdMOT
``

## Install using docker
```bash
docker compose  -f docker-compose.yml build
docker compose -f docker-compose.yml run birdmot-gpu
```
install fiftyone:
```console
pip install fiftyone
```
Some operating systems like ubuntu 22.ß4 need alternative builds from fiftyone: ToDo. Link to fiftyone alternative builds.
For example on Ubtuntu 22.04 you need to install fiftyone like this:
```
pip install fiftyone
pip install fiftyone-db-ubuntu2204
```

## Environment Variables
Set following environment variable to the path of your local data folder. This is where your dataset, sliced datasets, models and more will be stored.
```bash
export BirdMOT_DATA_PATH="/home/jo/fids/birdmot_local_data"
echo $BirdMOT_DATA_PATH
```

## Local Data Folder Structure
```markdown
├── local_data
|    ├── dataset
|    │   ├── coco_files
|    │   ├── annotations
|    │   ├── images
|    ├── sliced_datasets
|    │   ├── {height}_{width}_{overlap_height_ratio}_{overlap_width_ratio}_{min_area_ratio} 
|    │   │   ├── coco_files
|    │   │   │   ├── annotations
|    │   │   │   ├── images
|    │   │   │   ├── results
|    │   │   ├── yolov5_files
|    ├── models
```
Poetry
```bash
poetry lock -vv && poetry install
```

## Assemble Dataset
```bash
python -m scripts.assemble_dataset first_mix --coco_annotations_dir /media/data/BirdMOT/local_data/dataset/coco_files --output_path /media/data/BirdMOT/local_data/dataset/coco_files/dataset_assemblies --categories_path /home/fids/fids/BirdMOT/tests/fixtures/coco_fixtures/BirdMOT_categories_three_classes.json --config /home/fids/fids/BirdMOT/experiments/dataset_assembly1.json
```

## Tasks
- [ ] Get Dataset Stats
- [ ] Save models and data
- [ ] implement tune hyperparameters for sliced yolo training
- [ ] run hyperparmater for sliced yolo training a little
- [ ] implement fps metric
- [ ] implement tune hyperparameters for sahi prediction
- [ ] run hyperparmater for sahi prediction a little
- [ ] implement OC_Sort tracking
- [ ] implement tracking hyperparameter tuning
- [ ] run tracking hyperparameter tuning a little
- [ ] implement augmentation
- [ ] long run of hyperparameter tuning

## Features
- MLFlow Integration

### MLFlow
Set the tracking uri to your mlflow server:
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
echo $MLFLOW_TRACKING_URI
```
``

## Training
Remember to delete old sliced_datasets if data has changed.
Careful with creating yolo files several times as they will be copied several times leading to more and more files ...