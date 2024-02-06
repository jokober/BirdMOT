BirdMOT Detection
========
This repository contains the code used to run experiments on bird detection for my master thesis. 

.. image:: http://unmaintained.tech/badge.svg
  :target: http://unmaintained.tech
  :alt: No Maintenance Intended

*Note: This code is not intended to be maintained.*



## Install
### Install using Conda
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

### Install using Poetry
```console
poetry install
```
### Set Environment Variables
Set following environment variable to the path of your local data folder. This is where your dataset, sliced datasets, models and more will be stored.
```bash
export BirdMOT_DATA_PATH="/path/to/local_data"
echo $BirdMOT_DATA_PATH
```



## Getting started
### Create Local Data Folder
Create a folder structure as follows
```markdown
├── local_data
|    ├── dataset
|    │   ├── coco_files
|    │   ├── images
|    ├── configs
|    │   ├── categories
|    │   ├── dataset_assembly
|    │   ├── experiments
```
For further information please check out the 'local_data' folder in the 'fixtures' folder under 'tests'.

### Create a config files
Please check out example configs in the 'local_data' folder in the 'fixtures' folder under 'tests'.
### MLFlow
Run a mlflow server and set the tracking uri to your mlflow server:
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
echo $MLFLOW_TRACKING_URI
```