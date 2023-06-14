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
``

## Training
Remember to delete old sliced_datasets if data has changed.
Careful with creating yolo files several times as they will be copied several times leading to more and more files ...