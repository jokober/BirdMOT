import trackeval

eval_config = {'USE_PARALLEL': False, 'NUM_PARALLEL_CORES': 8, 'BREAK_ON_ERROR': True, 'RETURN_ON_ERROR': False, 'LOG_ON_ERROR': '/home/jo/coding_projects/TrackEval/error_log.txt', 'PRINT_RESULTS': True, 'PRINT_ONLY_COMBINED': False, 'PRINT_CONFIG': False, 'TIME_PROGRESS': True, 'DISPLAY_LESS_PROGRESS': False, 'OUTPUT_SUMMARY': True, 'OUTPUT_EMPTY_CLASSES': True, 'OUTPUT_DETAILED': True, 'PLOT_CURVES': True}
dataset_config = {'PRINT_CONFIG': False, 'GT_FOLDER': '/home/jo/coding_projects/BirdMOT/data/gt/mot_challenge/', 'TRACKERS_FOLDER': '/home/jo/coding_projects/BirdMOT/data/trackers/mot_challenge/', 'OUTPUT_FOLDER': None, 'TRACKERS_TO_EVAL': None, 'CLASSES_TO_EVAL': ['pedestrian'], 'BENCHMARK': 'MOT17', 'SPLIT_TO_EVAL': 'train', 'INPUT_AS_ZIP': False, 'DO_PREPROC': True, 'TRACKER_SUB_FOLDER': 'data', 'OUTPUT_SUB_FOLDER': '', 'TRACKER_DISPLAY_NAMES': None, 'SEQMAP_FOLDER': None, 'SEQMAP_FILE': '/home/jo/coding_projects/BirdMOT/data/gt/mot_challenge/seqmaps/MOT17-train.txt', 'SEQ_INFO': None, 'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt', 'SKIP_SPLIT_FOL': False}
metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}

# eval_config = {'USE_PARALLEL': False, 'NUM_PARALLEL_CORES': 8, 'BREAK_ON_ERROR': True, 'RETURN_ON_ERROR': False, 'LOG_ON_ERROR': '/home/jo/coding_projects/TrackEval/error_log.txt', 'PRINT_RESULTS': True, 'PRINT_ONLY_COMBINED': False, 'PRINT_CONFIG': False, 'TIME_PROGRESS': True, 'DISPLAY_LESS_PROGRESS': False, 'OUTPUT_SUMMARY': True, 'OUTPUT_EMPTY_CLASSES': True, 'OUTPUT_DETAILED': True, 'PLOT_CURVES': True}
# dataset_config = {'PRINT_CONFIG': False, 'GT_FOLDER': '/home/jo/coding_projects/BirdMOT/data/gt/mot_challenge/', 'TRACKERS_FOLDER': '/home/jo/coding_projects/BirdMOT/data/trackers/mot_challenge/', 'OUTPUT_FOLDER': None, 'TRACKERS_TO_EVAL': None, 'CLASSES_TO_EVAL': ['pedestrian'], 'BENCHMARK': 'BirdMOT2023', 'SPLIT_TO_EVAL': 'train', 'INPUT_AS_ZIP': False, 'DO_PREPROC': True, 'TRACKER_SUB_FOLDER': 'data', 'OUTPUT_SUB_FOLDER': '', 'TRACKER_DISPLAY_NAMES': None, 'SEQMAP_FOLDER': None, 'SEQMAP_FILE': '/home/jo/coding_projects/BirdMOT/data/gt/mot_challenge/seqmaps/BirdMOT2023_all.txt', 'SEQ_INFO': None, 'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt', 'SKIP_SPLIT_FOL': False}
# metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}


# Run code
evaluator = trackeval.Evaluator(eval_config)
dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
metrics_list = []
for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
    if metric.get_name() in metrics_config['METRICS']:
        metrics_list.append(metric(metrics_config))
if len(metrics_list) == 0:
    raise Exception('No metrics selected for evaluation')
evaluator.evaluate(dataset_list, metrics_list)
