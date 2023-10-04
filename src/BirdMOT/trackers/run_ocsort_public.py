'''
    This script makes tracking over the results of existing
    tracking algorithms. Namely, we run OC-SORT over theirdetections.
    Output in such a way is not strictly accurate because
    building tracks from existing tracking results causes loss
    of detections (usually initializing tracks requires a few
    continuous observations which are not recorded in the output
    tracking results by other methods). But this quick adaptation
    can provide a rough idea about OC-SORT's performance on
    more datasets. For more strict study, we encourage to implement 
    a specific detector on the target dataset and then run OC-SORT 
    over the raw detection results.
    NOTE: this script is not for the reported tracking with public
    detection on MOT17/MOT20 which requires the detection filtering
    following previous practice. See an example from centertrack for
    example: https://github.com/xingyizhou/CenterTrack/blob/d3d52145b71cb9797da2bfb78f0f1e88b286c871/src/lib/utils/tracker.py#L83
'''

import os
import time
from pathlib import Path

import numpy as np
from loguru import logger

from OC_SORT.trackers.ocsort_tracker.ocsort import OCSort
from src.BirdMOT.data.mot_data import load_mot17_det
from src.BirdMOT.trackers.OC_SORT.tools.interpolation import dti, eval_mota


# def compare_dataframes(gts, ts):
#     accs = []
#     names = []
#     for k, tsacc in ts.items():
#         if k in gts:
#             logger.info('Comparing {}...'.format(k))
#             accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
#             names.append(k)
#         else:
#             logger.warning('No ground truth for {}, skipping.'.format(k))
#
#     return accs, names


@logger.catch
def oc_sort(seq_name: str, path: Path, results_folder: Path, args):
    os.makedirs(results_folder, exist_ok=True)

    tracker = OCSort(args['det_thresh'], iou_threshold=args['iou_thresh'], delta_t=args['delta_t'],
                     asso_func=args['asso_func'], inertia=args['inertia'])

    out_file = os.path.join(results_folder, "{}.txt".format(seq_name))
    out_file = open(out_file, 'w')

    total_time = 0
    total_frame = 0

    mot_data = load_mot17_det(path)

    min_frame = mot_data[:, 0].min()
    max_frame = mot_data[:, 0].max()

    for frame_ind in range(int(min_frame), int(max_frame) + 1):
        dets = mot_data[np.where(mot_data[:, 0] == frame_ind)][:, 2:6]
        categories = mot_data[np.where(mot_data[:, 0] == frame_ind)][:, 1]
        scores = mot_data[np.where(mot_data[:, 0] == frame_ind)][:, 6]

        assert (dets.shape[0] == categories.shape[0])
        t0 = time.time()
        online_targets = tracker.update_public(dets, categories, scores)
        t1 = time.time()
        total_frame += 1
        total_time += t1 - t0
        trk_num = online_targets.shape[0]
        boxes = online_targets[:, :4]
        ids = online_targets[:, 4]
        frame_counts = online_targets[:, 6]
        sorted_frame_counts = np.argsort(frame_counts)
        frame_counts = frame_counts[sorted_frame_counts]
        categories = online_targets[:, 5]
        categories = categories[sorted_frame_counts].tolist()
        categories = [categories[int(catid)] for catid in categories]
        boxes = boxes[sorted_frame_counts]
        ids = ids[sorted_frame_counts]
        for trk in range(trk_num):
            lag_frame = frame_counts[trk]
            if frame_ind < 2 * args['min_hits'] and lag_frame < 0:
                continue
            """
                NOTE: here we use the Head Padding (HP) strategy by default, disable the following
                lines to revert back to the default version of OC-SORT.
            """
            out_line = "{},{},{},{},{},{},{},-1,-1,-1\n".format(int(frame_ind + lag_frame), int(ids[trk]),
                                                                boxes[trk][0], boxes[trk][1],
                                                                boxes[trk][2] - boxes[trk][0],
                                                                boxes[trk][3] - boxes[trk][1], 1)
            out_file.write(out_line)

    print("Running over {} frames takes {}s. FPS={}".format(total_frame, total_time, total_frame / total_time))
    return


if __name__ == "__main__":
    # ASSO_FUNCS = {"iou": iou_batch,
    #               "giou": giou_batch,
    #               "ciou": ciou_batch,
    #               "diou": diou_batch,
    #               "ct_dist": ct_dist}

    args = {'det_thresh': 0.1, 'max_age': 20, 'min_hits': 2, 'iou_thresh': 0.033, 'delta_t': 2, 'asso_func': 'iou',
        # association cost calculation
        'inertia': 0.2, 'use_byte': True}

    det_path = Path('/home/jo/coding_projects/fids/BirdMOT/data/gt/mot_challenge/BirdMOT2023-train/MOT-1657059041_zoex_0664999_0665083/det/det.txt')
    save_path =Path('/home/jo/test')
    data_root= '/home/jo/test'

    oc_sort(seq_name='abc',
            path=det_path,
            results_folder=save_path, args=args)

    dti('/home/jo/test/', '/home/jo/test/int', n_min=30, n_dti=20)

    #ToDo: Use mot eval instead of this
    print('Before DTI: ')
    eval_mota(data_root, '/home/jo/test/abc.txt2')
    print('After DTI:')
    eval_mota(data_root, save_path)
