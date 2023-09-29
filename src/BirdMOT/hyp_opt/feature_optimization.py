import json

import ray
import cv2 as cv
from imgaug import BoundingBox, Keypoint
from argparse import ArgumentParser
from pathlib import Path
import cv2

from birdwatchpy.sequences import load_sequence_from_pickle

params = {# Parameters for Shi-Tomasi corner detection
    "name": "goodFeaturesToTrack",
    "feature_params": {"maxCorners": 3000000, "qualityLevel": 0.03, "minDistance": 5, "blockSize": 50}}

quality_levels = [0.01,0.02,0.03]
blockSizes =[10,70,120, 150]
minDistances=[15,20,30,40,50]

@ray.remote
def sparse_lk_cpu_test_task(args):
    result = {}
    params, img_ref, bbxs = args

    bb_list = [
        BoundingBox(bb[0][0], bb[0][1], bb[1][0], bb[1][1]).extend(all_sides=params['feature_params']['minDistance'])
        for bb in bbxs]

    img = ray.get(img_ref)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for quality_level in quality_levels:
        for blockSize in blockSizes:
            for minDistance in minDistances:
                if params["name"] == 'fast':
                    feature_detector = cv.FastFeatureDetector_create()
                elif params["name"] == 'goodFeaturesToTrack':
                    features = cv.goodFeaturesToTrack(img_gray, params['feature_params']['maxCorners'],
                                                      quality_level,
                                                      minDistance,
                                                      blockSize)

                hits = 0

                if features is None:
                    result.update({f"{quality_level}_{minDistance}_{blockSize}": {'hits': 0, 'total_corners': 0, 'bb_count': len(bb_list)}})
                else:
                    features = features.reshape([-1,2])
                    for bb in bb_list:
                        for feature in features:
                            if bb.contains(Keypoint(x=feature[0], y =feature[1])):
                                hits += 1
                                break

                    result.update({f"{quality_level}_{minDistance}_{blockSize}": {'hits': hits, 'total_corners': len(features), 'bb_count': len(bb_list)}})
    print(result)
    return result

if __name__ == "__main__":
    result_dict_list = [{key: value} for key, value in {'0.01_15_10': {'hits': 37877, 'total_corners': 103694595, 'bb_count': 37890, 'avg_corners': 6067.559684025746
  }, '0.01_20_10': {'hits': 37874, 'total_corners': 69064464, 'bb_count': 37890, 'avg_corners': 4041.2208308952604
  }, '0.01_30_10': {'hits': 37768, 'total_corners': 39440762, 'bb_count': 37890, 'avg_corners': 2307.8269163253362
  }, '0.01_40_10': {'hits': 37217, 'total_corners': 25586287, 'bb_count': 37890, 'avg_corners': 1497.1496196606201
  }, '0.01_50_10': {'hits': 36036, 'total_corners': 18121871, 'bb_count': 37890, 'avg_corners': 1060.378642480983
  }, '0.01_15_70': {'hits': 37877, 'total_corners': 103694595, 'bb_count': 37890, 'avg_corners': 6067.559684025746
  }, '0.01_20_70': {'hits': 37874, 'total_corners': 69064464, 'bb_count': 37890, 'avg_corners': 4041.2208308952604
  }, '0.01_30_70': {'hits': 37768, 'total_corners': 39440762, 'bb_count': 37890, 'avg_corners': 2307.8269163253362
  }, '0.01_40_70': {'hits': 37217, 'total_corners': 25586287, 'bb_count': 37890, 'avg_corners': 1497.1496196606201
  }, '0.01_50_70': {'hits': 36036, 'total_corners': 18121871, 'bb_count': 37890, 'avg_corners': 1060.378642480983
  }, '0.01_15_120': {'hits': 37877, 'total_corners': 103694595, 'bb_count': 37890, 'avg_corners': 6067.559684025746
  }, '0.01_20_120': {'hits': 37874, 'total_corners': 69064464, 'bb_count': 37890, 'avg_corners': 4041.2208308952604
  }, '0.01_30_120': {'hits': 37768, 'total_corners': 39440762, 'bb_count': 37890, 'avg_corners': 2307.8269163253362
  }, '0.01_40_120': {'hits': 37217, 'total_corners': 25586287, 'bb_count': 37890, 'avg_corners': 1497.1496196606201
  }, '0.01_50_120': {'hits': 36036, 'total_corners': 18121871, 'bb_count': 37890, 'avg_corners': 1060.378642480983
  }, '0.01_15_150': {'hits': 37877, 'total_corners': 103694595, 'bb_count': 37890, 'avg_corners': 6067.559684025746
  }, '0.01_20_150': {'hits': 37874, 'total_corners': 69064464, 'bb_count': 37890, 'avg_corners': 4041.2208308952604
  }, '0.01_30_150': {'hits': 37768, 'total_corners': 39440762, 'bb_count': 37890, 'avg_corners': 2307.8269163253362
  }, '0.01_40_150': {'hits': 37217, 'total_corners': 25586287, 'bb_count': 37890, 'avg_corners': 1497.1496196606201
  }, '0.01_50_150': {'hits': 36036, 'total_corners': 18121871, 'bb_count': 37890, 'avg_corners': 1060.378642480983
  }, '0.02_15_10': {'hits': 37833, 'total_corners': 59414607, 'bb_count': 37890, 'avg_corners': 3476.5715038033936
  }, '0.02_20_10': {'hits': 37831, 'total_corners': 41469950, 'bb_count': 37890, 'avg_corners': 2426.562317144529
  }, '0.02_30_10': {'hits': 37709, 'total_corners': 24941239, 'bb_count': 37890, 'avg_corners': 1459.405441778818
  }, '0.02_40_10': {'hits': 37130, 'total_corners': 16809662, 'bb_count': 37890, 'avg_corners': 983.5963721474546
  }, '0.02_50_10': {'hits': 35901, 'total_corners': 12249084, 'bb_count': 37890, 'avg_corners': 716.7398478642481
  }, '0.02_15_70': {'hits': 37833, 'total_corners': 59414607, 'bb_count': 37890, 'avg_corners': 3476.5715038033936
  }, '0.02_20_70': {'hits': 37831, 'total_corners': 41469950, 'bb_count': 37890, 'avg_corners': 2426.562317144529
  }, '0.02_30_70': {'hits': 37709, 'total_corners': 24941239, 'bb_count': 37890, 'avg_corners': 1459.405441778818
  }, '0.02_40_70': {'hits': 37130, 'total_corners': 16809662, 'bb_count': 37890, 'avg_corners': 983.5963721474546
  }, '0.02_50_70': {'hits': 35901, 'total_corners': 12249084, 'bb_count': 37890, 'avg_corners': 716.7398478642481
  }, '0.02_15_120': {'hits': 37833, 'total_corners': 59414607, 'bb_count': 37890, 'avg_corners': 3476.5715038033936
  }, '0.02_20_120': {'hits': 37831, 'total_corners': 41469950, 'bb_count': 37890, 'avg_corners': 2426.562317144529
  }, '0.02_30_120': {'hits': 37709, 'total_corners': 24941239, 'bb_count': 37890, 'avg_corners': 1459.405441778818
  }, '0.02_40_120': {'hits': 37130, 'total_corners': 16809662, 'bb_count': 37890, 'avg_corners': 983.5963721474546
  }, '0.02_50_120': {'hits': 35901, 'total_corners': 12249084, 'bb_count': 37890, 'avg_corners': 716.7398478642481
  }, '0.02_15_150': {'hits': 37833, 'total_corners': 59414607, 'bb_count': 37890, 'avg_corners': 3476.5715038033936
  }, '0.02_20_150': {'hits': 37831, 'total_corners': 41469950, 'bb_count': 37890, 'avg_corners': 2426.562317144529
  }, '0.02_30_150': {'hits': 37709, 'total_corners': 24941239, 'bb_count': 37890, 'avg_corners': 1459.405441778818
  }, '0.02_40_150': {'hits': 37130, 'total_corners': 16809662, 'bb_count': 37890, 'avg_corners': 983.5963721474546
  }, '0.02_50_150': {'hits': 35901, 'total_corners': 12249084, 'bb_count': 37890, 'avg_corners': 716.7398478642481
  }, '0.03_15_10': {'hits': 37768, 'total_corners': 39650193, 'bb_count': 37890, 'avg_corners': 2320.081509654769
  }, '0.03_20_10': {'hits': 37764, 'total_corners': 28462518, 'bb_count': 37890, 'avg_corners': 1665.4486834406086
  }, '0.03_30_10': {'hits': 37636, 'total_corners': 17676029, 'bb_count': 37890, 'avg_corners': 1034.2907548273845
  }, '0.03_40_10': {'hits': 37043, 'total_corners': 12195850, 'bb_count': 37890, 'avg_corners': 713.6249268578116
  }, '0.03_50_10': {'hits': 35804, 'total_corners': 9046691, 'bb_count': 37890, 'avg_corners': 529.3558221181978
  }, '0.03_15_70': {'hits': 37768, 'total_corners': 39650193, 'bb_count': 37890, 'avg_corners': 2320.081509654769
  }, '0.03_20_70': {'hits': 37764, 'total_corners': 28462518, 'bb_count': 37890, 'avg_corners': 1665.4486834406086
  }, '0.03_30_70': {'hits': 37636, 'total_corners': 17676029, 'bb_count': 37890, 'avg_corners': 1034.2907548273845
  }, '0.03_40_70': {'hits': 37043, 'total_corners': 12195850, 'bb_count': 37890, 'avg_corners': 713.6249268578116
  }, '0.03_50_70': {'hits': 35804, 'total_corners': 9046691, 'bb_count': 37890, 'avg_corners': 529.3558221181978
  }, '0.03_15_120': {'hits': 37768, 'total_corners': 39650193, 'bb_count': 37890, 'avg_corners': 2320.081509654769
  }, '0.03_20_120': {'hits': 37764, 'total_corners': 28462518, 'bb_count': 37890, 'avg_corners': 1665.4486834406086
  }, '0.03_30_120': {'hits': 37636, 'total_corners': 17676029, 'bb_count': 37890, 'avg_corners': 1034.2907548273845
  }, '0.03_40_120': {'hits': 37043, 'total_corners': 12195850, 'bb_count': 37890, 'avg_corners': 713.6249268578116
  }, '0.03_50_120': {'hits': 35804, 'total_corners': 9046691, 'bb_count': 37890, 'avg_corners': 529.3558221181978
  }, '0.03_15_150': {'hits': 37768, 'total_corners': 39650193, 'bb_count': 37890, 'avg_corners': 2320.081509654769
  }, '0.03_20_150': {'hits': 37764, 'total_corners': 28462518, 'bb_count': 37890, 'avg_corners': 1665.4486834406086
  }, '0.03_30_150': {'hits': 37636, 'total_corners': 17676029, 'bb_count': 37890, 'avg_corners': 1034.2907548273845
  }, '0.03_40_150': {'hits': 37043, 'total_corners': 12195850, 'bb_count': 37890, 'avg_corners': 713.6249268578116
  }, '0.03_50_150': {'hits': 35804, 'total_corners': 9046691, 'bb_count': 37890, 'avg_corners': 529.3558221181978
  }}.items()]
    sorted_result_dict_list =sorted(result_dict_list, key=lambda k: (list(k.values())[0]['hits'],-list(k.values())[0]['avg_corners']), reverse=True)
    print(json.dumps(sorted_result_dict_list, indent=4))


from birdwatchpy.bird_flight_analysis import BirdFlightData

if __name__ != "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--recursive", action='store_true', help="Recursively load flights")
    parser.add_argument("--out", help="Output Path")
    parser.add_argument("path", help="Input Path")

    running_tasks = []
    final_results = {}
    result_counter=0

    args = parser.parse_args()
    if args.recursive:
        for path in Path(args.path).iterdir():
            if path.is_dir():
                print(path)
                if len(list(path.glob('single_bird*.json'))) > 0:
                    sequence_file_path = path / f"{path.name}.sequence"
                    try:
                        assert sequence_file_path.is_file()
                    except:
                        continue
                    sequence = load_sequence_from_pickle(sequence_file_path)

                    set_of_image_paths = set([])
                    for flight_file in Path(path).glob('single_bird*.json'):
                        bird = BirdFlightData.load(flight_file.as_posix())
                        set_of_image_paths = set_of_image_paths.union(set(bird.frame_numbers))

                    list_of_img_paths = list(set_of_image_paths)
                    list_of_img_paths.sort()

                    for frame_num in set_of_image_paths:
                        img_path = path / "images" / f"{path.name}-{frame_num}.png"
                        try:
                            assert img_path.is_file()
                        except:
                            continue
                        img_ref = ray.put(cv.imread(img_path.as_posix()))

                        bbxs = [det.as_dict()['bb'] for det in sequence.frames[frame_num].detection if
                                det.confidence > 0.15]

                        if len(bbxs)==0:
                            continue

                        args_ref = ray.put((params, img_ref, bbxs))

                        running_tasks.append(sparse_lk_cpu_test_task.remote(args_ref))

                        if len(running_tasks)>30:
                            done_tasks, running_tasks = ray.wait(running_tasks, num_returns=10)
                            for done_task_ref in done_tasks:
                                task_res = ray.get(done_task_ref)
                                result_counter += 1

                                for key, values in task_res.items():
                                    if not key in final_results:
                                        final_results.update({key: values})
                                    else:
                                        final_results[key]['hits'] = final_results[key]['hits'] + values['hits']
                                        final_results[key]['total_corners'] = final_results[key]['total_corners'] + values['total_corners']
                                        final_results[key]['bb_count'] = final_results[key]['bb_count'] + values['bb_count']
                                        final_results[key]['avg_corners'] = final_results[key]['total_corners'] / result_counter
                            print(result_counter)
                            print(final_results)

    ray.wait(running_tasks, num_returns = len(running_tasks))
    print(final_results)
    result_dict_list = [{key: value} for key, value in final_results]
    sorted_result_dict_list =sorted(result_dict_list, key=lambda k: (list(k.values())[0]['hits'],-list(k.values())[0]['avg_corners']), reverse=True)
    print(json.dumps(sorted_result_dict_list, indent=4))