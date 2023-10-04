from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

args = {
   'track_thresh':0.2,
   'track_buffer':0.2,
   'det_thresh':0.1,
}

"""
s (so x1, y1, x2, y2) and the fifth column is the confidence score of the bbox.
 0.5
"""

tracker = BYTETracker(args)
for image in images:
   dets = detector(image)
   online_targets = tracker.update(dets, info_imgs, img_size)