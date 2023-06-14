from pathlib import Path

import cv2
def save_frames_as_img(video_path: Path, out_path: Path, image_format: str='png'):
    """
    Save frames of a video as images
    Args:
        video_path: Path to video
        out_path: Path to output directory
        image_format: String of image format, e.g. 'png', 'jpg'

    Returns:

    """
    #assert type(video_path) == Path, f"video_path must be of type Path, but is {type(video_path)}"
    assert video_path.exists(), f"Video  {video_path} does not exist"

    vidcap = cv2.VideoCapture(video_path.as_posix())
    success, image = vidcap.read()
    count = 0
    while success:
        filename = out_path / f"{video_path.stem}-{count}.{image_format}"
        print(filename.as_posix())
        cv2.imwrite(filename.as_posix(), image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
