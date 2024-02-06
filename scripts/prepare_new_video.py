from argparse import ArgumentParser
from pathlib import Path

from BirdMOT.helper.cv import save_frames_as_img

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--image_format", type=str, required=False)
    args = parser.parse_args()

    save_frames_as_img(video_path=Path(args.video_path),
                       out_path=Path(args.out_path),
                       image_format=args.image_format
                       )
