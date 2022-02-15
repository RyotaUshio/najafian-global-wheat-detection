import _make_dataset as mkdata
import utils

from pathlib import Path
import sys
import argparse
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('root', type=Path)
    parser.add_argument('globs', nargs='+')
    parser.add_argument('n', type=int)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    return args

def main(
        *, # root, 
        globs, n, output, verbose
    ):
    root = Path('.')
    video_pathes = []
    for glob in globs:
        video_pathes += list(root.glob(glob))
    videos = [mkdata.Video(path) for path in video_pathes]
    n_video = len(videos)
    if n_video == 0:
        print(f'No video was found for globs {globs}.')
        sys.exit(1)
    if not output.exists():
        print(f'Output directory {output} was not found.')
        sys.exit(1)
    n_image_per_video = n//n_video + 1

    logger = utils.make_logger(verbose=verbose)
    logger(f'{n_video} video{"s" if n_video > 1 else ""} found')
    logger(f'{n_image_per_video} frames will be sampled for each video')

    with tqdm(total=n) as pbar:
        n_image = 0
        try:
            for video in videos:
                with video.open():
                    for i in range(n_image_per_video):
                        frame = video.random_read()
                        utils.save_image(frame, output / f'{video.stem}_{i}.jpg')
                        n_image += 1
                        pbar.update(1)
                        if n_image >= n:
                            raise StopIteration
        except StopIteration:
            pass


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
