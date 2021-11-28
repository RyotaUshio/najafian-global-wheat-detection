#!/usr/bin/env/python3
import argparse
import pathlib
import os
import sys
import errno
import time
import re
from collections import defaultdict, namedtuple
from tqdm import tqdm
import torch
import numpy as np

import _make_dataset as mkdata
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', 
        help='root directory of the project (optional). if given, all other pathes are assumed to be relative to the root.', 
        type=pathlib.Path
    )
    parser.add_argument(
        '--mask', 
        help='input directory where VOC mask files named [image_name].npy are placed', 
        type=pathlib.Path, 
        nargs='?',
        default='mask'
    )
    parser.add_argument(
        '--rep', 
        help='[required] input directory where representative image files are placed. Note that a filename must have the format of [video name]_[%%M%%S] where %%M and %%S denote minute and second as decimal numbers, respectedly', 
        type=pathlib.Path, 
        nargs='?',
        default='rep'
    )
    parser.add_argument(
        '--field',
        help='[required] input directory where video files of fields are placed', 
        type=pathlib.Path, 
        nargs='?',
        default='field'
    )
    parser.add_argument(
        '--back', '--background', 
        help='[required] input directory where background video files are placed', 
        type=pathlib.Path, 
        nargs='?',
        default='back'
    )
    parser.add_argument(
        '-o', '--output', 
        help='[required] output directory where generated dataset will be exported. Note that two directories will be made under the output_dir, namely "images" and "labels"', 
        type=pathlib.Path, 
        required=True
    )
    parser.add_argument(
        '-d', '--domain-adaptation', 
        help='output directory where images & labels for the first step of domain adaptation are placed (generated only if this flag is passed)',
        type=pathlib.Path
    )
    parser.add_argument(
        '--labels', 
        help='[required] path to the labels file', 
        type=pathlib.Path, 
        nargs='?',
        default='labels.txt'
    )
    parser.add_argument(
        '-n', '--n-sample', 
        help='[required] number of samples of the dataset that will be generated synthetically', 
        type=int, 
        required=True
    )
    parser.add_argument(
        '-p', '--probability', '--prob', 
        help='probability that a foreground object is chosen from each class when synthesizving dataset. This must have the same length as number of classes. Defaults to [1/n_class, 1/n_class, ...]', 
        nargs='*', 
        type=float
    )
    parser.add_argument(
        '-e', '--extension', 
        help='extension(s) of image files', 
        nargs='?', 
        default='.png'
    )
    parser.add_argument(
        '-v', '--verbose', 
        help='When specified, display the progress and time remaining', 
        action='store_true'
    )
    parser.add_argument(
        '-b', '--bbox', 
        help='When specified, images with calculated bounding boxes are also saved', 
        action='store_true'
    )
    parser.add_argument(
        '-c', '--cuda', 
        help='When specified, try to use cuda if available', 
        action='store_true'
    )
    parser.add_argument(
        '-r', '--resume',
        help='When specified, resume from where you left off',
        action='store_true'
    )

    args = parser.parse_args()
    if not args.extension.startswith('.'):
        args.extension = '.' + args.extension
    if args.root is not None:
        args_dict = vars(args)
        for key in ['mask', 'rep', 'field', 'back', 'output', 'domain_adaptation', 'labels']:
            if args_dict[key] is not None:
                args_dict[key] = args.root / args_dict[key]
            if key in ['mask', 'rep', 'field', 'back', 'labels']:
                if not args_dict[key].exists():
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(args_dict[key]))
            elif args_dict[key] is not None:
                if args_dict[key].exists() and not args.resume:
                    while True:
                        ans = input(f'{args_dict[key]} already exists. Are you sure to overwrite? (y/n) --> ')
                        if ans in ['y', 'yes', 'n', 'no']:
                            break
                    if ans.startswith('n'):
                        sys.exit(1)
                elif not args_dict[key].exists() and args.resume():
                    raise RuntimeError(f'cannot resume working on {args_dict[key]}: no such directory')

    return args

def make_path(args):
    pathes = namedtuple(
        'pathes',
        ['mask_dir', 'rep_dir', 'field_dir', 'back_dir', 
        'output_dir', 'output_images_dir', 'output_labels_dir', 'output_labeled_images_dir',
        'output_domain_adaptation_dir', 'output_domain_adaptation_images_dir', 'output_domain_adaptation_labels_dir', 'output_domain_adaptation_labeled_images_dir'],
        defaults=[None, None, None, None, None]
    )

    p = pathes(
        mask_dir=args.mask,
        rep_dir=args.rep,
        field_dir=args.field,
        back_dir=args.back,
        output_dir=args.output,
        output_images_dir=args.output / 'images/all',
        output_labels_dir=args.output / 'labels/all'
    )       
    p.output_images_dir.mkdir(parents=True, exist_ok=True)
    p.output_labels_dir.mkdir(parents=True, exist_ok=True)

    if args.bbox:
        p = p._replace(output_labeled_images_dir=p.output_dir / 'labeled_images')
        p.output_labeled_images_dir.mkdir(exist_ok=True)

    if args.domain_adaptation is not None:
        p = p._replace(output_domain_adaptation_dir=args.domain_adaptation)
        p = p._replace(output_domain_adaptation_images_dir=p.output_domain_adaptation_dir / 'images/all')
        p = p._replace(output_domain_adaptation_labels_dir=p.output_domain_adaptation_dir / 'labels/all')
        p.output_domain_adaptation_images_dir.mkdir(parents=True, exist_ok=True)
        p.output_domain_adaptation_labels_dir.mkdir(parents=True, exist_ok=True)
        if args.bbox:
            p = p._replace(output_domain_adaptation_labeled_images_dir=p.output_domain_adaptation_dir / 'labeled_images')
            p.output_domain_adaptation_labeled_images_dir.mkdir(parents=True, exist_ok=True)
            path = p.output_domain_adaptation_labeled_images_dir

    return p

def make_division(n_sample, *args):
    """calculate the optimal division of n_sample
    n_sample: total number of samples to be divided
    *args: iterables
    """
    lens = [len(arg) for arg in args]
    n_division = np.prod(lens)
    q = n_sample // n_division
    r = n_sample % n_division
    divisions = np.full(lens, q)
    for i in range(r):
        divisions.ravel()[i] += 1
    return divisions

def get_last_indices(pathes):
    counts = defaultdict(lambda: defaultdict(int))
    reps = list(set([rep.stem[:-5] for rep in pathes.rep_dir.glob('*.png')]))
    for rep in reps:
        for back in pathes.back_dir.iterdir():
            head = f'synth_back_{back.stem}_field_{rep}_'
            key = lambda p: int(re.match(head + r'(\d+)', p.stem).groups()[0])
            try:
                counts[back.stem][rep] = max(map(key, pathes.output_images_dir.glob(head + '*')))
            except ValueError:
                continue
    return {k: dict(v) for k, v in counts.items()}


def main():
    args = parse_args()
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    logger = utils.make_logger(verbose=args.verbose)
    
    logger(f'using {device} device')
    logger('making output directories...', end='')
    p = make_path(args)
    logger('done')

    logger('getting class information...', end='')
    classes = utils.get_classes(args.labels)
    n_class = len(classes)
    logger('done')

    if args.resume:
        logger('searching for where you left off...', end='')
        last_indices = get_last_indices(p)
        logger('done')

    # class probabilities
    if args.probability:
        assert len(args.probability) == n_class, 'n_probability must be equal to n_class'
        prob = np.array(args.probability)
        prob /= prob.sum()
    else:
        prob = None

    # generate objects of FieldVideo, RepImage, ForegroundObject, BackgroundVideo
    logger('constructing asset objects...', end='')
    field_videos = []
    for p_video in p.field_dir.glob('*.mp4'):
        video = mkdata.FieldVideo(
            p_video=p_video,
            p_rep_images=p.rep_dir,
            p_masks=p.mask_dir,
            rep_image_extension=args.extension,
            classes=classes
        )
        field_videos.append(video)

    back_videos = []
    for p_video in p.back_dir.glob('*.mp4'):
        video = mkdata.BackgroundVideo(p_video=p_video)
        back_videos.append(video)        
    logger('done')

    # simulate datasets for each pair of (a background video, set of foregound objects in a video)
    logger('synthesizing dataset...')

    field_videos_with_rep_images = [video for video in field_videos if video.rep_images]
    division = make_division(
        args.n_sample, 
        back_videos, 
        field_videos_with_rep_images
    ) # make optimal divisions
    try:
        with tqdm(total=args.n_sample) as pbar:
            n_sample_generated = 0
            for i, back_video in enumerate(back_videos):
                with back_video.open():
                    for j, field_video in enumerate(field_videos_with_rep_images):
                        if args.resume:
                            try:
                                last_index = last_indices[back_video.stem][field_video.stem]
                            except KeyError:
                                last_index = -1
                            start_index = last_index + 1
                            if start_index > division[i, j]:
                                raise ValueError(f'it seems like more than {args.n_sample} samples have already been generated')
                            n_sample_generated += start_index
                            pbar.update(start_index)
                        else:
                            start_index = 0
                        for index in range(start_index, division[i, j]):
                            mkdata.make_fomposite_image(
                                field_video=field_video, 
                                back_video=back_video, 
                                pathes=p, 
                                prob=prob,
                                args=args, 
                                device=device,
                                index=index
                            )
                            time.sleep(0.3)
                            n_sample_generated += 1
                            pbar.update(1)
                            # if n_sample_generated >= args.n_sample: # これいらないのでは？
                            #     raise StopIteration
    except KeyboardInterrupt:
        logger('interrupted by keyboard')
    except StopIteration:
        logger('done')

    if args.domain_adaptation is not None:
        logger('generating images & labels for the first step of domain adaptation...')
        n_rep_image = sum([len(video.rep_images) for video in field_videos])
        try:
            with tqdm(total=n_rep_image * 360) as pbar:
                for field_video in field_videos:
                    for rep_image in field_video.rep_images:
                        mkdata.make_rotated_rep_image(rep_image, pathes=p, bbox=args.bbox, pbar=pbar)
        except KeyboardInterrupt:
            logger('interrupted by keyboard')
        else:
            logger('done')
    
if __name__ == '__main__':
    main()
