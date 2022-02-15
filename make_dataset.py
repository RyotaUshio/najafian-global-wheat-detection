#!/usr/bin/env/python3
import argparse
import pathlib
import os
import sys
import errno
import time
import re
from collections import defaultdict, namedtuple
from tqdm.auto import tqdm
import torch
import numpy as np

import _make_dataset as mkdata
import utils
import label_editors

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'task',
        help='what kind of dataset to make. "composite": dataset of composite images / "rotate": dataset of rotated representative images',
    )
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
    # parser.add_argument(
    #     '-d', '--domain-adaptation', 
    #     help='output directory where images & labels for the first step of domain adaptation are placed (generated only if this flag is passed)',
    #     type=pathlib.Path
    # )
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
        # required=True
    )
    parser.add_argument(
        '-p', '--probability', '--prob', 
        help='probability that a foreground object is chosen from each class when synthesizving dataset. This must have the same length as number of classes. If not specified, each class will be chosen following the distribution observed in the original representative images', 
        nargs='*', 
        type=float
    )
    parser.add_argument(
        '-s', '--suffix', 
        help='suffix of output image files',
        nargs='?', 
        default='.jpeg'
    )
    parser.add_argument(
        '-v', '--verbose', 
        help='when specified, display the progress and time remaining', 
        action='store_true'
    )
    parser.add_argument(
        '-b', '--bbox', 
        help='when specified, images with calculated bounding boxes are also saved', 
        action='store_true'
    )
    parser.add_argument(
        '--label_editor', 
        help='[advanced] name of a function defined in label_editors.py'
    )
    # parser.add_argument(
    #     '-c', '--cuda', 
    #     help='[deprecated] when specified, try to use cuda if available', 
    #     action='store_true'
    # )
    parser.add_argument(
        '-r', '--resume',
        help='when specified, resume from where you left off',
        action='store_true'
    )
    parser.add_argument(
        '-a', '--augment', '--augment-intensity',
        help='(floating point number between 0.0 and 1.0) intensity of data augmentation applied to foreground and background images',
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--scale-jitter',
        help='when specified, large scale jittering will be applied before copy-paste', 
        action='store_true'
    )
    parser.add_argument(
        '-m', '--min-area',
        help=('minimum area required for each bounding box in the domain adaptation dataset\n'
        'An area is calculated as width x height, where width and height are an float in range (0, 1]'),
        type=float
    )
    parser.add_argument(
        '-f', '--force',
        help='if specified, existing output directory will be overwritten without asking',
        action='store_true'
    )

    args = parser.parse_args()
    if args.task == 'rotate':
        if args.resume:
            raise NotImplementedError('resume mode for the task "rotate" is not implemented yet')
        if args.probability:
            raise ValueError('an invalid option for the task "rotate": probablity')
        if args.n_sample is not None:
            raise ValueError('an invalid option for the task "rotate": n_sample')
        if args.min_area is not None and (args.min_area < 0.0 or 1.0 < args.min_area):
            raise ValueError(f'min_area must be between 0.0 and 1.0, got {args.min_area}')
    elif args.task == 'composite':
        if args.n_sample is None:
            raise ValueError('-n / --n-sample is required for the task "composite"')
        if args.augment < 0.0 or 1.0 < args.augment:
            raise ValueError(f'augment must be between 0.0 and 1.0, got {args.augment}')
        if args.min_area is not None:
            raise ValueError('an invalid option for the task "composite": min_area')
    else:
        raise ValueError(f'invalid task is given: {args.task}')

    args.suffix = utils.with_dot(args.suffix)
    args_dict = vars(args)
    keys = [
        'mask', 'rep', 'field', 'back', 'output', 
        # 'domain_adaptation', 
        'labels'
    ]
    if args.root is not None: # prepend args.root to each path
        for key in keys:
            if args_dict[key] is not None:
                args_dict[key] = args.root / args_dict[key]
    for key in keys:
        if key in ['mask', 'rep', 'field', 'back', 'labels']: # existence check
            if not args_dict[key].exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(args_dict[key]))
        # elif args_dict[key] is not None: # args.domain_adaptation might be None
        #     if args_dict[key].exists() and not args.resume:
        elif not args.force and args_dict[key].exists() and not args.resume:
            while True:
                ans = input(f'{args_dict[key]} already exists. Are you sure to overwrite? (y/n) --> ')
                if ans in ['y', 'yes', 'n', 'no']:
                    break
            if ans.startswith('n'):
                sys.exit(1)
        elif not args_dict[key].exists() and args.resume:
            raise RuntimeError(f'cannot resume working on {args_dict[key]}: no such directory')
    
    return args

def make_path(args):
    pathes = namedtuple(
        'pathes',
        ['mask_dir', 'rep_dir', 'field_dir', 'back_dir', 
        'output_dir', 'output_images_dir', 'output_labels_dir', 'output_masks_dir', 'output_labeled_images_dir'],
        # 'output_domain_adaptation_dir', 'output_domain_adaptation_images_dir', 'output_domain_adaptation_labels_dir', 'output_domain_adaptation_labeled_images_dir'],
        # defaults=[None, None, None, None, None]
        defaults=[None]
    )

    p = pathes(
        mask_dir=args.mask,
        rep_dir=args.rep,
        field_dir=args.field,
        back_dir=args.back,
        output_dir=args.output,
        output_images_dir=args.output / 'images/all',
        output_labels_dir=args.output / 'labels/all',
        output_masks_dir=args.output / 'masks/all'
    )       
    p.output_images_dir.mkdir(parents=True, exist_ok=True)
    p.output_labels_dir.mkdir(parents=True, exist_ok=True)
    p.output_masks_dir.mkdir(parents=True, exist_ok=True)

    if args.bbox:
        p = p._replace(output_labeled_images_dir=p.output_dir / 'labeled_images')
        p.output_labeled_images_dir.mkdir(exist_ok=True)

    return p

# def make_division(n_sample, *args):
#     """calculate the optimal division of n_sample
#     n_sample: total number of samples to be divided
#     *args: iterables
#     """
#     lens = [len(arg) for arg in args]
#     n_division = np.prod(lens)
#     q = n_sample // n_division
#     r = n_sample % n_division
#     divisions = np.full(lens, q)
#     for i in range(r):
#         divisions.ravel()[i] += 1
#     return divisions

def get_start_indices(pathes):
    start_indices = defaultdict(int)
    for back in pathes.back_dir.iterdir():
        head = f'composite_back_{back.stem}_'
        key = lambda p: int(re.match(head + r'(\d+)', p.stem).groups()[0])
        try:
            start_indices[back.stem] = max(map(key, pathes.output_images_dir.glob(head + '*'))) + 1 # start_index = last_index + 1
        except ValueError:
            continue
    return start_indices

# class NoJobToDo(Exception):
#     pass

def make_field_videos(*, classes, p_field_dir, p_rep_dir, p_mask_dir, label_editor=None):
    p_field_dir = pathlib.Path(p_field_dir)
    p_rep_dir = pathlib.Path(p_rep_dir)
    p_mask_dir = pathlib.Path(p_mask_dir)
    field_videos = []
    for suffix in utils.VID_FORMATS:
        suffix = utils.with_dot(suffix)
        for p_video in p_field_dir.glob(f'*{suffix}'):
            video = mkdata.FieldVideo(
                p_video=p_video,
                p_rep_images=p_rep_dir,
                p_masks=p_mask_dir,
                classes=classes,
                label_editor=label_editor
            )
            field_videos.append(video)
    return field_videos

def make_back_videos(*, p_back_dir):
    back_videos = []
    for suffix in utils.VID_FORMATS:
        suffix = utils.with_dot(suffix)
        for p_video in p_back_dir.glob(f'*{suffix}'):
            video = mkdata.BackgroundVideo(p_video=p_video)
            back_videos.append(video)
    return back_videos

def make_database(*, field_videos):
    database = mkdata.ObjectDatabase(classes=field_videos[0].classes)
    for field_video in field_videos:
        database.add(field_video)
    return database

def main():
    args = parse_args()
    # device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    logger = utils.make_logger(verbose=args.verbose)
    
    # logger(f'using {device} device')
    logger('making output directories...', end='')
    p = make_path(args)
    logger('done')

    logger('getting class information...', end='')
    classes = utils.get_classes(args.labels)
    n_class = len(classes)
    logger('done')

    # generate objects of FieldVideo, RepImage, ForegroundObject, BackgroundVideo, ObjectDatabase
    logger('constructing asset objects...', end='')

    # FieldVideo, RepImage & ForegroundObject
    label_editor = getattr(label_editors, args.label_editor) if args.label_editor is not None else None
    field_videos = make_field_videos(classes=classes, p_field_dir=p.field_dir, p_rep_dir=p.rep_dir, p_mask_dir=p.mask_dir, label_editor=label_editor)
    # BackgroundVideo
    back_videos = make_back_videos(p_back_dir=p.back_dir)
    # ObjectDatabase
    database = make_database(field_videos=field_videos)

    logger('done')

    # do stuff
    if args.task == 'composite':
        composite(args=args, p=p, logger=logger, database=database, field_videos=field_videos, back_videos=back_videos, classes=classes, n_class=n_class)
    elif args.task == 'rotate':
        rotate(args=args, p=p, logger=logger, field_videos=field_videos)

def composite(*, args, p, logger, database, field_videos, back_videos, classes, n_class):
    if args.resume:
        logger('searching for where you left off...', end='')
        start_indices = get_start_indices(p)
        logger('done')

    # class probabilities
    if args.probability:
        # assert len(args.probability) == n_class, 'n_probability must be equal to n_class'
        prob = np.array(args.probability)
        prob /= prob.sum()
    else:
        prob = None

    # simulate datasets for each pair of (a background video, set of foregound objects in a video)
    logger('synthesizing dataset...')

    n_back = len(back_videos)
    q = args.n_sample // n_back
    r = args.n_sample % n_back
    # number of the composite images that will be generated for each BackgroundVideo object
    n_composite_per_back_video = np.array([q] * n_back)
    n_composite_per_back_video[:r] += 1 

    # get statistics used to determine number of objects in each composite image
    n_obj_mean, n_obj_std = database.stats('n_obj', mode=['mean', 'std'], pivot='image')

    # get where to begin
    initial = 0
    if args.resume:
        n_sample_generated = sum(start_indices.values())
        initial = n_sample_generated
        if n_sample_generated >= args.n_sample:
            return
    try:
        with tqdm(initial=initial, total=args.n_sample, dynamic_ncols=True) as pbar:
            for back_video, n_composite in zip(back_videos, n_composite_per_back_video):
                with back_video.open():
                    if args.resume:
                        start_index = start_indices[back_video.stem]
                    else:
                        start_index = 0
                    for index in range(start_index, n_composite):
                        mkdata.make_composite_image(
                            database=database,
                            back_video=back_video,
                            pathes=p,
                            prob=prob, 
                            suffix=args.suffix,
                            bbox=args.bbox,
                            augment_intensity=args.augment,
                            index=index,
                            n_obj_mean=n_obj_mean,
                            n_obj_std=n_obj_std,
                            scale_jitter=args.scale_jitter
                        )
                        pbar.update(1)
    except KeyboardInterrupt:
        logger('interrupted by keyboard')
    else:
        logger('done')

    # field_videos_with_rep_images = [video for video in field_videos if video.rep_images]
    # division = make_division(
    #     args.n_sample, 
    #     back_videos, 
    #     field_videos_with_rep_images
    # ) # make optimal divisions
    # try:
    #     initial = 0
    #     if args.resume:
    #         n_sample_generated = sum([sum(v.values()) for v in start_indices.values()])
    #         initial = n_sample_generated
    #         if n_sample_generated >= args.n_sample:
    #             raise NoJobToDo(f'{n_sample_generated} samples have already been generated.')
    #     with tqdm(initial=initial, total=args.n_sample, dynamic_ncols=True) as pbar:
    #         for i, back_video in enumerate(back_videos):
    #             with back_video.open():
    #                 for j, field_video in enumerate(field_videos_with_rep_images):
    #                     if args.resume:
    #                         start_index = start_indices[back_video.stem][field_video.stem] # last_index + 1 if key exists, 0 otherwise
    #                         if start_index > division[i, j]:
    #                             raise ValueError(f'it seems like more than {args.n_sample} samples have already been generated')
    #                     else:
    #                         start_index = 0
    #                     for index in range(start_index, division[i, j]):
    #                         mkdata.make_composite_image(
    #                             field_video=field_video, 
    #                             back_video=back_video, 
    #                             pathes=p, 
    #                             prob=prob,
    #                             suffix=args.suffix,
    #                             bbox=args.bbox,
    #                             # device=device,
    #                             augment_intensity=args.augment,
    #                             index=index
    #                         )
    #                         pbar.update(1)
    # except KeyboardInterrupt:
    #     logger('interrupted by keyboard')
    # except NoJobToDo as e:
    #     logger(e)
    # else:
    #     logger('done')

def rotate(*, logger, field_videos, args, p):
    logger('generating images & labels for the first step of domain adaptation...')
    n_rep_image = sum([len(video.rep_images) for video in field_videos])
    try:
        with tqdm(total=n_rep_image * 360, dynamic_ncols=True) as pbar:
            for field_video in field_videos:
                for rep_image in field_video.rep_images:
                    mkdata.make_rotated_rep_image(rep_image, pathes=p, bbox=args.bbox, suffix=args.suffix, pbar=pbar, min_area=args.min_area)
    except KeyboardInterrupt:
        logger('interrupted by keyboard')
    else:
        logger('done')
    
if __name__ == '__main__':
    main()
