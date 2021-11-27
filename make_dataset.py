#!/usr/bin/env/python3
import argparse
import pathlib
import os
import errno
import time
from collections import defaultdict, namedtuple
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
        help='[required] input directory where VOC mask files named [image_name].npy are placed', 
        type=pathlib.Path, 
        required=True
    )
    parser.add_argument(
        '--rep', 
        help='[required] input directory where representative image files are placed. Note that a filename must have the format of [video name]_[%%M%%S] where %%M and %%S denote minute and second as decimal numbers, respectedly', 
        type=pathlib.Path, 
        required=True
    )
    parser.add_argument(
        '--field',
        help='[required] input directory where video files of fields are placed', 
        type=pathlib.Path, 
        required=True
    )
    parser.add_argument(
        '--back', '--background', 
        help='[required] input directory where background video files are placed', 
        type=pathlib.Path, 
        required=True
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
        required=True
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

    args = parser.parse_args()
    if not args.extension.startswith('.'):
        args.extension = '.' + args.extension
    if args.root is not None:
        args_dict = vars(args)
        for key in ['mask', 'rep', 'field', 'back', 'output', 'domain_adaptation', 'labels']:
            args_dict[key] = args.root / args_dict[key]
            if key in ['mask', 'rep', 'field', 'back', 'labels']:
                if not args_dict[key].exists():
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(args_dict[key]))
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

    # class probabilities
    if args.probability:
        assert len(args.probability) == n_class, 'n_probability must be equal to n_class'
    else:
        args.probability = [1.0/n_class] * n_class
    prob = np.array(args.probability)
    prob /= prob.sum()

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
    counts = defaultdict(lambda: defaultdict(int)) # counters for avoiding file name collision
    n_back_video = len(back_videos)
    n_field_video = len(field_videos)
    n_sample_per_pair = max(1, args.n_sample // (n_back_video * n_field_video))

    fmt = lambda n_sample_generated, n_sample, time_elapsed, time_remaining: (
        f'  {n_sample_generated}/{n_sample} = {n_sample_generated/n_sample*100:.1f}% completed. '
        f'About {time.strftime("%Hh %Mmin", time.gmtime(time_remaining + 30))} left. '
        f'({time_elapsed / n_sample_generated:.2f} sec per sample)'
    )
    counter = utils.make_counter(
        n_total=args.n_sample,
        fmt=fmt,
        verbose=args.verbose, 
        newline=False
    )

    try:
        while True:
            for back_video in back_videos:
                with back_video.open():
                    for field_video in field_videos:
                        if not field_video.rep_images:
                            continue
                        for _ in range(n_sample_per_pair):
                            frame = back_video.random_read(device=device, noexcept=True)
                            frame, label = mkdata.synthesize(frame=frame, field_video=field_video, prob=prob)

                            # save as files
                            index = counts[back_video.stem][field_video.stem]
                            output_stem = f'synth_back_{back_video.stem}_field_{field_video.stem}_{index}'
                            output_image_name = p.output_images_dir / (output_stem + args.extension)
                            output_label_name = p.output_labels_dir / (output_stem + '.txt')
                            utils.save_image(frame, output_image_name)
                            label.save(output_label_name)

                            if args.bbox:
                                output_labeled_image_name = p.output_labeled_images_dir / output_image_name.name
                                utils.save_labeled_image(frame, label, output_labeled_image_name, classes)

                            counter()
                            counts[back_video.stem][field_video.stem] += 1

    except StopIteration:
        logger('done')

    if args.domain_adaptation is not None:
        logger('generating images & labels for the first step of domain adaptation...')
        n_rep_image = sum([len(video.rep_images) for video in field_videos])
        counter = utils.make_counter(
            n_total=n_rep_image * 360,
            fmt=fmt,
            verbose=args.verbose, 
            newline=False
        )
        try:
            for field_video in field_videos:
                for rep_image in field_video.rep_images:
                    mkdata.generate_rotated_rep_image(rep_image, pathes=p, bbox=args.bbox, counter=counter)
        except StopIteration:
            logger('done')
    
if __name__ == '__main__':
    main()
