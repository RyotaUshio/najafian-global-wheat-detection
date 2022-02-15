#!/usr/bin/env/python3
import numpy as np
import pathlib
import shutil
import argparse

def parse_args():
  parser = argparse.ArgumentParser(
    description='Split a dataset into train/test/val subsets and export the result in various ways.',
    formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument(
    'path',
    help='''\
    path to the root of your dataset
    It is expected to have the following structure:
      path
      ├───images
      |   └───all
      |       |───image1.png
      |       ...
      ├───labels
          └───all
              |───image1.txt
              ...
                     
    images/labels in "all" will be split into train/val/test subsets.''',
    nargs='+'
  )
  parser.add_argument(
    '--train', 
    help='number or proportion of samples in the training set',
    type=float,
    default=0.0
  )
  parser.add_argument(
    '--test', 
    help='number or proportion of samples in the test set',
    type=float,
    default=0.0
  )
  parser.add_argument(
    '--val', '--valid', 
    help='number or proportion of samples in the validation set',
    type=float,
    default=0.0
  )
  parser.add_argument(
    '-o', '--output',
    help='path to the output directory where divided data and log files will be placed',
  )
  parser.add_argument(
    '-c', '--copy',
    help='[deprecated] whether to copy the files in "images/all" and "labels/all" to the output directory',
    action='store_true'
  )
  parser.add_argument(
    '-m', '--move',
    help='[deprecated] whether to move the files in "images/all" and "labels/all" to the output directory. This action is destructive',
    action='store_true'
  )
  parser.add_argument(
    '-t', '--text',
    help='whether to write results in text files',
    action='store_true'
  )
  args = parser.parse_args()

  sizes = [args.train, args.test, args.val]
  if all([size.is_integer() for size in sizes]):
    args.train, args.test, args.val = map(int, sizes)
    size_spec = 'abs'
  elif all([size < 1.0 for size in sizes]):
    if sum(sizes) != 1.0:
      raise ValueError('The proportions does not sum up to 1')
    size_spec = 'rel'
  else:
    raise ValueError(f"Invalid sizes {tuple(sizes)} was given")

  if args.move and args.copy:
    raise ValueError("Cannot specify both 'move' and 'copy'")

  if not (args.move or args.copy or args.text):
    raise ValueError("Results will not be exported unless at least one of ('move', 'copy', 'text') is specified")

  if args.output is None:
    args.output = args.path[0]
  
  return args, size_spec

def get_image_list(path):
  path = pathlib.Path(path)
  globs = ['*.png', '*.jpg', '*.jpeg']
  globs += list(map(lambda s: s.upper(), globs)) # ignore cases
  ret = []
  for glob in globs:
    ret += list(path.glob(glob))
  return ret

def random_split(seq, n_train, n_test, n_val):
  rng = np.random.default_rng()
  seq = seq[:n_train + n_test + n_val]
  rng.shuffle(seq)
  train, test, val = np.split(
    seq, 
    [n_train, n_train + n_test]
  )
  return train, test, val


def main():
  args, size_spec = parse_args()
  p_out = pathlib.Path(args.output)
  p_out.mkdir(parents=True, exist_ok=True)

  images = []
  for path in args.path:
    p_root = pathlib.Path(path)
    p_images = p_root / 'images'
    p_labels = p_root / 'labels'
    p_images_all = p_images / 'all'
    p_labels_all = p_labels / 'all'
    images += get_image_list(p_images_all)

  n_image_all = len(images)
  if size_spec == 'abs':
    n_image = args.train + args.test + args.val
    if n_image > n_image_all:
      raise ValueError(
        f'{args.train} + {args.test} + {args.val} = {n_image} exceeds the total number of images {n_image_all}')

  # make sure args.train, args.test and args.val are absolute numbers, not proportions
  if size_spec == 'rel':
    args.train = int(n_image_all * args.train)
    args.test = int(n_image_all * args.test)
    args.val = n_image_all - args.train - args.test

  print(f'{n_image_all} images found: {args.train} for train, {args.test} for test and {args.val} for val')

  train, test, val = random_split(images, args.train, args.test, args.val)
  divisions = dict(train=train, test=test, val=val)

  # export 
  for name, division in divisions.items():
    if division.size == 0:
      continue
    if args.move or args.copy:
      p_images_division = p_out / 'images' / name
      p_labels_division = p_out / 'labels' / name
      p_images_division.mkdir(parents=True, exist_ok=True)
      p_labels_division.mkdir(parents=True, exist_ok=True)
    if args.text:
      logfile = p_out / (name + '.txt')
      f = open(logfile, 'w')
    for p_image in division:
      image_name = p_image.name
      label_name = p_image.stem + '.txt'
      p_label = p_labels_all / label_name # here can be a bug for len(args.path) > 1
      if args.copy:
        shutil.copy(p_image, p_images_division)
        shutil.copy(p_label, p_labels_division)
      elif args.move:
        shutil.move(str(p_image), str(p_images_division))
        shutil.move(str(p_label), str(p_labels_division))
      if args.text:
        try:
          f.write(str(p_image.resolve()) + '\n')
        except Exception as e:
          print(f'An exception occured while handling {p_image}:')
          raise e

if __name__ == '__main__':
  main()

