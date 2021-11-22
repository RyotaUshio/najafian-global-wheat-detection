import numpy as np
from sklearn.model_selection import train_test_split
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
                     
    images/labels in "all" will be split into train/val/test subsets.'''
  )
  parser.add_argument(
    '--train', 
    help='number or proportion of samples in the training set',
    type=float
  )
  parser.add_argument(
    '--test', 
    help='number or proportion of samples in the test set',
    type=float
  )
  parser.add_argument(
    '--val', '--valid', 
    help='number or proportion of samples in the validation set',
    type=float
  )
  parser.add_argument(
    '-o', '--output',
    help='path to the output directory where divided data and log files will be placed',
  )
  parser.add_argument(
    '-c', '--copy',
    help='whether to copy the files in "images/all" and "labels/all" to the output directory',
    action='store_true'
  )
  parser.add_argument(
    '-m', '--move',
    help='whether to move the files in "images/all" and "labels/all" to the output directory. This action is destructive',
    action='store_true'
  )
  parser.add_argument(
    '-t', '--text',
    help='whether to write results in text files',
    action='store_true'
  )
  args = parser.parse_args()

  sizes = [args.train, args.test, args.val]
  if any([size is None for size in sizes]):
    raise ValueError("--train, --test and --val are not optional")
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
    args.output = args.path
  
  return args, size_spec

def get_image_list(path):
  path = pathlib.Path(path)
  globs = ['*.png', '*.jpg', '*.jpeg']
  globs += list(map(lambda s: s.upper(), globs)) # ignore cases
  ret = []
  for glob in globs:
    ret += list(path.glob(glob))
  return ret

def main():
  args, size_spec = parse_args()
  p_root = pathlib.Path(args.path)
  p_images = p_root / 'images'
  p_labels = p_root / 'labels'
  p_images_all = p_images / 'all'
  p_labels_all = p_labels / 'all'
  p_out = pathlib.Path(args.output)
  p_out.mkdir(parents=True, exist_ok=True)

  # divide into (train+val) and test
  images = get_image_list(p_images_all)
  train_val, test = train_test_split(
    images, 
    test_size=args.test,
    train_size=args.train + args.val
  )
  
  # divide (train+val) into train and val
  if size_spec == 'rel':
    ratios = np.array([args.train, args.val])
    ratios /= ratios.sum() # normalize
    train, val = ratios
  else:
    train, val = args.train, args.val

  train, val = train_test_split(
    train_val,
    test_size=args.val,
    train_size=args.train
  )

  divisions = dict(train=train, test=test, val=val)

  # export 
  for name, division in divisions.items():
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
      p_label = p_labels_all / label_name
      if args.copy:
        shutil.copy(p_image, p_images_division)
        shutil.copy(p_label, p_labels_division)
      elif args.move:
        shutil.move(p_image, p_images_division)
        shutil.move(p_label, p_labels_division)
      if args.move or args.copy:
        p_image_after = p_images_division / image_name
      else:
        p_image_after = p_image
      if args.text:
        f.write(str(p_image_after) + '\n')

if __name__ == '__main__':
  main()

