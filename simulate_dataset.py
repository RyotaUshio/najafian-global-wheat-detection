import torch
import torchvision
import cv2
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
import re
import argparse
import dataclasses
from typing import ClassVar, Callable
import time
from collections import OrderedDict

_patterns = [
  re.compile(r'(.+)_(\d{4})_class(\d)\.(.+)'),
  re.compile(r'(.+)_(\d{4})(?!_class\d)\.(.+)'),
  re.compile(r'(.+)\.(.+)')
]

def parse_fname(fname):
    for i, pattern in enumerate(_patterns):
        m = pattern.match(fname)
        if m:
            break
        if i == len(_patterns) - 1:
            raise Exception(f'An invalid file name "{fname}" was given')

    ret = {k: None for k in ['stem', 'time', 'i_class', 'extension']}
    
    if i == 0:
      keys = ['stem', 'time', 'i_class', 'extension']
    elif i == 1:
      keys = ['stem', 'time', 'extension']
    elif i == 2:
      keys = ['stem', 'extension']

    ret.update({k: v for k, v in zip(keys, m.groups())})
    if ret['i_class'] is not None:
      ret['i_class'] = int(ret['i_class'])

    return ret

def get_image_path(p_mask, p_rep_dir, extension):
    parsed = parse_fname(p_mask.name)
    if not extension.startswith('.'):
        extension = '.' + extension
    image_name = parsed['stem'] + '_' + parsed['time'] + extension
    return p_rep_dir / image_name

def get_classes(p_labels):
    classes = []
    with open(p_labels) as f:
        for line in f:
            line = line.rstrip()
            if line not in ['__ignore__', '_background_']:
                classes.append(line)
    return classes

def tensorimage_to_numpy(tensor):
    """(C x H x W) to (H x W x C)
    """
    return tensor.numpy().transpose((1, 2, 0))

from torchvision.transforms.functional import to_tensor

def imread(fname):
    return torchvision.io.read_image(
      str(fname),
      mode=torchvision.io.ImageReadMode.RGB
    )

read_image = imread

def read_frame(cap, i_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    opened, frame = cap.read()
    if not opened:
        raise RuntimeError('Could not read {i_frame}-th frame from the background video {p_video} with {n_frame} frames')
    frame = frame[:, :, [2, 1, 0]] # OpenCV reads image in (B, G, R(, A)) order
    frame = frame.transpose((2, 0, 1)) # (H, W, C) -> (C, H, W)
    frame = torch.as_tensor(frame)
    return frame
    

def save_image(tensor, fname):
    if isinstance(tensor, torch.ByteTensor):
        tensor = tensor.float()
        tensor /= 255.0
    torchvision.utils.save_image(tensor, fname)
  
def get_bbox(mask):
    if mask.ndim <= 2:
        mask = torch.unsqueeze(mask, 0)
    [[left, top, right, bottom]] = torchvision.ops.masks_to_boxes(mask).to(int)
    return dict(
        left=left.item(), 
        right=right.item(), 
        top=top.item(), 
        bottom=bottom.item()
    )

    
@dataclasses.dataclass
class foreground_obj:
    _random_rotate: ClassVar[Callable] = torchvision.transforms.RandomRotation(degrees=180, expand=True, fill=0)

    image: torch.Tensor
    mask: torch.Tensor
    image_path: pathlib.Path
    mask_path: pathlib.Path
    stem: str
    time: time.struct_time
    i_class: int
    class_name: str
    extension: str
    image_height: int = dataclasses.field(init=False)
    image_width: int = dataclasses.field(init=False)
    bbox: dict = dataclasses.field(init=False)
    image_cropped: torch.Tensor = dataclasses.field(init=False)
    mask_cropped: torch.Tensor = dataclasses.field(init=False)

    def __post_init__(self):
        _, self.image_height, self.image_width = self.image.size()
        self.bbox = get_bbox(self.mask)
        top, bottom, left, right = self.bbox['top'], self.bbox['bottom'], self.bbox['left'], self.bbox['right']
        self.image_cropped = self.image[:, top:bottom, left:right]
        self.mask_cropped = self.mask[top:bottom, left:right]
        
    def random_place(self, background):
        """Randomly place the object on the given background image. This is an in-place operation.
        
        Parameters
        ----------
        background: array or tensor
          An image on which the object is placed
        """
        if background.shape[-1] == 3:
            background = background.transpose((-1, -3, -2))

        rotated = self._random_rotate(torch.cat([self.image_cropped, torch.unsqueeze(self.mask_cropped, 0)]))
        image_cropped, mask_cropped = torch.split(rotated, [3, 1], dim=0)
        mask_cropped = torch.squeeze(mask_cropped, dim=0).to(torch.bool)
        _, h, w = image_cropped.size() # size of cropped region after rotation
        
        top = np.random.randint(0, self.image_height - h)
        left = np.random.randint(0, self.image_width - w)
        background[:, top:top+h, left:left+w][:, mask_cropped] = image_cropped[:, mask_cropped]
        
        # bounding box info
        bbox = get_bbox(mask_cropped)
        bbox['top'] += top
        bbox['bottom'] += top
        bbox['left'] += left
        bbox['right'] += left
        return bbox
        
    def argumented(self):
        """新しいオブジェクトを返すように！！"""
    
    @classmethod
    def from_npy(cls, image_path, mask_path, classes=None):
        """generate a list of foreground objects from .npy file
        """
        image_path = pathlib.Path(image_path)
        mask_path = pathlib.Path(mask_path)
        parsed = parse_fname(mask_path.name)
        
        stem = parsed['stem'] # filename without extensions
        time_elapsed = time.strptime(parsed['time'], '%M%S') # time elapsed since the start of the vid
        i_class = parsed['i_class']
        class_name = None
        if classes is not None:
            class_name = classes[i_class - 1] # i_class is 1-origin
        extension = parsed['extension'] # extension of image file e.g. 'png', 'jpeg', etc.

        # read mask & image files
        masks = torch.as_tensor(np.load(mask_path))
        image = imread(image_path)

        objects = []
        for mask in masks:
            fg = foreground_obj(
                image=image, 
                mask=mask,
                image_path=image_path,
                mask_path=mask_path,
                stem=stem,
                time=time_elapsed,
                i_class = i_class,
                class_name=class_name,
                extension = extension
            )               
            objects.append(fg)

        return objects, stem, class_name


class yolo_label:
  def __init__(self, image_width, image_height):
    self.bboxes = []
    self.image_width = image_width
    self.image_height = image_height
    self.dicts =  []

  def add(self, i_class: int, bbox: dict):
    x_center = 0.5 * (bbox['left'] + bbox['right'])
    y_center = 0.5 * (bbox['top'] + bbox['bottom'])
    width = bbox['right'] - bbox['left']
    height = bbox['bottom'] - bbox['top']

    # normalize so that everything will be in [0, 1]
    x_center /= self.image_width
    y_center /= self.image_height
    width /= self.image_width
    height /= self.image_height

    self.bboxes.append(
      OrderedDict(
        i_class=i_class, 
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height
      )
    )
    self.dicts.append(bbox)

  def save(self, fname):
    with open(fname, 'w') as f:
      for label in self.bboxes:
        row = ' '.join([str(value) for value in label.values()])
        row += '\n'
        f.write(row)

  def to_tensor(self):
      return torch.tensor([[bbox['left'], bbox['top'], bbox['right'], bbox['bottom']] for bbox in self.dicts])

  def class_name_list(self, classes):
      return [classes[bbox['i_class']] for bbox in self.bboxes]


def synthesize(frame, objs, classes, prob, width, height):
    """randomly place foreground objects onto the given frame in-place
    """
    rng = np.random.default_rng()
    n_obj = int(rng.normal(loc=6, scale=1.5)) # 謎のヒューリティクス
    n_class = len(classes)
    label = yolo_label(image_width=width, image_height=height)
    # frame = background_argumentation(frame)

    for i in range(n_obj):
        while True:
          i_class = rng.choice(range(n_class), p=prob)
          class_name = classes[i_class]
          objects = objs[class_name]
          if objects:
            break
        obj = rng.choice(objects)
        # obj = obj.argumented() 
        bbox = obj.random_place(frame)
        label.add(i_class, bbox)
    return frame, label
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', help='root directory of the project (optional). if given, all other pathes are assumed to be relative to the root.', type=pathlib.Path)
    parser.add_argument('mask_dir', help='input directory where mask files named [image_name]_class[class_id].npy are placed', type=pathlib.Path)
    parser.add_argument('rep_dir', help='input directory where representative image files are placed. Note that a filename must have the format of [video name]_[%%M%%S] where %%M and %%S denote minute and second as decimal numbers, respectedly', type=pathlib.Path)
    parser.add_argument('video_dir', help='input directory where video files of chestnut fields are placed', type=pathlib.Path)
    parser.add_argument('background_dir', help='input directory where background video files are placed', type=pathlib.Path)
    parser.add_argument('output_dir', help='output directory where generated dataset will be exported. Note that two directories will be made under the output_dir, namely "images" and "labels"', type=pathlib.Path)
    parser.add_argument('labels', help='path to the labels file', type=pathlib.Path)
    parser.add_argument('n_sample', help='number of samples of the dataset that will be generated synthetically', type=int)
    parser.add_argument('-p', '--probability', help='probability that a foreground object is chosen from each class when synthesizving dataset. This must have the same length as number of classes. Defaults to [1/n_class, 1/n_class, ...]', nargs='*', type=float)
    parser.add_argument('-e', '--extension', help='extension(s) of image files', nargs='?', default='.png')
    parser.add_argument('-v', '--verbose', help='if specified, display the progress and time remaining', action='store_true')
    parser.add_argument('-b', '--bbox', help='if specified, images with calculated bounding boxes are also saved', action='store_true')
    args = parser.parse_args()
    if not args.extension.startswith('.'):
        args.extension = '.' + args.extension
    if args.root is not None:
        args_dict = vars(args)
        for key in ['mask_dir', 'rep_dir', 'video_dir', 'background_dir', 'output_dir', 'labels']:
            args_dict[key] = args.root / args_dict[key]
    return args

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    args = parse_args()
    
    p_mask_dir = args.mask_dir
    p_rep_dir = args.rep_dir
    p_video_dir = args.video_dir
    p_back_dir = args.background_dir
    p_output_dir = args.output_dir
    p_output_images_dir = p_output_dir / 'images/all'
    p_output_labels_dir = p_output_dir / 'labels/all'        
    p_output_images_dir.mkdir(parents=True, exist_ok=True)
    p_output_labels_dir.mkdir(parents=True, exist_ok=True)

    if args.bbox:
        p_output_labeled_images_dir = p_output_dir / 'labeled_images'
        p_output_labeled_images_dir.mkdir(exist_ok=True)
    
    classes = get_classes(args.labels)
    n_class = len(classes)

    # class probabilities
    if args.probability:
        assert len(args.probability) == n_class
    else:
        args.probability = [1.0/n_class] * n_class
    prob = np.array(args.probability)
    prob /= prob.sum()

    # get the stems of representative frames
    list_json = list(p_rep_dir.glob('*.json'))
    json_stems = [parse_fname(json.name)['stem'] for json in list_json]
    rep_stems = list(set(json_stems)) # get unique items

    # stem = filename - suffix : corresponds to each video clip with foreground objects
    stems = [parse_fname(p_video.name)['stem'] for p_video in p_video_dir.glob('*.mp4')]
    obj_dict = {
        stem: {class_name: [] for class_name in classes}
        for stem in rep_stems
    }

    # generate foreground objects and put them into obj_dict
    if args.verbose:
        print('constructing foreground objects...', end='')
    for p_mask in p_mask_dir.glob('*.npy'):
        p_image = get_image_path(p_mask=p_mask, p_rep_dir=p_rep_dir, extension=args.extension)
        objects, stem, class_name = foreground_obj.from_npy(p_image, p_mask, classes) # a list of foreground objects contained in p_image
        obj_dict[stem][class_name].extend(objects)
    if args.verbose:
        print('done')

    # simulate datasets for each pair of (a background video, set of foregound objects in a video)
    rng = np.random.default_rng()
    n_back_video = len(list(p_back_dir.glob('*.mp4')))
    n_obj_video = len(stems)
    n_sample_per_pair = args.n_sample // (n_back_video * n_obj_video)
    n_sample_generated = 0

    if args.verbose:
        print('synthesizing dataset...')
    
    t0 = time.time()
    counts = dict()
    finished = False

    while True:
        for p_video in p_back_dir.glob('*.mp4'):

            # open background video
            cap = cv2.VideoCapture(str(p_video))
            if not cap.isOpened():
                raise Exception(f'Cannot open file {p_video}')
            n_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            n_frame = int(n_frame)
            if p_video not in counts:
                counts[p_video] = {}
            
            for stem in rep_stems:

                if stem not in counts[p_video]:
                    counts[p_video][stem] = 0
                else:
                    counts[p_video][stem] += 1
                
                for _ in range(n_sample_per_pair):
                    # randomly select a background frame from p_video
                    i_frame = rng.choice(n_frame)
                    try:
                        frame = read_frame(cap, i_frame)
                    except RuntimeError as e:
                        print('Something went wrong while reading frames from a video:')
                        print(e)

                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                    frame, label = synthesize(
                        frame=frame,
                        objs=obj_dict[stem],
                        classes=classes,
                        prob=prob,
                        width=width,
                        height=height
                    )

                    # save as files
                    output_stem = f'back_{p_video.stem}_rep_{stem}_synthesized_{counts[p_video][stem]}'
                    output_image_name = p_output_images_dir / (output_stem + args.extension)
                    output_label_name = p_output_labels_dir / (output_stem + '.txt')
                    save_image(frame, output_image_name)
                    label.save(output_label_name)

                    if args.bbox:
                        output_labeled_image_name = p_output_labeled_images_dir / (output_stem + args.extension)
                        save_image(
                            torchvision.utils.draw_bounding_boxes(
                                image=frame,
                                boxes=label.to_tensor(),
                                labels=label.class_name_list(classes)
                            ),
                            output_labeled_image_name,
                        )

                    n_sample_generated += 1
                    if n_sample_generated >= args.n_sample:
                        finished = True
                        break

                    if args.verbose:
                        time_elapsed = time.time() - t0
                        velo = n_sample_generated / time_elapsed
                        time_remaining = (args.n_sample - n_sample_generated) / velo
                        print(
                            f'  {n_sample_generated}/{args.n_sample} = {n_sample_generated/args.n_sample*100:.1f}% completed. '
                            f'About {time.strftime("%Hh %Mmin", time.gmtime(time_remaining + 30))} left.\r',
                            end=''
                        )

                    if finished:
                        break

                if finished:
                    break

            if finished:
                break

        if finished:
            break

    if args.verbose:
        print('\ndone')

