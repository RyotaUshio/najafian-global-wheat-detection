import torch
import torchvision
import cv2
import albumentations as A
import numpy as np
import skimage.measure
import pathlib
import os
import errno
import re
import argparse
import dataclasses
from typing import ClassVar, Callable
import time
from collections import namedtuple, OrderedDict

_patterns = [
  re.compile(r'(.+)_(\d{4})(\..+)'),
  re.compile(r'(.+)(\..+)')
]

def parse_fname(fname):
    fname = pathlib.Path(fname).name
    for i, pattern in enumerate(_patterns):
        m = pattern.match(fname)
        if m:
            break
        if i == len(_patterns) - 1:
            raise Exception(f'An invalid file name "{fname}" was given')

    ret = {k: None for k in ['stem', 'time', 'extension']}
    
    if i == 0:
      keys = ['stem', 'time', 'extension']
    elif i == 1:
      keys = ['stem', 'extension']
    else:
        assert False

    ret.update({k: v for k, v in zip(keys, m.groups())})
    return ret

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

def read_image(path, device=None):
    image = torchvision.io.read_image(
      str(path),
      mode=torchvision.io.ImageReadMode.RGB
    )
    if device is not None:
        image = image.to(device)
    return image

def read_frame(cap, i_frame, device=None):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    opened, frame = cap.read()
    if not opened:
        raise RuntimeError(f'Could not read {i_frame}-th frame from the background video {p_video} with {n_frame} frames')
    frame = frame[:, :, [2, 1, 0]] # OpenCV reads image in (B, G, R(, A)) order
    frame = frame.transpose((2, 0, 1)) # (H, W, C) -> (C, H, W)
    frame = torch.as_tensor(frame)
    if device is not None:
        frame = frame.to(device)
    return frame
    
def save_image(tensor, fname):
    tensor = tensor.cpu()
    if isinstance(tensor, torch.ByteTensor):
        tensor = tensor.float()
        tensor /= 255.0
    torchvision.utils.save_image(tensor, fname)

def save_labeled_image(image, label, output_labeled_image_name):
    save_image(
        torchvision.utils.draw_bounding_boxes(
            image=image.cpu(),
            boxes=label.to_tensor(),
            labels=label.class_name_list(classes)
        ),
        output_labeled_image_name,
    )    
  
def get_bbox(masks):
    if masks.ndim <= 2:
        masks = torch.unsqueeze(masks, 0)

    boxes = torchvision.ops.masks_to_boxes(masks).to(int)
    ret = []
    for box in boxes:
        left, top, right, bottom = box
        ret.append(dict(left=left.item(), right=right.item(), top=top.item(), bottom=bottom.item()))
    if len(ret) == 1:
        ret = ret[0]
    return ret

def make_classwise_mask(p_mask, n_class):
    """p_mask: path to .npy file which contains class-wise masks in the VOC format
    """
    labels = torch.as_tensor(np.load(p_mask))
    # ret = dict()
    masks = []
    for i_class in range(n_class):
        classwise_mask = (labels == i_class + 1)
        # ret.update({i_class: classwise_mask})
        masks.append(classwise_mask)
    # return ret
    return torch.stack(masks)

def make_objectwise_mask(classwise_masks, n_class):
    """classwise_masks: tensor of shape n_class x H x W (stack of masks)
    """
    ret = dict()
    # masks = []
    for i_class in range(n_class):
        objectwise_masks = skimage.measure.label(classwise_masks[i_class]) # labels connected components
        object_id = np.unique(objectwise_masks)[1:] # ignore background = 0
        objectwise_masks = (objectwise_masks == object_id.reshape(-1, 1, 1))
        ret.update({i_class: torch.as_tensor(objectwise_masks)})
        # masks.append(torch.as_tensor(objectwise_masks))
    return ret
    # return torch.stack(masks)
    
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
        rotated = self._random_rotate(
            torch.cat(
                [self.image_cropped, torch.unsqueeze(self.mask_cropped, 0)]
            )
        )
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
    def from_voc(cls, p_image, p_mask, classes, device=None):
        """generate a list of foreground objects from .npy file
        """
        p_image = pathlib.Path(p_image)
        p_mask = pathlib.Path(p_mask)
        n_class = len(classes)

        # read mask & image files
        classwise_mask = make_classwise_mask(p_mask, n_class)
        objectwise_mask = make_objectwise_mask(classwise_mask, n_class)
        if device is not None:
            classwise_mask = classwise_mask.to(device)
            for i_class, masks in objectwise_mask.items():
                objectwise_mask.update({i_class: masks.to(device)})
        image = read_image(p_image, device)

        parsed = parse_fname(p_mask)
        objects = dict()
        stem = parsed['stem'] # filename without extensions
        time_elapsed = time.strptime(parsed['time'], '%M%S') # time elapsed since the start of the vid

        for i_class in range(n_class):
            obj_list = []
            class_name = classes[i_class]
            extension = p_image.suffix # extension of image file e.g. 'png', 'jpeg', etc.
            masks = objectwise_mask[i_class]
            for mask in masks:
                obj = cls(
                    image=image, 
                    mask=mask,
                    image_path=p_image,
                    mask_path=p_mask,
                    stem=stem,
                    time=time_elapsed,
                    i_class=i_class,
                    class_name=class_name,
                    extension=extension
                )        
                obj_list.append(obj)
            objects.update({i_class: obj_list})
            
        return_type = namedtuple('return_type', ['objects', 'classwise_mask', 'objectwise_mask'])
        return return_type(objects, classwise_mask, objectwise_mask)

class yolo_label:
  def __init__(self, image):
    self.bboxes = []
    self.image_width = float(image.size(-1))
    self.image_height = float(image.size(-2))
    self.dicts = []

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


def synthesize(frame, objs, classes, prob):
    """randomly place foreground objects onto the given frame in-place
    """
    rng = np.random.default_rng()
    n_obj = int(rng.normal(loc=6, scale=1.5)) # 謎のヒューリティクス
    n_class = len(classes)
    label = yolo_label(frame)
    # frame = background_argumentation(frame)

    for i in range(n_obj):
        while True:
          i_class = rng.choice(range(n_class), p=prob)
          objects = objs[i_class]
          if objects:
            break
        obj = rng.choice(objects)
        # obj = obj.argumented() 
        bbox = obj.random_place(frame)
        label.add(i_class=i_class, bbox=bbox)
    return frame, label

def generate_rotated_rep_images(p_image, classwise_mask, *, p_output_dir, n_class, device, bbox, counter):
    image = read_image(p_image)
    concat = torch.cat(
        [image, classwise_mask]
    ).to(device)
    
    p_output_images_dir = p_output_dir / 'images/all'
    p_output_labels_dir = p_output_dir / 'labels/all'
    p_output_images_dir.mkdir(parents=True, exist_ok=True)
    p_output_labels_dir.mkdir(parents=True, exist_ok=True)
    if bbox:
        p_output_labeled_images_dir = p_output_dir / 'labeled_images'
        p_output_labeled_images_dir.mkdir(parents=True, exist_ok=True)

    for degree in range(360):
        image, label = _generate_rotated_rep_images_impl(concat, degree=degree, n_class=n_class)
        output_stem = f'{p_image.stem}_{degree:03}'
        output_image_name = p_output_images_dir / (output_stem + p_image.suffix)
        output_label_name = p_output_labels_dir / (output_stem + '.txt')
        save_image(image, output_image_name)
        label.save(output_label_name)
        if bbox:
            output_labeled_image_name = p_output_labeled_images_dir / (output_stem + args.extension)
            save_labeled_image(frame, label, output_labeled_image_name)
        counter()
            
def _generate_rotated_rep_images_impl(concat, *, degree, n_class):
    """image: Tensor (C, H, W)
    classwise_mask: Tensor (n_class, H, W)
    """
    rotated = torchvision.transforms.functional.rotate(
        concat, degree
    )
    image, classwise_mask = torch.split(rotated, [3, n_class], dim=0)
    objectwise_mask = make_objectwise_mask(classwise_mask, n_class)
    label = yolo_label(image)
        
    # bounding box info
    for i_class in range(n_class):
        bboxes = get_bbox(objectwise_mask[i_class])
        for bbox in bboxes:
            label.add(i_class=i_class, bbox=bbox)

    return image, label

        
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
        '--video', 
        help='[required] input directory where video files of chestnut fields are placed', 
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
    parser.add_argument(
        '-d', '--domain-adaptation', 
        help='output directory where images & labels for the first step of domain adaptation are placed (generated only if this flag is passed)',
        type=pathlib.Path
    )

    args = parser.parse_args()
    if not args.extension.startswith('.'):
        args.extension = '.' + args.extension
    if args.root is not None:
        args_dict = vars(args)
        for key in ['mask_dir', 'rep_dir', 'video_dir', 'background_dir', 'output_dir', 'labels']:
            args_dict[key] = args.root / args_dict[key]
            if key in ['mask_dir', 'rep_dir', 'video_dir', 'background_dir']:
                if not args_dict[key].exists():
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(args_dict[key]))
    return args

def make_counter(*, n_total, fmt, verbose, newline):
    i = 0
    t0 = time.time()
    def counter(*args, **kwargs):
        nonlocal i
        nonlocal t0
        nonlocal n_total
        nonlocal verbose
        nonlocal newline
        i += 1
        time_elapsed = time.time() - t0
        velocity = i / time_elapsed
        time_remaining = (n_total - i) / velocity
        if verbose:
            end = '\n' if newline else '\r'
            string = fmt(i, n_total, time_elapsed, time_remaining, *args, **kwargs)
            #if not newline:
            #    string += '\r'
            print(string, end=end)
        if i >= n_total:
            raise StopIteration
    return counter

def make_logger(*, verbose):
    def log(*args, **kwargs):
        nonlocal verbose
        if verbose:
            print(*args, **kwargs)
    return log

if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    logger = make_logger(verbose=args.verbose)
    
    logger(f'using {device} device')
    logger('making output directories...', end='')
    
    p_mask_dir = args.mask
    p_rep_dir = args.rep
    p_video_dir = args.video
    p_back_dir = args.back
    p_output_dir = args.output
    p_output_images_dir = p_output_dir / 'images/all'
    p_output_labels_dir = p_output_dir / 'labels/all'        
    p_output_images_dir.mkdir(parents=True, exist_ok=True)
    p_output_labels_dir.mkdir(parents=True, exist_ok=True)

    if args.bbox:
        p_output_labeled_images_dir = p_output_dir / 'labeled_images'
        p_output_labeled_images_dir.mkdir(exist_ok=True)

    logger('done')
    logger('getting class information...', end='')
    
    classes = get_classes(args.labels)
    n_class = len(classes)

    logger('done')

    # class probabilities
    if args.probability:
        assert len(args.probability) == n_class, 'n_probability must be equal to n_class'
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
        stem: {i_class: [] for i_class in range(n_class)}
        for stem in rep_stems
    }

    # generate foreground objects and put them into obj_dict
    logger('constructing foreground objects...', end='')

    classwise_masks = dict()
    for p_mask in p_mask_dir.glob('*.npy'):
        stem = parse_fname(p_mask)['stem']
        p_image = p_rep_dir / (p_mask.stem + args.extension) # get_image_path(p_mask=p_mask, p_rep_dir=p_rep_dir, extension=args.extension)
        objects, classwise_mask, objectwise_mask = foreground_obj.from_voc(p_image, p_mask, classes, device) # a list of foreground objects contained in p_image
        for i_class in range(n_class):
            obj_dict[stem][i_class].extend(objects[i_class])
        classwise_masks.update({p_mask.stem: classwise_mask})
    logger('done')

    # simulate datasets for each pair of (a background video, set of foregound objects in a video)
    rng = np.random.default_rng()
    n_back_video = len(list(p_back_dir.glob('*.mp4')))
    n_obj_video = len(stems)
    n_sample_per_pair = max(1, args.n_sample // (n_back_video * n_obj_video))

    logger('synthesizing dataset...')
    
    t0 = time.time()
    counts = dict()
    finished = False

    fmt = lambda n_sample_generated, n_sample, time_elapsed, time_remaining: (
        f'  {n_sample_generated}/{n_sample} = {n_sample_generated/n_sample*100:.1f}% completed. '
        f'About {time.strftime("%Hh %Mmin", time.gmtime(time_remaining + 30))} left. '
        f'({time_elapsed / n_sample_generated:.2f} sec per sample)'
    )
    counter = make_counter(
        n_total=args.n_sample,
        fmt=fmt,
        verbose=args.verbose, 
        newline=False
    )

    try:
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
                        while True:
                            try:
                                frame = read_frame(cap, i_frame, device)
                            except RuntimeError as e:
                                print('Something went wrong while reading frames from a video:')
                                print(e)
                            else:
                                break

                        frame, label = synthesize(
                            frame=frame,
                            objs=obj_dict[stem],
                            classes=classes,
                            prob=prob
                        )

                        # save as files
                        output_stem = f'back_{p_video.stem}_rep_{stem}_synthesized_{counts[p_video][stem]}'
                        output_image_name = p_output_images_dir / (output_stem + args.extension)
                        output_label_name = p_output_labels_dir / (output_stem + '.txt')
                        save_image(frame, output_image_name)
                        label.save(output_label_name)

                        if args.bbox:
                            output_labeled_image_name = p_output_labeled_images_dir / (output_stem + args.extension)
                            save_labeled_image(frame, label, output_labeled_image_name)

                        counter()

    except StopIteration:
        logger('\ndone')

    if args.domain_adaptation is not None:
        logger('generating images & labels for the first step of domain adaptation...')
        n_rep = len(list_json)
        try:
            for p_json in list_json:
                p_image = p_rep_dir / (p_json.stem + args.extension)
                generate_rotated_rep_images(
                    p_image, 
                    classwise_masks[p_image.stem], 
                    p_output_dir=args.domain_adaptation, 
                    n_class=n_class, 
                    device=device,
                    bbox=args.bbox,
                    counter=make_counter(
                        n_total=n_rep * 360,
                        fmt=fmt,
                        verbose=args.verbose, 
                        newline=False
                    )
                )
        except StopIteration:
            logger('\ndone')
