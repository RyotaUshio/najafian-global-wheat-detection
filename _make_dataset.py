from __future__ import annotations
import torch
import torchvision
import cv2
import albumentations as A
import numpy as np
import skimage.measure
import pathlib
import re
import argparse
import warnings
import dataclasses
from typing import ClassVar, Callable, Iterable, Union, Dict, List
import time
from collections import OrderedDict, defaultdict, namedtuple
import contextlib
import itertools
from tqdm.auto import tqdm

import utils

IMAGE_SUFFIXES = ['.png', '.jpeg', '.jpg', '.bmp', ] # acceptable suffixes of image files

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
  
def get_bbox(masks):
    return_list = True
    if masks.ndim <= 2:
        return_list = False
        masks = torch.unsqueeze(masks, 0)

    boxes = torchvision.ops.masks_to_boxes(masks).to(int)
    ret = []
    for box in boxes:
        left, top, right, bottom = box
        ret.append(dict(left=left.item(), right=right.item(), top=top.item(), bottom=bottom.item()))
    return ret if return_list else ret[0]

def bbox_to_pascal_voc(bbox: dict):
    """(x_min, y_min, x_max, y_max)
    """
    return bbox['left'], bbox['top'], bbox['right'], bbox['bottom']

def bbox_to_albumentations(bbox: dict, *, image_width: int, image_height: int):
    """normalized (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = bbox_to_pascal_voc(bbox)
    x_min /= image_width
    y_min /= image_height
    x_max /= image_width
    y_max /= image_height
    return x_min, y_min, x_max, y_max

def bbox_to_coco(bbox: dict):
    """(x_min, y_min, width, height)
    """
    x_min, y_min, x_max, y_max = bbox_to_pascal_voc(bbox)
    width = x_max - x_min
    height = y_max - y_min
    return x_min, y_min, width, height
        
def bbox_to_yolo(bbox: dict, *, image_width: int, image_height: int):
    """normalized (x_center, y_center, width, height)
    """
    x_min, y_min, x_max, y_max = bbox_to_pascal_voc(bbox)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    width = x_max - x_min
    height = y_max - y_min

    # normalize so that everything will be in [0, 1]
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    assert all([0<= val <=1] for val in [x_center, y_center, width, height])

    return x_center, y_center, width, height

def yolo_to_pascal_voc(bbox: dict, *, image_width: int, image_height: int):
    x_center, y_center, width, height = bbox['x_center'], bbox['y_center'], bbox['width'], bbox['height']
    left = x_center - 0.5 * width
    left = int(left * image_width + 0.5)
    right = x_center + 0.5 * width
    right = int(right * image_width + 0.5)
    top = y_center - 0.5 * height
    top = int(top * image_height + 0.5)
    bottom = y_center + 0.5 * height
    bottom = int(bottom * image_height + 0.5)
    return dict(left=left, right=right, top=top, bottom=bottom)

def make_classwise_mask(p_mask, n_class):
    """p_mask: path to .npy file which contains class-wise masks in the VOC format
    """
    labels = np.load(p_mask)
    labels = torch.as_tensor(labels)
    masks = []
    for i_class in range(n_class):
        classwise_mask = (labels == i_class + 1)
        masks.append(classwise_mask)
    return torch.stack(masks)

def make_objectwise_mask(classwise_masks, n_class):
    """classwise_masks: tensor of shape n_class x H x W (stack of masks)
    """
    ret = OrderedDict()
    for i_class in range(n_class):
        objectwise_masks = skimage.measure.label(classwise_masks[i_class].cpu()) # labels connected components
        object_id = np.unique(objectwise_masks)[1:] # ignore background = 0
        objectwise_masks = (objectwise_masks == object_id.reshape(-1, 1, 1))
        ret.update({i_class: torch.as_tensor(objectwise_masks)})
    return ret
    

@dataclasses.dataclass
class Video:
    p_video: pathlib.Path

    stem: str = dataclasses.field(init=False)
    capture: cv2.VideoCapture = dataclasses.field(init=False, default=None)
    is_open: bool = dataclasses.field(init=False, default=False)

    def __post_init__(self):
        parsed = parse_fname(self.p_video)
        self.stem = parsed['stem']

    def make_capture(self) -> None:
        if self.is_open:
            return 
        self.capture = cv2.VideoCapture(str(self.p_video))
        if not self.capture.isOpened():
            raise Exception(f'Cannot open file {p_video}')
        self.is_open = True

    def close(self) -> None:
        self.capture.release()
        self.is_open = False
        assert not self.capture.isOpened()

    @contextlib.contextmanager
    def open(self):
        try:
            self.make_capture()
            yield
        finally:
            self.close()

    def read_frame(self, i_frame, device=None, as_numpy=False):
        try:
            frame = utils.read_frame(self.capture, i_frame, device=device, as_numpy=as_numpy)
        except utils.FrameCannotBeLoaded:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(i_frame % 10, 'th')
            raise utils.FrameCannotBeLoaded(f'Cannot load {i_frame}{suffix} frame from a video {self.p_video}')
        return frame

    def __len__(self):
        if not self.is_open:
            raise ValueError('cannot get the number of frames from closed video')
        n_frame = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(n_frame)

    def random_read(self, device=None, noexcept=True, as_numpy=False):
        n_frame = len(self)
        while True:
            try:
                i_frame = np.random.choice(n_frame)
                frame = self.read_frame(i_frame, device, as_numpy=as_numpy)
            except utils.FrameCannotBeLoaded as e:
                print('Something went wrong while reading frames from a video:')
                if noexcept:
                    print(f'  {e}')
                else:
                    raise e
            else:
                break
        return frame


@dataclasses.dataclass(repr=False)
class BackgroundVideo(Video):
    pass


@dataclasses.dataclass(repr=False)
class FieldVideo(Video):
    p_rep_images: dataclasses.InitVar[pathlib.Path]
    p_masks: dataclasses.InitVar[pathlib.Path]
    classes: List[str]
    device: dataclasses.InitVar[Union[str, torch.device]] = 'cpu'

    label_editor: dataclasses.InitVar[Callable] = None

    rep_images: List['RepImage'] = dataclasses.field(init=False, default_factory=list)
    orig_images: List[torch.Tensor] = dataclasses.field(init=False)

    def __post_init__(
        self, 
        p_rep_images: pathlib.Path, 
        p_masks: pathlib.Path,
        device: Union[str, torch.device],
        label_editor
    ):
        super().__post_init__()
        # make sure pathes are pathlib.Path objects
        self.p_video = pathlib.Path(self.p_video)
        p_rep_images = pathlib.Path(p_rep_images)
        p_masks = pathlib.Path(p_masks)
        for suffix in utils.IMG_FORMATS:
            for p_rep_image in p_rep_images.glob(f'{self.stem}_*{suffix}'):
                p_mask = p_masks / (p_rep_image.stem + '.npy')
                image = RepImage(p_image=p_rep_image, p_mask=p_mask, video=self, classes=self.classes, device=device, stem=self.stem, label_editor=label_editor)
                self.rep_images.append(image)
        self.classes = image.classes # reflect the effect of label_editor
        self.orig_images = [rep_image.orig_image for rep_image in self.rep_images] # make copy for rep_image_as()

    @contextlib.contextmanager
    def rep_images_as(self, tmp_images):
        """context manager to set temporary rep-images

        The passed images must be \"spatially compatible\" with the original images. 
        You will get a broken result if **tmp_images** are output of data augmentation
        where coordinates are not preserved.
        """
        assert len(tmp_images) == len(self.orig_images)
        try:
            for rep_image, tmp_image in zip(self.rep_images, tmp_images):
                rep_image.set_image(tmp_image)
            yield
        finally:
            for rep_image, orig_image in zip(self.rep_images, self.orig_images):
                rep_image.set_image(orig_image)

    def get_objects(self, **kwargs):
        objects = []
        for rep_image in self.rep_images:
            objects += rep_image.get_objects(**kwargs)
        return objects


@dataclasses.dataclass(repr=False)
class RepImage:
    p_image: pathlib.Path
    p_mask: pathlib.Path
    video: FieldVideo
    classes: List[str]
    device: torch.device = 'cpu'
    stem: str = None

    label_editor: dataclasses.InitVar[Callable] = None

    timestamp: time.struct_time = dataclasses.field(init=False)
    image: torch.Tensor = dataclasses.field(init=False)
    orig_image: torch.Tensor = dataclasses.field(init=False)
    height: int = dataclasses.field(init=False)
    width: int = dataclasses.field(init=False)
    classwise_mask: torch.Tensor = dataclasses.field(init=False)
    objectwise_mask: dict = dataclasses.field(init=False)
    objects: dict = dataclasses.field(init=False)

    def __post_init__(self, label_editor):
        self.device = torch.device(self.device)
        parsed = parse_fname(self.p_image)
        self.timestamp = time.strptime(parsed['time'], '%M%S')
        self.image = utils.read_image(self.p_image, device=self.device)
        self.height = self.image.size(-2)
        self.width = self.image.size(-1)
        # read mask
        n_class = len(self.classes)
        classwise_mask = make_classwise_mask(self.p_mask, n_class)
        objectwise_mask = make_objectwise_mask(classwise_mask, n_class)

        ### edit masks
        if label_editor is not None:
            self.classes, classwise_mask, objectwise_mask = label_editor(
                classes=self.classes, 
                classwise_mask=classwise_mask, 
                objectwise_mask=objectwise_mask
            )

        # send masks to device
        classwise_mask = classwise_mask.to(self.device)
        for i_class, masks in objectwise_mask.items():
            objectwise_mask.update({i_class: masks.to(self.device)})

        # construct foreground objects
        self.objects = {}
        for i_class, class_name in enumerate(self.classes):
            obj_list = []
            masks = objectwise_mask[i_class] # obj-wise masks of objects of i-th class
            for mask in masks:
                obj = ForegroundObject(
                    rep_image=self,
                    mask=mask,
                    i_class=i_class,
                    class_name=class_name
                )        
                obj_list.append(obj)
            self.objects.update({i_class: obj_list})

        self.classwise_mask = classwise_mask
        self.objectwise_mask = objectwise_mask
        self.orig_image = self.image.detach().clone() # make copy

    def to(self, device, *, mask=False) -> None:
        self.image = self.image.to(device)
        self.device = self.image.device
        if mask:
            self.classwise_mask = self.classwise_mask.to(device)
            for i_class, masks in self.objectwise_mask.items():
                self.objectwise_mask.update({i_class: masks.to(device)})

    def set_image(self, new_image: torch.Tensor) -> None:
        """set a new image as the object's image attribute and make a new cropped image.
        new_image must be consistent with the old one in spatial information, i.e. you will
        get a nonsense result if you pass an image which went through data augmentation 
        where spatial information is not preserved.
        """
        assert new_image.size() == self.image.size(), 'cannot set a tensor of incompatible size'
        self.image[:] = new_image # substitute in-place

    @contextlib.contextmanager
    def image_as(self, tmp_image):
        """context manager to set a temporary image

        The passed image must be \"spatially compatible\" with the original image. 
        You will get a broken result if **tmp_image** is output of data augmentation
        where coordinates are not preserved.
        """
        try:
            self.set_image(tmp_image)
            yield
        finally:
            self.set_image(self.orig_image)

    def get_objects(self, **kwargs):
        keys = ['i_class', 'class_name']
        if not kwargs:
            return list(itertools.chain.from_iterable(self.objects.values()))
            # objects = []
            # for i_class in range(len(self.classes)):
            #     objects += self.get_objects(i_class=i_class)
            # return objects

        if len(kwargs) != 1 or all([not key in kwargs for key in keys]):
            raise ValueError(f'{self.__class__.__name__}.get_objects requires exactly 1 keyword argument: i_class or class_name')
        i_class = kwargs.get('i_class')
        if i_class is None:
            i_class = self.classes.index(kwargs['class_name'])
        return self.objects[i_class]

    def __eq__(self, other: RepImage):
        return self.p_image.samefile(other.p_image) and self.p_mask.samefile(other.p_mask)


@dataclasses.dataclass
class ForegroundObject:
    # ToDo: rename __random_rotate to __random_flip_and_rotate
    __random_rotate: ClassVar[Callable] = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(degrees=180, expand=True, fill=0)
    ])
    # __random_rotate: ClassVar[Callable] = torchvision.transforms.RandomRotation(degrees=180, expand=True, fill=0)
    __bbox_formats: ClassVar[List[str]] = ['pascal_voc', 'albumentations', 'coco', 'yolo']

    rep_image: RepImage
    mask: torch.Tensor
    i_class: int
    class_name: str

    bbox: dict = dataclasses.field(init=False)
    image_cropped: torch.Tensor = dataclasses.field(init=False)
    mask_cropped: torch.Tensor = dataclasses.field(init=False)

    def __post_init__(self):
        self.bbox = get_bbox(self.mask)
        top, bottom, left, right = self.bbox['top'], self.bbox['bottom'], self.bbox['left'], self.bbox['right']
        self.image_cropped = self.crop(self.rep_image.image) # self.image[:, top:bottom, left:right]
        self.mask_cropped = self.crop(self.mask) # self.mask[top:bottom, left:right]

    def crop(self, tensor):
        top, bottom, left, right = self.bbox['top'], self.bbox['bottom'], self.bbox['left'], self.bbox['right']
        tensor = tensor.transpose(0, -2).transpose(1, -1)
        tensor = tensor[top:bottom, left:right]
        tensor = tensor.transpose(1, -1).transpose(0, -2)
        return tensor

    def bbox_to(self, format):
        if format not in self.__bbox_formats:
            raise ValueError(f'invalid format "{format}" was given')
        func = getattr(self, 'bbox_to_' + format)
        return func()

    def bbox_to_pascal_voc(self):
        return bbox_to_pascal_voc(self.bbox)

    def bbox_to_albumentations(self):
        return bbox_to_albumentations(self.bbox, image_width=self.rep_image.width, image_height=self.rep_image.height)

    def bbox_to_coco(self):
        return bbox_to_coco(self.bbox)

    def bbox_to_yolo(self):
        return bbox_to_yolo(self.bbox, image_width=self.rep_image.width, image_height=self.rep_image.height)

    def random_place(self, background, return_bbox=True, return_mask=False, scale_jitter=True):
        """Randomly place the object on the given background image. This is an in-place operation.
        
        Parameters
        ----------
        background: array or tensor
          An image on which the object is placed
        """
        image_and_mask_cropped = torch.cat(
            [self.image_cropped, torch.unsqueeze(self.mask_cropped, 0)]
        )
        if scale_jitter:
            _, crop_height, crop_width = self.image_cropped.size()
            scale = np.random.uniform(0.1, 2.0) # Ghiasi, G., Cui, Y., Srinivas, A., Qian, R., Lin, T.-Y., Cubuk, E. D., Le, Q. V., & Zoph, B. (2021). Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr46437.2021.00294
            resized_height, resized_width = int(crop_height * scale), int(crop_width * scale)
            resized = torchvision.transforms.functional.resize(
                image_and_mask_cropped, 
                size=(resized_height, resized_width)
            )
            rotated = self.__random_rotate(resized)
        else:
            rotated = self.__random_rotate(image_and_mask_cropped)
        image_cropped, mask_cropped = torch.split(rotated, [3, 1], dim=0)
        mask_cropped = torch.squeeze(mask_cropped, dim=0).to(torch.bool)
        _, h, w = image_cropped.size() # size of cropped region after rotation
        
        top = np.random.randint(0, self.rep_image.height - h)
        left = np.random.randint(0, self.rep_image.width - w)
        background[:, top:top+h, left:left+w][:, mask_cropped] = image_cropped[:, mask_cropped]
        
        # bounding box info
        bbox = get_bbox(mask_cropped)
        bbox['top'] += top
        bbox['bottom'] += top
        bbox['left'] += left
        bbox['right'] += left

        # mask info
        mask_info = dict(mask_cropped=mask_cropped, top=top, left=left, i_class=self.i_class)

        if return_bbox:
            if return_mask:
                return bbox, mask_info
            return bbox
        if return_mask:
            return mask_info


class ObjectDatabase:
    def __init__(self, classes: List[str]):
        self.classes = classes
        self.n_class = len(self.classes)
        self.table = [[] for _ in self.classes] # self.table[i_class] is a list of ForegroundObjects belonging to the i_class-th class
        self.rep_images = []
        self.orig_images = []
        self.stats = DatabaseStats(self)

    def get(self, key: Union[int, str]=None):
        try:
            if key is None: # return objects of the all classes
                return list(itertools.chain.from_iterable(self.table))
            elif isinstance(key, (int, np.int64)): # key is i_class
                return self.table[key]
            elif isinstance(key, str): # key is class_name
                i_class = self.classes.index(key)
                return self.table[i_class]
            raise TypeError(f'{self.__class__.__name__}.get(): Key of an invalid type (key={key}: {type(key)})')
        except Exception as e:
            print(f'{self.__class__.__name__}.get(): An exception occured when accessing the objects of class {key}:')
            raise e

    def add(self, obj: Union[ForegroundObject, RepImage, FieldVideo]):
        if isinstance(obj, (RepImage, FieldVideo)):
            for obj in obj.get_objects():
                self.add(obj)
                # self.table[i_class] += objects
        elif isinstance(obj, ForegroundObject):
            i_class = obj.i_class
            self.table[i_class].append(obj)
            if obj.rep_image not in self.rep_images:
                self.rep_images.append(obj.rep_image)
                self.orig_images.append(obj.rep_image.orig_image)
        else:
            raise TypeError(f'Argument of invalid type {type(obj)} was given')
        
    def __repr__(self):
        n_obj = []
        for i_class, class_name in enumerate(self.classes):
            objects = self.get(i_class)
            n_obj.append(len(objects))
        return '\n'.join(
            [f'{self.__class__.__name__}['] + 
            [f'    {class_name}:' + '\t' + f'{n_obj[i_class]} objects' for i_class, class_name in enumerate(self.classes)] + 
            ['    ' + '-'*25] + 
            ['    all:' + '\t' + f'{sum(n_obj)} objects'] + 
            [']']
        )

    @contextlib.contextmanager           
    def set_images_temporarily(self, tmp_images):
        """context manager to set temporary rep-images

        The passed images must be \"spatially compatible\" with the original images. 
        You will get a broken result if **tmp_images** are output of data augmentation
        where coordinates are not preserved (i.e. augmentation which is not pixel-wise).
        """
        assert len(tmp_images) == len(self.orig_images)
        try:
            for rep_image, tmp_image in zip(self.rep_images, tmp_images):
                rep_image.set_image(tmp_image)
            yield
        finally:
            for rep_image, orig_image in zip(self.rep_images, self.orig_images):
                rep_image.set_image(orig_image)

    def random_iter(self, *, total, p, min_n_obj):
        return DatabaseRandomIterator(self, total=total, p=p, min_n_obj=min_n_obj)


class DatabaseStats:
    __agg_func = {
        'mean': np.mean, 
        'std': lambda data: np.std(data, ddof=1),
        'raw': lambda data: data,
        'sum': np.sum
    }

    def __init__(self, database):
        self.database = database

    def n_obj(self, *, pivot):
        if pivot == 'class':
            return [len(self.database.get(i_class) for i_class in range(self.database.n_class))]
        elif pivot == 'image': 
            return [len(rep_image.get_objects()) for rep_image in self.database.rep_images]
        raise ValueError(f'expected "class" or "image" as pivot, got {pivot}')

    def __call__(self, name: str, mode=[], **kwargs):
        data_getter = getattr(self, name)
        data = data_getter(**kwargs)
        ret = []
        for mode in mode:
            stat = self.__agg_func[mode](data)
            ret.append(stat)
        return tuple(ret)


class DatabaseRandomIterator:
    def __init__(self, database: ObjectDatabase, *, total: int, p: List[float]=None, min_n_obj=3):
        self.database = database
        self.n_class = database.n_class
        self.min_n_obj = min_n_obj
        self.validate_database() 
        self.counter = 0
        self.total = total
        if p is None:
            self.get_candidate_objects = self._get_candidate_objects_non_class_aware # randomly sample an object regardless of its class
        else:
            self.get_candidate_objects = self._get_candidate_objects_class_aware # randomly sample a class first, then randomly sample an object from that class
            self.p = p # class probabilities
        self.rng = np.random.default_rng()

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter >= self.total:
            raise StopIteration
        objects = self.get_candidate_objects()
        obj = self.rng.choice(objects)
        self.counter += 1
        return obj

    def _get_candidate_objects_non_class_aware(self):
        return self.database.get()

    def _get_candidate_objects_class_aware(self):
        i_class = self.rng.choice(range(self.n_class), p=self.p)
        return self.database.get(i_class)

    def validate_database(self):
        for i_class in range(self.n_class):
            objects = self.database.get(i_class)
            n_obj = len(objects)
            if n_obj < self.min_n_obj:
                class_name = self.database.classes[i_class]
                raise ValueError(f'database failed in validation: class "{class_name}" has too few objects ({n_obj} < {self.min_n_obj})')


class BaseLabel:
    """Base class of YoloLable, MaskLabel(, PascalVOCLabel, CocoLabel).
    """
    def __init__(self, image: Union[torch.Tensor, np.ndarray]=None, *, image_width:int=None, image_height:int=None, requires_size:bool=True):
        self._validate_args(image=image, image_width=image_width, image_height=image_height)
        if requires_size:
            if isinstance(image, torch.Tensor):
                self.image_width = float(image.shape[-1])
                self.image_height = float(image.shape[-2])
            elif isinstance(image, np.ndarray):
                self.image_width = float(image.shape[-2])
                self.image_height = float(image.shape[-3])
            elif image is None:
                self.image_width = image_width
                self.image_height = image_height
            else:
                raise TypeError('image must be either tensor or ndarray')
    
    @staticmethod
    def _validate_args(*, image, image_width, image_height):
        if image is None:
            assert image_width is not None and image_height is not None
        else:
            assert image_width is None and image_height is None


class YoloLabel(BaseLabel):
    """YOLO format bounding box annotation.
    """
    def __init__(self, image: Union[torch.Tensor, np.ndarray]=None, *, image_width:int=None, image_height:int=None, min_area:float=None):
        super().__init__(image=image, image_width=image_width, image_height=image_height, requires_size=True)
        self.bboxes = []
        self.bboxes_voc = []
        if min_area is None:
            min_area = 0.0
        self.min_area = min_area

    @staticmethod
    def _validate_args(*, image, image_width, image_height):
        if image is None:
            assert image_width is not None and image_height is not None
        else:
            assert image_width is None and image_height is None

    def add(self, *, obj: ForegroundObject=None, i_class: int=None, bbox: dict=None):
        """call with signature of add(obj=obj) or add(i_class=i_class, bbox=bbox)
        """
        if obj is not None:
            assert i_class is None and bbox is None
            x_center, y_center, width, height = obj.bbox_to_yolo()    
            i_class = obj.i_class
        else:
            assert i_class is not None and bbox is not None
            x_center, y_center, width, height = bbox_to_yolo(bbox, image_width=self.image_width, image_height=self.image_height)

        area = width * height
        if area <= self.min_area:
            warnings.warn(f'ignore a bounding box whose area is {area} <= min_area')
            return

        self.bboxes.append(
            OrderedDict(
                i_class=i_class, 
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height
            )
        )
        self.bboxes_voc.append(bbox)

    @staticmethod
    def parse_line(line):
        vals = line.split()
        if len(vals) == 6:
            i_class, x_center, y_center, width, height, conf = vals
        elif len(vals) == 5:
            if float(vals[0]).is_integer():
                i_class, x_center, y_center, width, height = vals
            else:
                x_center, y_center, width, height, conf = vals
        elif len(vals) == 4:
            x_center, y_center, width, height = vals
        i_class = int(i_class)
        x_center = float(x_center)
        y_center = float(y_center)
        width = float(width)
        height = float(height)
        # conf = float(conf)
        # genertic situation is not fully considered yet
        return OrderedDict(i_class=i_class, x_center=x_center, y_center=y_center, width=width, height=height)

    @classmethod
    def load(cls, fname, *, image=None, image_width=None, image_height=None):
        path = pathlib.Path(fname)
        if path.is_dir():
            return [cls.load(p, image_width=image_width, image_height=image_height) for p in path.glob('*.txt')]

        label = cls(image=image, image_width=image_width, image_height=image_height)
        with open(path) as f:
            for line in f:
                bbox = cls.parse_line(line)
                label.bboxes.append(bbox)
                bbox = {k: v for k, v in bbox.items() if k != 'i_class'}
                bbox_voc = yolo_to_pascal_voc(bbox, image_width=label.image_width, image_height=label.image_height)
                label.bboxes_voc.append(bbox_voc)
        return label

    def save(self, fname):
        with open(fname, 'w') as f:
            for label in self.bboxes:
                row = ' '.join([str(value) for value in label.values()])
                row += '\n'
                f.write(row)

    def to_tensor(self):
        return torch.tensor([[bbox['left'], bbox['top'], bbox['right'], bbox['bottom']] for bbox in self.bboxes_voc])

    def class_name_list(self, classes):
        return [classes[bbox['i_class']] for bbox in self.bboxes]


class MaskLabel(BaseLabel):
    """Pixel-wise label of semantic masks.
    """
    def __init__(self, image: Union[torch.Tensor, np.ndarray]=None, *, image_width:int=None, image_height:int=None):
        super().__init__(image=image, image_width=image_width, image_height=image_height, requires_size=True)
        self.instance_masks = []
        self.n_class = 0

    def add(self, mask_info: dict):
        self.instance_masks.append(mask_info)
        i_class = mask_info['i_class']
        if i_class >=  self.n_class:
            self.n_class = i_class + 1

    def to_voc(self, rgb=True):
        ret = np.zeros((int(self.image_height), int(self.image_width)), dtype=int)
        for mask_info in self.instance_masks:
            mask_cropped = mask_info['mask_cropped'].numpy()
            top = mask_info['top']
            left = mask_info['left']
            i_class = mask_info['i_class']
            h, w = mask_cropped.shape
            ret[top:top+h, left:left+w][mask_cropped] = i_class + 1
        if rgb:
            ret = utils.PASCAL_VOC_CMAP[ret]
        return utils.numpyimage_to_tensor(ret)

    def save(self, fname):
        voc = self.to_voc()
        utils.save_image(voc, fname)

def foreground_augmentation(field_video: FieldVideo, intensity: float):
    """apply data augmentation to all the representative images of the given field video.

    Note
    ----
    An error will be raised if any of the rep-images is on a cuda device because 
    Albumentations cannot deal with images on GPUs.
    """
    images = []
    transform = utils.make_foreground_augmentation(p=intensity)
    for rep_image in field_video.rep_images:
        transformed = transform(image=utils.tensorimage_to_numpy(rep_image.image))
        image = transformed['image']
        image = utils.numpyimage_to_tensor(image)
        images.append(image)
    return images

def background_augmentation(frame: Union[np.ndarray, torch.Tensor], intensity: float, large_scale_jitter=False):
    """apply data augmentation to the given frame of a background video.

    Note
    ----
    returns torch.Tensor on cpu device.
    """
    if isinstance(frame, torch.Tensor):
        frame = utils.tensorimage_to_numpy(frame)
    if large_scale_jitter:
        frame = utils.random_scale_jitter(frame, mode='large')
    transform = utils.make_background_augmentation(p=intensity)
    transformed = transform(image=frame)
    image = transformed['image']
    image = utils.numpyimage_to_tensor(image)
    return image

# class HasNoRepImage(Exception):
#     """raised when a given FieldVideo object has no RepImage objects in it.
#     """


def make_composite_image(
        *, 
        database: ObjectDatabase,
        # field_video: FieldVideo, 
        back_video: BackgroundVideo, 
        pathes: namedtuple, 
        prob: Union[List[float], None], 
        suffix: str, 
        bbox: bool, 
        # device, 
        augment_intensity: float, 
        index: int,
        n_obj_mean: float, 
        n_obj_std: float,
        scale_jitter: bool = True
    ):
    frame = back_video.random_read(device='cpu', noexcept=True, as_numpy=True) # must load into CPU for Albumentations
    frame, label, mask = synthesize(frame=frame, database=database, prob=prob, augment_intensity=augment_intensity, n_obj_mean=n_obj_mean, n_obj_std=n_obj_std, scale_jitter=scale_jitter)# , device=device)

    # save as files
    output_stem = f'composite_back_{back_video.stem}_{index}'
    output_image_name = pathes.output_images_dir / (output_stem + utils.with_dot(suffix))
    output_label_name = pathes.output_labels_dir / (output_stem + '.txt')
    output_mask_name = pathes.output_masks_dir / (output_stem + '.jpg')
    utils.save_image(frame, output_image_name)
    label.save(output_label_name)
    mask.save(output_mask_name)

    if bbox:
        output_labeled_image_name = pathes.output_labeled_images_dir / output_image_name.name
        utils.save_labeled_image(frame, label, output_labeled_image_name, database.classes)

def synthesize(*, frame, database, augment_intensity: float, prob: None, n_obj_mean: float, n_obj_std: float, scale_jitter=True): # , device='cpu'):
    """randomly place foreground objects onto the given frame in-place

    frame: a frame taken from a background video
    field_video: a field video
    prob: the probablity that a foreground object will be picked from each class

    Note
    ----
    - frame and all the rep-images of field_video are assume to be on cpu device at the call of this function.
    - device control is not implemented yet.
    """
    n_obj = int(np.random.normal(loc=n_obj_mean, scale=n_obj_std)) # number of the objects scattered in the frame
    n_class = database.n_class
    yololabel = YoloLabel(frame)
    masklabel = MaskLabel(frame)
    rep_images_aug = foreground_augmentation(database, augment_intensity)
    frame_aug = background_augmentation(frame, augment_intensity)
    
    with database.set_images_temporarily(rep_images_aug):
        for obj in database.random_iter(total=n_obj, p=prob, min_n_obj=3):
            bbox, mask_info = obj.random_place(frame_aug, return_bbox=True, return_mask=True, scale_jitter=scale_jitter)
            yololabel.add(i_class=obj.i_class, bbox=bbox)
            masklabel.add(mask_info)

    return frame_aug, yololabel, masklabel

def make_rotated_rep_image(rep_image: RepImage, *, pathes: namedtuple, bbox: bool, suffix: str, pbar: tqdm, min_area:float=None):
    n_class = len(rep_image.classes)
    concat = torch.cat([rep_image.image, rep_image.classwise_mask])
    
    for degree in range(360):
        image, label = _make_rotated_rep_image_impl(concat, degree=degree, n_class=n_class, min_area=min_area)
        output_stem = f'{rep_image.p_image.stem}_{degree:03}degrees'
        output_image_name = pathes.output_images_dir / (output_stem + utils.with_dot(suffix))
        output_label_name = pathes.output_labels_dir / (output_stem + '.txt')
        utils.save_image(image, output_image_name)
        label.save(output_label_name)
        if bbox:
            output_labeled_image_name = pathes.output_labeled_images_dir / output_image_name.name
            utils.save_labeled_image(image, label, output_labeled_image_name, rep_image.classes)
        pbar.update(1)
            
def _make_rotated_rep_image_impl(concat, *, degree, n_class, min_area=None):
    """image: Tensor (C, H, W)
    classwise_mask: Tensor (n_class, H, W)
    """
    rotated = torchvision.transforms.functional.rotate(
        concat, degree
    )
    image, classwise_mask = torch.split(rotated, [3, n_class], dim=0)
    objectwise_mask = make_objectwise_mask(classwise_mask, n_class)
    label = YoloLabel(image, min_area=min_area)
        
    # bounding box info
    for i_class in range(n_class):
        bboxes = get_bbox(objectwise_mask[i_class])
        for bbox in bboxes:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                label.add(i_class=i_class, bbox=bbox)
                
    return image, label

# def make_yolo_labels_from_masks(field_videos, p_output):
#     p_output = pathlib.Path(p_output)
#     for field_video in field_videos:
#         rep_images = field_video.rep_images
#         if not rep_images:
#             continue
#         for rep_image in rep_images:
#             output_path = pathlib.Path(p_output)
#             label = YoloLabel(rep_image.image)
#             for obj in rep_image.objects:
#                 label.add(obj)
#             label.save()
