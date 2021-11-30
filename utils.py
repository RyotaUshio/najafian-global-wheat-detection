import torch
import torchvision
import cv2
import albumentations as A
import pathlib
import time
import sys
import os

# acceptable image/video suffixes: same as YOLOv5
IMG_FORMATS = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo']
VID_FORMATS = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

def with_dot(suffix):
    """ensure that a suffix starts with a period.
    """
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    return suffix

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
    return tensor.cpu().numpy().transpose((1, 2, 0))

def adjust_opencv(image):
    # OpenCV reads image in (B, G, R(, A)) order
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image[:, :, [2, 1, 0]] 
    return image

def numpyimage_to_tensor(ndarr, device=None, opencv=False):
    if opencv:
        ndarr = adjust_opencv(ndarr)
    ndarr = ndarr.transpose((2, 0, 1)) # (H, W, C) -> (C, H, W)
    tensor = torch.as_tensor(ndarr)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def read_image(path, device=None):
    image = torchvision.io.read_image(
      str(path),
      mode=torchvision.io.ImageReadMode.RGB
    )
    if device is not None:
        image = image.to(device)
    return image

class FrameCannotBeLoaded(Exception):
    """raised when a frame cannot be loaded from a cv2.VideoCapture object.
    """

def read_frame(cap, i_frame, device=None, as_numpy=False):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    opened, frame = cap.read()
    if not opened:
        raise FrameCannotBeLoaded
    if as_numpy:
        frame = adjust_opencv(frame)
    else:
        frame = numpyimage_to_tensor(frame, device=device, opencv=True)
    return frame
    
def save_image(tensor, fname):
    tensor = tensor.cpu()
    if isinstance(tensor, torch.ByteTensor):
        tensor = tensor.float()
        tensor /= 255.0
    torchvision.utils.save_image(tensor, fname)

def save_labeled_image(image, label, output_labeled_image_name, classes):
    save_image(
        torchvision.utils.draw_bounding_boxes(
            image=image.cpu(),
            boxes=label.to_tensor(),
            labels=label.class_name_list(classes)
        ),
        output_labeled_image_name,
    )

def make_logger(*, verbose):
    def log(*args, **kwargs):
        nonlocal verbose
        if verbose:
            print(*args, **kwargs)
    return log

def make_strong_augmentation(*, bbox_format=None, min_area=0.0, min_visibility=0.0, p=0.5):
    if bbox_format is None:
        if not (min_area == min_visibility == 0.0):
            raise ValueError('bbox_format is required')
        kwargs = {}
    else:
        kwargs = dict(bbox_params=A.BboxParams(
            format=bbox_format,
            min_area=min_area,
            min_visibility=min_visibility, 
            label_fields=['i_class']
        ))
    
    transform = A.Compose([
        A.Blur(p=p),
        A.ChannelShuffle(p=p),
        A.CLAHE(p=p),
        A.ColorJitter(p=p),
        A.Equalize(p=p),
        A.FancyPCA(p=p),
        A.Flip(p=p),
        A.GaussianBlur(p=p),
        A.GaussNoise(p=p),
        A.GlassBlur(p=p),
        A.HorizontalFlip(p=p),
        A.HueSaturationValue(p=p),
        A.InvertImg(p=p),
        A.MedianBlur(p=p),
        A.MotionBlur(p=p), # ADDED!
        A.MultiplicativeNoise(p=p),
        A.Posterize(p=p),
        A.RandomBrightnessContrast(p=p),
        A.RandomSnow(p=p),
        A.RandomSunFlare(p=p),
        A.RGBShift(p=p),
        A.Solarize(p=p),
        A.ToGray(p=p),
        A.VerticalFlip(p=p)],
        **kwargs
    )
    return transform

def make_foreground_augmentation(p=0.5):
    # とりあえず、strong augmentationからpixel-wiseでないものを除く。
    transform = A.Compose([
        A.Blur(p=p),
        # A.ChannelShuffle(p=p),
        A.CLAHE(p=p),
        A.ColorJitter(p=p),
        A.Equalize(p=p),
        A.FancyPCA(p=p),
        A.GaussianBlur(p=p),
        A.GaussNoise(p=p),
        A.GlassBlur(p=p),
        A.HueSaturationValue(p=p),
        # A.InvertImg(p=p),
        A.MedianBlur(p=p),
        A.MotionBlur(p=p),
        A.MultiplicativeNoise(p=p),
        A.Posterize(num_bits=6, p=p),
        A.RandomBrightnessContrast(p=p),
        # A.RandomSnow(p=p),
        # A.RandomSunFlare(p=p),
        A.RGBShift(p=p),
        # A.Solarize(p=p),
        A.ToGray(p=p)]
    )
    return transform

def make_background_augmentation(p=0.5):
    transform = A.Compose([
        A.Blur(p=p),
        A.ChannelShuffle(p=p),
        A.ChannelDropout(p=p),
        A.CLAHE(p=p),
        A.ColorJitter(p=p),
        A.Equalize(p=p),
        A.FancyPCA(p=p),
        A.Flip(p=p),
        A.GaussianBlur(p=p),
        A.GaussNoise(p=p),
        A.GlassBlur(p=p),
        A.HorizontalFlip(p=p),
        A.HueSaturationValue(p=p),
        # A.InvertImg(p=p),
        A.MedianBlur(p=p),
        A.MotionBlur(p=p), # ADDED!
        A.MultiplicativeNoise(p=p),
        A.Posterize(num_bits=4, p=p),
        A.RandomBrightnessContrast(p=p),
        A.RandomSnow(p=p),
        A.RandomSunFlare(p=p),
        A.RGBShift(p=p),
        A.ToGray(p=p),
        A.VerticalFlip(p=p)]
    )
    return transform