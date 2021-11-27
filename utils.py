import torch
import torchvision
import cv2
import albumentations as A
import pathlib
import time
import sys
import os

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

def numpyimage_to_tensor(ndarr, device=None, opencv=False):
    if opencv:
        ndarr = ndarr[:, :, [2, 1, 0]] # OpenCV reads image in (B, G, R(, A)) order
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

def read_frame(cap, i_frame, device=None):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    opened, frame = cap.read()
    if not opened:
        raise RuntimeError(f'Could not read {i_frame}-th frame from the background video {p_video} with {n_frame} frames')
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
            string = fmt(i, n_total, time_elapsed, time_remaining, *args, **kwargs)
            if newline:
                string = string + os.linesep
            else:
                string = '\r' + string
            sys.stdout.write(string)
            sys.stdout.flush()
        if i >= n_total:
            if verbose and not newline:
                sys.stdout.write(os.linesep)
                sys.stdout.flush()
            raise StopIteration
    return counter

def make_logger(*, verbose):
    def log(*args, **kwargs):
        nonlocal verbose
        if verbose:
            print(*args, **kwargs)
    return log

def make_strong_augmentation(*, bbox_format=None, min_area=0.0, min_visibility=0.0):
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
        A.Blur(),
        A.ChannelShuffle(),
        A.CLAHE(),
        A.ColorJitter(),
        A.Equalize(),
        A.FancyPCA(),
        A.Flip(),
        A.GaussianBlur(),
        A.GaussNoise(),
        A.GlassBlur(),
        A.HorizontalFlip(),
        A.HueSaturationValue(),
        A.InvertImg(),
        A.MedianBlur(),
        A.MotionBlur(), # ADDED!
        A.MultiplicativeNoise(),
        A.Posterize(),
        A.RandomBrightnessContrast(),
        A.RandomSnow(),
        A.RandomSunFlare(),
        A.RGBShift(),
        A.Solarize(),
        A.ToGray(),
        A.VerticalFlip()],
        **kwargs
    )
    return transform

def make_foreground_augmentation():
    # とりあえず、strong augmentationからpixel-wiseでないものを除く。
    transform = A.Compose([
        A.Blur(),
        A.ChannelShuffle(),
        A.CLAHE(),
        A.ColorJitter(),
        A.Equalize(),
        A.FancyPCA(),
        # A.Flip(),
        A.GaussianBlur(),
        A.GaussNoise(),
        A.GlassBlur(),
        # A.HorizontalFlip(),
        A.HueSaturationValue(),
        A.InvertImg(),
        A.MedianBlur(),
        A.MotionBlur(), # ADDED!
        A.MultiplicativeNoise(),
        A.Posterize(),
        A.RandomBrightnessContrast(),
        A.RandomSnow(),
        A.RandomSunFlare(),
        A.RGBShift(),
        A.Solarize(),
        A.ToGray()],
        # A.VerticalFlip()]
    )
    return transform

def make_background_augmentation():
    return make_strong_augmentation()