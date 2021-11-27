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
