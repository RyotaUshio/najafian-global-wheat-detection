import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
import re
import argparse
import time

_parse_fname_pattern = re.compile(r'(.+)_(\d{4})(_class\d)?\.(.+)')
_parse_fname_i_class_pattern = re.compile(r'_class(\d+)')

def parse_fname(fname):
    m = _parse_fname_pattern.match(fname)
    if m is None:
        raise Exception(f'An invalid file name "{fname}" was given')
    ret = {k: v for k, v in zip(['stem', 'time', 'i_class', 'extension'], m.groups())}

    i_class = ret['i_class']
    if i_class is not None:
        i_class = _parse_fname_i_class_pattern.match(i_class).groups()[0]
        i_class = int(i_class)
        ret['i_class'] = i_class

    return ret

def get_image_path(p_mask, p_rep_dir, extension):
    parsed = parse_fname(p_mask.name)
    if not extension.startswith('.'):
        extension = '.' + extension
    image_name = parsed['stem'] + '_' + parsed['time'] + extension
    return p_rep_dir / image_name

def imread(fname):
    return torchvision.io.read_image(fname)

def get_classes(p_labels):
    classes = []
    with open(p_labels) as f:
        for line in f:
            line = line.rstrip()
            if line not in ['__ignore__', '_background_']:
                classes.append(line)
    return classes

def mask_3_channels(mask):
    return torch.stack([mask] * 3)

def tensorimage_to_numpy(tensor):
    """(C x H x W) to (H x W x C)
    """
    return tensor.numpy().transpose((1, 2, 0))

class foreground_obj:
    _affine = torchvision.transforms.RandomAffine(
        degrees=180,
        translate=(0.4, 0.4)
    )
    _totensor = torchvision.transforms.ToTensor()
    
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask
        self.mask_rgb = mask_3_channels(self.mask)
        self.height, self.width = self.mask.shape
        
        # get the center of gravity
        indices = np.indices(self.mask.shape)[:, self.mask]
        g_x = np.mean(indices[1])
        g_y = np.mean(indices[0])
        self.g = torch.tensor([g_x, g_y])
        
        self.centered_image = self.centerize(self.image)
        self.centered_mask = self.centerize(self.mask)
        self.centered_mask_rgb = mask_3_channels(self.centered_mask)

    def centerize(self, img):
        # move to the center
        center = torch.tensor([self.width, self.height]) * 0.5
        translate = center - self.g
        return torchvision.transforms.functional.affine(
            img=img,
            angle=0.0,
            translate=translate,
            scale=1.0,
            shear=0
        )

    def random_place(self, background):
        """Randomly place the object on the given background image.
        
        Parameters
        ----------
        background: array
          An image on which the object is placed
        """
        image = self._affine(self.centered_image)
        mask = self._affine(self.centered_mask)
        mask_rgb = maks_3_channels(mask)
        return torch.where(mask_rgb, image, background)
        
    def argument(self):
        pass
        
    def make_bbox(self):
        pass

    def show(self, center=False):
        mask = self.centered_mask_rgb if center else self.mask_rgb
        image = self.centered_image if center else self.image
        masked_image = torch.where(mask, image, 255)
        masked_image = tensorimage_to_numpy(masked_image)
        plt.imshow(masked_image)
        plt.scatter(*self.g, marker='x', color='r')
    
class set_foreground_obj:
    def __init__(self, image_path, mask_path, classes=None):
        self.image_path = image_path
        self.mask_path = mask_path

        parsed = parse_fname(self.mask_path.name)
        
        self.stem = parsed['stem'] # filename without extensions
        self.time = time.strptime(parsed['time'], '%M%S') # time elapsed since the start of the vid
        self.i_class = parsed['i_class']
        self.class_name = None
        if classes is not None:
            self.class_name = classes[self.i_class - 1] # i_class is 1-origin
        self.extension = parsed['extension'] # extension of image file e.g. 'png', 'jpeg', etc.

        # read mask & image files
        self.masks = torch.as_tensor(np.load(self.mask_path))
        self.image = imread(self.image_path)

        self.objects = []
        for mask in self.masks:
            fg = foreground_obj(image=self.image, mask=mask)
            self.objects.append(fg)

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, key):
        return self.objects[key]

    def __iter__(self):
        return iter(self.objects)
    
    
class background:
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mask_dir', help='input directory where mask files named [image_name]_class[class_id].npy are placed')
    parser.add_argument('rep_dir', help='input directory where representative image files are placed. Note that a filename must have the format of [video name]_[%%M%%S] where %%M and %%S denote minute and second as decimal numbers, respectedly')
    parser.add_argument('video_dir', help='input directory where video files of chestnut fields are placed')
    parser.add_argument('background_dir', help='input directory where background video files are placed')
    parser.add_argument('labels', help='labels file')
    parser.add_argument('-e', '--extension', help='extension(s) of image files', nargs='?', default='.png')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    p_mask_dir = pathlib.Path(args.mask_dir)
    p_rep_dir = pathlib.Path(args.rep_dir)
    classes = get_classes(args.labels)

    # fg: foreground images
    # bg: background images

    i = 0
    for p_mask in p_mask_dir.glob('*.npy'):
        p_image = get_image_path(p_mask=p_mask, p_rep_dir=p_rep_dir, extension=args.extension)
        objs = set_foreground_obj(p_image, p_mask, classes) # a set of foreground objects contained in p_image
        for obj in objs:
            obj.show(center=True)
            plt.clf()
        
if __name__ == '__main__':
    main()

