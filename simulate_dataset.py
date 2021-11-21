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
import time

_parse_fname_pattern = re.compile(r'(.+)_(\d{4})?(_class\d)?\.(.+)')
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

from torchvision.transforms.functional import to_tensor

def imread(fname):
    return torchvision.io.read_image(
      str(fname),
      mode=torchvision.io.ImageReadMode.RGB
    )
    

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
        delta = center - self.g
        delta_x, delta_y = delta.to(int).tolist()
        return torch.roll(img, shifts=(delta_x, delta_y), dims=(-1, -2))

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
        """新しいオブジェクトを返すように！！"""
        
    def make_bbox(self):
        pass

    def show(self, center=False):
        mask = self.centered_mask_rgb if center else self.mask_rgb
        image = self.centered_image if center else self.image
        masked_image = torch.where(mask, image, torch.tensor(255, dtype=torch.uint8))
        masked_image = tensorimage_to_numpy(masked_image)
        plt.imshow(masked_image)
        if not center:
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

    @classmethod
    def merge(cls, *args):
        ret = []
        for arg in args:
            assert isinstance(arg, cls)
            ret += arg.objects
        return ret


def synthesize(frame, objs, classes, prob, output_name):
    rng = np.random.default_rng()
    
    """nut(fine)
    nut(empty)
    burr
    burr+nut
    """
    n_obj = rng.normal(loc=6, scale=2) # ???
    
    for i in range(n_obj):
        class_name = rng.choice(classes, p=prob)
        objects = objs[class_name]
        obj = rng.choice(objects)
        obj = obj.argumented()
        frame = background_argumentation(frame)
        synthesized = obj.random_place(frame)
        torchvision.utils.save_image(synthesized, output_name)
        
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mask_dir', help='input directory where mask files named [image_name]_class[class_id].npy are placed')
    parser.add_argument('rep_dir', help='input directory where representative image files are placed. Note that a filename must have the format of [video name]_[%%M%%S] where %%M and %%S denote minute and second as decimal numbers, respectedly')
    parser.add_argument('video_dir', help='input directory where video files of chestnut fields are placed')
    parser.add_argument('background_dir', help='input directory where background video files are placed')
    parser.add_argument('output_dir', help='output directory where generated dataset will be exported. Note that two directories will be made under the output_dir, namely "images" and "labels"')
    parser.add_argument('labels', help='labels file')
    parser.add_argument('n_sample', help='number of samples of the dataset that will be generated synthetically', type=int)
    parser.add_argument('-p', '--probability', help='probability that a foreground object is chosen from each class when synthesizving dataset. This must have the same length as number of classes. Defaults to [1/n_class, 1/n_class, ...]', nargs='*', type=float)
    parser.add_argument('-e', '--extension', help='extension(s) of image files', nargs='?', default='.png')
    args = parser.parse_args()
    if not args.extension.startswith('.'):
        args.extension = '.' + args.extension
    return args

def main():
    args = parse_args()
    p_mask_dir = pathlib.Path(args.mask_dir)
    p_rep_dir = pathlib.Path(args.rep_dir)
    p_video_dir = pathlib.Path(args.video_dir)
    p_back_dir = pathlib.Path(args.background_dir)
    classes = get_classes(args.labels)
    n_class = len(classes)

    # class probabilities
    if args.probability:
        assert len(args.probability) == n_class
    else:
        args.probability = [1.0/n_class] * n_class
    prob = np.array(args.probablity)
    prob /= prob.sum()

    # stem = filename - suffix : corresponds to each video clip with foreground objects
    stems = [parse_fname(p_video.name)['stem'] for p_video in p_video_dir.glob('*.mp4')]
    obj_dict = {
        stem: {class_name: [] for class_name in classes}
        for stem in stems
    }

    # generate foreground objects and put them into obj_dict
    for p_mask in p_mask_dir.glob('*.npy'):
        p_image = get_image_path(p_mask=p_mask, p_rep_dir=p_rep_dir, extension=args.extension)
        set_obj = set_foreground_obj(p_image, p_mask, classes) # a set of foreground objects contained in p_image
        obj_dict[set_obj.stem][set_obj.class_name].extend(set_obj.objects)

    # simulate datasets for each pair of (a background video, set of foregound objects in a video)
    rng = np.random.default_rng()
    p_back_generator = p_back_dir.glob('*.mp4')
    n_back_video = len(list(p_back_generator))
    n_obj_video = len(stems)
    n_sample_per_pair = args.n_sample // (n_back_video * n_obj_video)
    n_sample_generated = 0
    
    for p_video in p_back_generator:

        # open background video
        cap = cv2.VideoCapture(str(p_video))
        if not cap.isOpened():
            raise Exception(f'Cannot open file {p_video}')
        n_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        for stem in stems:
            # randomly select a background frame from p_video
            idx_frame = rng.choice(
                range(n_frame),
                size=n_sample_per_pair
                replace=False
            )
            for i_frame in idx_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
                ret, frame = cap.read()
                assert ret, f'Could not read {i_frame}-th frame from the background video {p_video} with {n_frame} frames'
                synthesize(
                    frame=frame,
                    objs=obj_dict[stem],
                    classes=classes,
                    prob=prob,
                    output_name=os.path.join(args.output_dir, f'back_{p_video.stem}_rep_{stem}{args.extension}')
                )


    
        
        sec = cap.get(cv2.CAP_PROP_POS_MSEC) * 1000
        
        func(frame, sec, *args, **kwargs)
    
    cap.release()

        
if __name__ == '__main__':
    main()

