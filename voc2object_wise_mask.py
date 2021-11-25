import numpy as np
from skimage import measure
import pathlib
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='input directory in which VOC-format labels are stored as .py files', type=pathlib.Path)
    parser.add_argument('output_dir', help='output directory in which .npy files representing object-wise masks will be saved', type=pathlib.Path)
    parser.add_argument('labels', help='labels file')
    args = parser.parse_args()
    return args

def get_classes(p_labels):
    classes = []
    with open(p_labels) as f:
        for line in f:
            line = line.rstrip()
            if line not in ['__ignore__', '_background_']:
                classes.append(line)
    return classes

def make_objectwise_mask(path, n_class):
    """path: path to .npy file which contains class-wise masks in the VOC format
    """
    labels = np.load(path)
    ret = dict()
    for i_class in range(n_class):
        objectwise_labels = measure.label(labels == i_class + 1) # labels connected components
        object_id = np.unique(objectwise_labels)[1:] # ignore background = 0
        masks = (objectwise_labels == object_id.reshape(-1, 1, 1))
        ret.update({i_class: masks})
    return ret

def main():
    args = parse_args()
    # make output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # read labels.txt and get class information
    classes = get_classes(args.labels)
    n_class = len(classes)
    
    for path in args.input_dir.glob('*.npy'):
        masks = make_objectwise_mask(path, n_class)
        for i_class in range(n_class):
            output_filename = args.output_dir / f'{path.stem}_class{i_class+1}'
            np.save(output_filename, masks[i_class])
            
if __name__ == '__main__':
    main()
