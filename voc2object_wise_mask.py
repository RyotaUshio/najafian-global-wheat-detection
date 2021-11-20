import numpy as np
from skimage import measure
import pathlib
import os
import argparse

def main():
    plt.ion()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='input directory in which VOC-format labels are stored as .py files')
    parser.add_argument('output_dir', help='output directory in which .npy files representing object-wise masks will be saved')
    parser.add_argument('labels', help='labels file')
    args = parser.parse_args()

    classes = []
    with open(args.labels) as f:
        for line in f:
            line = line.rstrip()
            if line not in ['__ignore__', '_background_']:
                classes.append(line)
    n_class = len(classes)
                
    masks_each_class = {class_name: [] for class_name in classes}
    
    for path in pathlib.Path(args.input_dir).glob('*.npy'):
        labels = np.load(path)
        for i_class, class_name in enumerate(classes):
            object_wise_labels = measure.label(labels == i_class + 1) # labels connected components
            object_id = np.unique(object_wise_labels)[1:] # ignore background = 0
            masks = (object_wise_labels == object_id.reshape(-1, 1, 1))
            masks_each_class[class_name].append(masks)

    for class_name in classes:
        masks = np.concatenate(masks_each_class[class_name], axis=0)
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        output_filename = os.path.join(args.output_dir, class_name)
        np.save(output_filename, masks)
            
if __name__ == '__main__':
    main()
