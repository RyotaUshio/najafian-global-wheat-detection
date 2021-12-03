import _make_dataset as mkdata
from utils import read_image, get_classes, save_labeled_image
import matplotlib.pyplot as plt

stem = 'synth_back_WIN_20211112_12_31_27_Pro_field_WIN_20211112_14_48_16_Pro_0'
image = read_image(f'datasets/20211121/composite/images/all/{stem}.jpeg')
label = mkdata.YoloLabel.load(
    f'yolov5/runs/val/composite20000/labels/{stem}.txt',
    image=image
)
classes = get_classes('datasets/20211121/labels.txt')
save_labeled_image(image, label, 'val.jpeg', classes)
