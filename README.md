# Image synthesis with Copy-Paste for Global Wheat Detection

This is an implementation of the method proposed in [Keyhan Najafian, Alireza Ghanbari, Ian Stavness, Lingling Jin, Gholam Hassan Shirdel, and Farhad Maleki. Semi-self-supervised Learning Approach for Wheat Head Detection using Extremely Small Number of Labeled Samples. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 1342-1351, 2021.](https://openaccess.thecvf.com/content/ICCV2021W/CVPPA/papers/Najafian_A_Semi-Self-Supervised_Learning_Approach_for_Wheat_Head_Detection_Using_Extremely_ICCVW_2021_paper.pdf)

Note that **it is not an official repo by the authors of that paper.**

## Requirements
Python>=3.6.0  
NumPy>=??  
PyTorch>=1.7  
OpenCV>=??  
Albumentation>=1.1.0

## Preparing datasets
1. Prepare your videos. Put video clips of backgrounds in `back` directory, fields in `field`.

2. Choose 1 or more "representative frames" from each video clip of fields. Put them in `rep` directory.

3. Make pixel-wise labels for each representative image, just like you would do for semantic segmentation.

4. Export the labels in Pascal VOC format and put them in `mask` directory. If you have made the labels with [labelme](https://github.com/wkentaro/labelme), you can export them with the following command:
    ```
    python labelme2voc.py [input_dir] dataset_voc --labels labels.txt
    ```
    where`input_dir` is where you've put the `.json` files generated by labelme.
    Now you've got `.npy` files in `dataset_voc/SegmentationClass`.
    Note that `labelme2voc.py` is taken from [the original labelme repo](https://github.com/wkentaro/labelme/blob/main/examples/semantic_segmentation/labelme2voc.py) and fixed a little bit to handle wider range of images.

5. Create a text file named `labels.txt` and list class names in it. For example, it will look like this if you are working on chestnut detection:
```
nut(fine)
nut(empty)
burr
burr+nut
```

6. Make sure your project directory is organized as below:

    ```
    project_root
    │   
    └───back
    │   │   background_video1.mp4
    │   │   background_video2.mp4
    │   │   ...
    │   
    └───field
    │   │   field_video1.mp4
    │   │   field_video2.mp4
    │   │	...
    │
    └───rep # representative frames taken from each field video
    │   │   field_video1_0031.png # [video_name]_[timestamp].png
    │   │   field_video1_0118.png
    │   │	field_video2_0005.png
    │   │	...
    │
    └───mask
    │   │   field_video1_0031.npy # [video_name]_[timestamp].npy
    │   │   field_video1_1018.npy
    │   │	field_video2_0005.npy
    │   │	...
    │
    └───labels.txt
    ```

5. Run the following to generate a dataset of composite images.
    ```
    python make_dataset.py composite --root [path/to/project_root] -o composite -n [number  of images to generate] --augment 0.05 --verbose
    ```
    `make_dataset.py` has more fine-grained control on the generated images. 
    Run `python make_dataset.py --help` for the details.

6. Run the following command to generate a dataset of rotated representative images.
    ```
    python make_dataset.py rotate --root [path/to/project_root] -o rotated --verbose
    ```
    
7. Run the following command to generate a dataset of randomly sampled frames of the field videos.
    ```
    mkdir -p field_frames/images
    python make_dataset_of_randomly_sampled_frames.py field/* [number of frames to be sampled] -o field_frames/images -v
    ```

8. Make `.yaml` files for YOLOv5 training with the datasets you've made.

## Train your model
10. Clone [YOLOv5 repo](https://github.com/ultralytics/yolov5) and install required packages.
    ```
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    pip install -r requirements.txt
    ```

11. Train YOLOv5 with the dataset of composite images.
    ```
    python train.py --img 640 --batch -1 --epochs 25 --data ../composite.yaml --weights yolov5m.pt --name composite
    ```

12. Fine-tune the model with the dataset of rotated representative images.
    ```
    python train.py --strong-augment 0.05 --hyp hyp.finetune.yaml --data ../rotate.yaml --name composite_finetune --rect --img 640 --batch-size -1 --epochs 5 --weights runs/train/composite/weights/last.pt
    ```
    `--strong-augment` flag is not available in the original version of YOLOv5. 
    Modify the code of `class Albumentations` in `yolov5/utils/augmentations.py` to enable the data augmentation that is referred to as "strong augmentation" in the paper.

14. Run `detect.py` to make pseudo labels for `field_frames` dataset with the last model you've trained.
    ```
    python detect.py --weights runs/train/composite_finetune/weights/last.pt --source ../field_frames/images --imgsz 640 --conf-thres 0.7 --save-txt --nosave --name composite_finetune_pseudo_labels
    mv runs/detect/composite_finetune_pseudo_labels/labels ../field_frames
    ```

15. Train the model further with the pseudo labels.
    ```
    python train.py --data ../field_frames.yaml --name composite_finetune_self_train --img 640 --batch-size -1 --epochs 25 --weights runs/train/composite_finetune/weights/last.pt
    ```
