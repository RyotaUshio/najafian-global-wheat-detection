# chestnut-detection

This is an implementation of the method proposed in
> Keyhan Najafian, Alireza Ghanbari, Ian Stavness, Lingling Jin, Gholam Hassan Shirdel, and Farhad Maleki. Semi-self-supervised Learning Approach for Wheat Head Detection using Extremely Small Number of Labeled Samples. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 1342-1351, 2021.

First, you have to conduct pixel-wise annotation, just like you would do in semantic segmentation (but for  only a few frames in the video).
Here we assume that you use [labelme](https://github.com/wkentaro/labelme).

After completing annotation, run the following command to export the labels in the VOC format.
```
python labelme2voc.py [input_dir] dataset_voc --labels labels.txt
```
Here, `input_dir` is where you've put the `.json` files generated by labelme. Note that `labelme2voc.py` is taken from [the original labelme repo](https://github.com/wkentaro/labelme/blob/main/examples/semantic_segmentation/labelme2voc.py) and fixed a little bit to handle wider range of images.

Now you've got the labels in `dataset_voc/SegmentationClass`. Let's use them to generate a large fully-annotated dataset.
```
python simulate_dataset.py --mask mask/ --rep rep/ --video chestnut/ --back back/ -o simulated/ --labels labels.txt -n 36000 --verbose
```

