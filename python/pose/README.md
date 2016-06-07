# Pose Estimation

First, you should build Caffe and its Python bindings as described in the official documentation.

In order to run the demo of pose estimation execute the following:

```
$ cd <caffe_dir>
$ export PYTHONPATH=`pwd`/python
$ cd python/pose
$ python ./pose_demo.py --model=<path_to_caffe_model> image.png
```
