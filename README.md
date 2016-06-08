# DeeperCut

Here you can find the implementation of the CNN-based human body part detectors,
used in the [DeeperCut](http://arxiv.org/abs/1605.03170).

First of all, you should build Caffe and its Python bindings as described in the [official documentation](http://caffe.berkeleyvision.org/installation.html).

In order to run the demo of pose estimation execute the following:

```
# you will need to install python's click package, ex. by executing
$ pip install click
```

```
$ cd <caffe_dir>
$ export PYTHONPATH=`pwd`/python

# Download Caffe model files
$ cd models/deepercut
$ ./download_models.sh

# Run demo of single person pose estimation
$ cd ../../python/pose
$ python ./pose_demo.py image.png --out_name=prediction
```


## Citation
Please cite Deep(er)Cut in your publications if it helps your research:

    @article{insafutdinov2016deepercut,
        author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele},
        url = {http://arxiv.org/abs/1605.03170}
        title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
        year = {2016}
    }

    @inproceedings{pishchulin16cvpr,
	    title = {DeepCut: Joint Subset Partition and Labeling for Multi Person Pose Estimation},
	    booktitle = {CVPR'16},
	    url = {},
	    author = {Leonid Pishchulin and Eldar Insafutdinov and Siyu Tang and Bjoern Andres and Mykhaylo Andriluka and Peter Gehler and Bernt Schiele}
    }