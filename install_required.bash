#!/bin/bash
apt install -y --allow-unauthenticated libgflags-dev libgoogle-glog-dev protobuf-compiler liblmdb-dev libleveldb-dev libsnappy-dev libatlas-dev libatlas-base-dev cuda-cublas-dev-8-0 nvidia-cuda-toolkit libhdf5-dev libopencv-dev libhdf5-serial-dev libboost-all-dev nvidia-cuda-toolkit 

ln -s /usr/include/hdf5/serial/* /usr/include/.

