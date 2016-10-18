#!/bin/bash
sudo apt install -y --allow-unauthenticated libgflags-dev libgoogle-glog-dev protobuf-compiler liblmdb-dev libleveldb-dev libsnappy-dev libatlas-dev libatlas-base-dev libhdf5-dev libopencv-dev libhdf5-serial-dev libboost-all-dev 

cd models/deepercut
bash download_models.sh &
cd ../..

ln -s ~/caffe/python/caffe python/pose/.

mkdir build
cd build
cmake ..
make -j8
cd ..

#ln -s /usr/include/hdf5/serial/* /usr/include/.

