#ifndef MISC_POSE_HPP
#define MISC_POSE_HPP

#include <vector>
#include <caffe/common.hpp>

class MyRandGen
{
public:
    MyRandGen();
    unsigned int PrefetchRand();

private:
    caffe::shared_ptr<caffe::Caffe::RNG> prefetch_rng_;
};

typedef std::vector<std::pair<int, std::pair<float, float> > > JointList;
const int NUM_SEGM_CLASSES = 10;

template<typename Dtype>
void sticks_segmentation(Dtype *label, float scale, int segm_stride,
                         int sc_map_width, int sc_map_height,
                         int item_id, JointList joints,
                         MyRandGen *rand_gen);

#endif // MISC_POSE_HPP
