#ifndef MISC_POSE_HPP
#define MISC_POSE_HPP

#include <vector>
#include <caffe/common.hpp>

#include "opencv2/core/core.hpp"

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

std::vector<cv::Point2f> jointlist_to_points(const JointList &joints);

// RPN stuff
const int num_anchors = 5;
const int num_reg_targs = 4;

template<typename Dtype>
void prepareRPNtargets(std::string filename,
        MyRandGen *rand_gen,
        Dtype* top_rpn_cls_label,
        Dtype* top_rpn_reg_targets,
        Dtype* top_rpn_reg_weights,
        int item_id,
        int sc_map_width, int sc_map_height,
        int truncated_width, int truncated_height,
        const std::vector<JointList> &all_people,
        float rpn_distance_threshold,
        float scale);

#endif // MISC_POSE_HPP
