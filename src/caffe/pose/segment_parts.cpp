#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/pose/misc.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

MyRandGen::MyRandGen()
{
    const unsigned int prefetch_rng_seed = caffe::caffe_rng_rand();
    prefetch_rng_.reset(new caffe::Caffe::RNG(prefetch_rng_seed));
}

unsigned int MyRandGen::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}


namespace {
    static const int NumParts = 14;
}

cv::Point2f round_pt(cv::Point2f pt)
{
    return cv::Point2f(round(pt.x), round(pt.y));
}

cv::Point2f normalise(cv::Point2f pt)
{
    return pt * (1.0 / cv::norm(pt));

}

template<typename Dtype>
void sample_negatives(Dtype *label, int k, int sc_map_width, int sc_map_height,
                      int item_id, MyRandGen *rand_gen)
{
    std::vector<bool> sample_mask(sc_map_width*sc_map_height, false);
    int num_positives = 0;
    for(int j = 0; j < sc_map_height; ++j)
    {
        for(int i = 0; i < sc_map_width; ++i)
        {
            int cls_index = ((item_id * NUM_SEGM_CLASSES + k) * sc_map_height + j)
                * sc_map_width + i;
            if(label[cls_index] == 1.0)
            {
                sample_mask[sc_map_width*j+i] = true;
                num_positives++;
            }
        }
    }

    float pos_fraction = 0.25;
    const int max_negatives = num_positives * (1.0-pos_fraction) / pos_fraction;
    int num_negatives = 0;
    const int max_iter = max_negatives * 10;

    for(int l = 0; l < max_iter; ++l)
    {
        int j = rand_gen->PrefetchRand() % sc_map_height;
        int i = rand_gen->PrefetchRand() % sc_map_width;
        int short_index = j * sc_map_width + i;
        if (sample_mask[short_index])
            continue;

        // assign zero label
        int cls_index = ((item_id * NUM_SEGM_CLASSES + k) * sc_map_height + j)
            * sc_map_width + i;
        label[cls_index] = 0.0;

        // bookkeeping
        sample_mask[short_index] = 1;
        num_negatives += 1;
        if(num_negatives == max_negatives)
            break;
    }

    //LOG(INFO) << "positives, negatives (" << k << "): " << num_positives << " " << num_negatives << std::endl;
}

template<typename Dtype>
void negate_symmetric(Dtype *label,int sc_map_width, int sc_map_height,
                      int item_id, const std::vector<std::pair<int,int>> symm)
{
    for(int idx = 0; idx < symm.size(); ++idx)
    {
        int k_this = symm[idx].first;
        int k_that = symm[idx].second;

        for(int j = 0; j < sc_map_height; ++j)
        {
            for(int i = 0; i < sc_map_width; ++i)
            {
                int index_this = ((item_id * NUM_SEGM_CLASSES + k_this) * sc_map_height + j)
                    * sc_map_width + i;
                int index_that = ((item_id * NUM_SEGM_CLASSES + k_that) * sc_map_height + j)
                    * sc_map_width + i;
                if(label[index_that] == 1 && label[index_this] != 1)
                    label[index_this] = 0;
            }
        }
    }
}

template<typename Dtype>
void negate_all(Dtype *label,int sc_map_width, int sc_map_height,
                int item_id)
{
    for(int k = 0; k < NUM_SEGM_CLASSES; ++k)
    {
        for(int j = 0; j < sc_map_height; ++j)
        {
            for(int i = 0; i < sc_map_width; ++i)
            {
                int index_this = ((item_id * NUM_SEGM_CLASSES + k) * sc_map_height + j)
                    * sc_map_width + i;

                for(int k_that = 0; k_that < NUM_SEGM_CLASSES; ++k_that)
                {
                    if(k_that == k)
                        continue;
                    int index_that = ((item_id * NUM_SEGM_CLASSES + k_that) * sc_map_height + j)
                        * sc_map_width + i;
                    if(label[index_that] == 1 && label[index_this] != 1)
                    {
                        label[index_this] = 0;
                        break;
                    }
                }
            }
        }
    }
}

template<typename Dtype>
void sticks_segmentation(Dtype *label, float scale,
                         int segm_stride,
                         int sc_map_width, int sc_map_height,
                         int item_id, JointList joints,
                         MyRandGen *rand_gen)
{
    const int segm_half_stride = segm_stride / 2;
    const float sz = 17;

    std::vector<cv::Point2f> joint_list(NumParts+1);
    std::fill(joint_list.begin(), joint_list.end(), cv::Point(-1,-1));
    for(int k = 0; k < joints.size(); ++k)
    {
        auto jnt = joints[k];
        joint_list[jnt.first] = cv::Point2f(jnt.second.first, joints[k].second.second);
    }

    const int num_sticks = NUM_SEGM_CLASSES - 1;
    int joint_pairs[num_sticks][2] = {{1, 2}, {2, 3}, {6, 5}, {4, 5}, {7, 8}, {8, 9}, {12, 11}, {11, 10}, {13, 14}};
    float limb_size_coefs[num_sticks] = {1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 1.0};

    int symmetric_limbs[][2] = {{0, 2}, {1, 3}, {4, 6}, {5, 7}};
    std::vector<std::pair<int, int>> symmetric_pairs;
    for(int k = 0; k < 4; ++k)
    {
        int l0 = symmetric_limbs[k][0];
        int l1 = symmetric_limbs[k][1];
        symmetric_pairs.push_back(std::make_pair(l0, l1));
        symmetric_pairs.push_back(std::make_pair(l1, l0));
    }

    bool sample_neg = true;
    bool neg_symm = true;
    bool neg_all = true;

    for(int k = 0; k < num_sticks; ++k)
    {
        cv::Point2f jnt1 = joint_list[joint_pairs[k][0]];
        cv::Point2f jnt2 = joint_list[joint_pairs[k][1]];

        if(jnt1.x == -1 || jnt2.x == -1)
            continue;

        if (!sample_neg)
        {
            // zero out
            for(int j = 0; j < sc_map_height; ++j)
                for(int i = 0; i < sc_map_width; ++i)
                {
                    int cls_index = ((item_id * NUM_SEGM_CLASSES + k) * sc_map_height + j)
                        * sc_map_width + i;

                    label[cls_index] = 0.0;
                }
        }


        auto diff = jnt2 - jnt1;
        const float limb_sz = sz * limb_size_coefs[k];

        if(cv::norm(diff) > 1.0)
        {
            std::vector<cv::Point2f> poly;
            auto perp = cv::Point2f(-diff.y, diff.x);
            perp = perp * (1.0 / cv::norm(perp));
            poly.push_back(jnt1-perp*limb_sz);
            poly.push_back(jnt1+perp*limb_sz);
            poly.push_back(jnt2+perp*limb_sz);
            poly.push_back(jnt2-perp*limb_sz);
            poly.push_back(poly[0]);

            for(int j = 0; j < sc_map_height; ++j)
            {
                for(int i = 0; i < sc_map_width; ++i)
                {
                    auto crd = cv::Point2f(i*segm_stride+segm_half_stride, j*segm_stride+segm_half_stride) * (1.0/scale);
                    bool inpoly = pointPolygonTest(poly, crd, false) >= 0;
                    int cls_index = ((item_id * NUM_SEGM_CLASSES + k) * sc_map_height + j)
                        * sc_map_width + i;
                    if(inpoly)
                        label[cls_index] = 1.0;
                }
            }
        }

        // now circular blobs at the joint ends
        for(int j = 0; j < sc_map_height; ++j)
        {
            for(int i = 0; i < sc_map_width; ++i)
            {
                auto crd = cv::Point2f(i*segm_stride+segm_half_stride, j*segm_stride+segm_half_stride) * (1.0/scale);
                // don't add blobs for head
                if(k != (num_sticks-1))
                {
                    bool inpoly = false;
                    if(cv::norm(crd-jnt1) <= limb_sz || cv::norm(crd-jnt2) <= limb_sz)
                        inpoly = true;
                    int cls_index = ((item_id * NUM_SEGM_CLASSES + k) * sc_map_height + j)
                        * sc_map_width + i;

                    if(inpoly)
                        label[cls_index] = 1.0;
                }
            }
        }

        if(sample_neg)
            sample_negatives(label, k, sc_map_width, sc_map_height, item_id, rand_gen);
    }
    if(neg_all)
        negate_all(label, sc_map_width, sc_map_height,
                   item_id);
    else if(neg_symm)
        negate_symmetric(label, sc_map_width, sc_map_height,
                         item_id, symmetric_pairs);

    cv::Point2f jnt1 = round_pt(joint_list[3]);
    cv::Point2f jnt2 = round_pt(joint_list[4]);
    cv::Point2f jnt3 = round_pt(joint_list[9]);
    cv::Point2f jnt4 = round_pt(joint_list[10]);

    if(jnt1.x != -1 && jnt2.x != -1 && jnt3.x != -1 && jnt4.x != -1)
    {
        std::vector<cv::Point2f> points;

        if(jnt1 == jnt2)
            jnt2.x = jnt1.x + 1;

        auto diff12 = normalise(jnt2-jnt1);
        points.push_back(jnt2 + diff12*sz);
        points.push_back(jnt1 - diff12*sz);

        if (jnt1 == jnt3)
            jnt3.y = jnt1.y - 1;

        auto diff13 = normalise(jnt3-jnt1);
        points.push_back(jnt3 + diff13*sz);
        points.push_back(jnt1 - diff13*sz);

        if (cv::norm(jnt3 - jnt4) <= sz*1.5)
        {
            if (jnt4 == jnt3)
                jnt4.x = jnt3.x + 1;

            auto diff34 = normalise(jnt4-jnt3);
            points.push_back(jnt4 + diff34*sz);
            points.push_back(jnt3 - diff34*sz);
        }

        if (jnt2 == jnt4)
            jnt4.y = jnt2.y - 1;

        auto diff24 = normalise(jnt4-jnt2);
        points.push_back(jnt4 + diff24*sz);
        points.push_back(jnt2 - diff24*sz);

        cv::Mat poly;
        cv::convexHull(cv::Mat(points), poly);

        const int torso_id = NUM_SEGM_CLASSES - 1;

        for(int j = 0; j < sc_map_height; ++j)
            for(int i = 0; i < sc_map_width; ++i)
            {
                auto crd = cv::Point2f(i*segm_stride+segm_half_stride, j*segm_stride+segm_half_stride) * (1.0/scale);
                bool inpoly = pointPolygonTest(poly, crd, false) >= 0;
                int cls_index = ((item_id * NUM_SEGM_CLASSES + torso_id) * sc_map_height + j)
                    * sc_map_width + i;
                if(inpoly)
                    label[cls_index] = 1.0;
            }

        if(sample_neg)
            sample_negatives(label, torso_id, sc_map_width, sc_map_height, item_id, rand_gen);
    }
}

template
void sticks_segmentation<float>(float *label, float scale, int segm_stride, int sc_map_width, int sc_map_height, int item_id, JointList joints, MyRandGen *rand_gen);
template
void sticks_segmentation<double>(double *label, float scale, int segm_stride, int sc_map_width, int sc_map_height, int item_id, JointList joints, MyRandGen *rand_gen);
