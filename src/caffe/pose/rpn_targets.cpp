#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/pose/misc.hpp"

using namespace std;

typedef cv::Rect_<float> RectF;

float rect_IoU(const RectF &r0, const RectF &r1)
{
    const float eps = 0.01;
    cv::Point2f p11 = r0.tl();
    cv::Point2f p12 = r0.br();
    cv::Point2f p21 = r1.tl();
    cv::Point2f p22 = r1.br();

    float x_overlap = std::max(0.f, std::min(p12.x,p22.x) - std::max(p11.x,p21.x));
    float y_overlap = std::max(0.f, std::min(p12.y,p22.y) - std::max(p11.y,p21.y));
    float overlap = x_overlap * y_overlap;

    float area0 = r0.area();
    float area1 = r1.area();

    float union_ = area0+area1-overlap;
    if(union_ < eps)
        return -1;
    return overlap / union_;
}

cv::Point2f center_of_mass(std::vector<cv::Point2f> points)
{
    cv::Moments M = cv::moments(points);
    return cv::Point2f(M.m10/M.m00, M.m01/M.m00);
}


template<typename Dtype>
void prepareRPNtargets(std::string filename,
        MyRandGen *rand_gen,
        Dtype* top_rpn_cls_label,
        Dtype* top_rpn_reg_targets,
        Dtype* top_rpn_reg_weights,
        int item_id,
        int sc_map_width, int sc_map_height,
        int truncated_width, int truncated_height,
        const vector<JointList> &all_people,
        float rpn_distance_threshold,
        float scale)
{
    const int total_num_regr_targs = num_anchors * num_reg_targs;
    const float IoU_threshold = 0.7;
    const int stride = 8;
    const int half_stride = stride/2;
    const float margin = 20;
    const float dist_threshold = rpn_distance_threshold;

    const bool use_center_of_mass = true;

    /*
     * anchors, computed on the training set
     * short side - aspect ratio(h:w)
     *
        211 - 1:1
        153 - 2:1
        125 - 3:1
         97 - 4:1
     */
    float anchors [num_anchors][2] =
    {
        {1, 130},
        {1, 211},
        {2, 153},
        {3, 125},
        {4, 97}
    };
    // hardcore single person for now
    const JointList joints = all_people[0];

    auto joint_vec = jointlist_to_points(joints);
    RectF target_rect = cv::boundingRect(joint_vec);

    cv::Point2f centre = (target_rect.tl() + target_rect.br())*0.5;
    float x_s = centre.x;
    float y_s = centre.y;
    float w_s = target_rect.width;
    float h_s = target_rect.height;

    auto c_mass = center_of_mass(joint_vec);

//    LOG(WARNING) << "x_s,y_s,w_s,h_s: " << x_s << " " << y_s << " "
//                 << w_s << " " << h_s;

    int num_positives = 0;

    float max_iou = -1000;

    float x1, y1, x2, y2;
    int ratio;
    int ii, jj;

    std::vector<int> sample_mask(sc_map_width*sc_map_height);

    for (int j = 0; j < truncated_height; ++j) {
      for (int i = 0; i < truncated_width; ++i) {
        cv::Point2f pt = cv::Point2f(i*stride+half_stride, j*stride+half_stride) * (1.0/scale);
        float x_a = pt.x;
        float y_a = pt.y;

        auto target_pt = use_center_of_mass ? c_mass : cv::Point2f(x_s, y_s);
        cv::Point2f diff = target_pt - pt;
        float dist = sqrt(diff.dot(diff));
        if(dist > dist_threshold)
            continue;

        float best_iou = -1000;
        int best_anchor = -1;

        for(int k = 0; k < num_anchors; ++k) // k - anchor idx
        {
            float w_a = anchors[k][1];
            float h_a = w_a * anchors[k][0];

            RectF rect_a(x_a-0.5*w_a, y_a-0.5*h_a, w_a, h_a);
            float iou = rect_IoU(target_rect, rect_a);

            if(iou > max_iou)
            {
                max_iou = iou;
                ii = i;
                jj = j;
                x1 = rect_a.tl().x;
                y1 = rect_a.tl().y;
                x2 = rect_a.br().x;
                y2 = rect_a.br().y;
                ratio = anchors[k][0];
            }

            if(iou > best_iou)
            {
                best_iou = iou;
                best_anchor = k;
            }
        }

        // now assign anchor
        {
            int k = best_anchor;
            float w_a = anchors[k][1];
            float h_a = w_a * anchors[k][0];

            // positive sample for this anchor,
            // fill in targets for cls:
            int cls_index = ((item_id * num_anchors + k) * sc_map_height + j)
                * sc_map_width + i;
            top_rpn_cls_label[cls_index] = 1;

            // ... and reg
            float t[num_reg_targs];
            t[0] = (x_s-x_a)/w_a;
            t[1] = (y_s-y_a)/h_a;
            t[2] = std::log(w_s/w_a);
            t[3] = std::log(h_s/h_a);
            for(int l = 0; l < num_reg_targs; ++l)
            {
                int reg_index = ((item_id * total_num_regr_targs + k*num_reg_targs+l) *
                                  sc_map_height + j) * sc_map_width + i;
                top_rpn_reg_targets[reg_index] = t[l];
                top_rpn_reg_weights[reg_index] = 1.0;
            }

            num_positives += 1;
        }

        int short_index = j * sc_map_width + i;
        sample_mask[short_index] = 1;
      }
    }

    // sample negatives
    if(true)
    {
        float pos_fraction = 0.25;
        const int max_negatives = num_positives * (1.0-pos_fraction) / pos_fraction;
        int num_negatives = 0;
        const int max_iter = max_negatives * 10;

        for(int k = 0; k < max_iter; ++k)
        {
            int j = rand_gen->PrefetchRand() % truncated_height;
            int i = rand_gen->PrefetchRand() % truncated_width;
            int short_index = ((item_id) * sc_map_height + j)
                               * sc_map_width + i;
            if (sample_mask[short_index] == 1)
                continue;

            // assign zero label
            for(int k = 0; k < num_anchors; ++k) // k - anchor idx
            {
                int cls_index = ((item_id * num_anchors + k) * sc_map_height + j)
                    * sc_map_width + i;
                top_rpn_cls_label[cls_index] = 0;
            }

            // bookkeeping
            sample_mask[short_index] = 1;
            num_negatives += 1;
            if(num_negatives == max_negatives)
                break;
        }
        //LOG(WARNING) << "positives, negatives: " << num_positives << " " << num_negatives << std::endl;
    }
//    LOG(WARNING) << filename;
//    LOG(WARNING) << "  ratio, ovelap " << ratio << " " << max_iou;
//    LOG(WARNING) << "  i, j " << ii << " " << jj;
//    LOG(WARNING) << "  * x1, y1, x2, y2: " << min_x << " " << min_y << " "
//                 << max_x << " " << max_y;
//    LOG(WARNING) << "  a x1, y1, x2, y2: " << x1 << " " << y1 << " "
//                 << x2 << " " << y2;
    //LOG(WARNING) << "  num anchors " << anchors_in_image;
}

template
void prepareRPNtargets<double>(std::string filename,
        MyRandGen *rand_gen,
        double* top_rpn_cls_label,
        double* top_rpn_reg_targets,
        double* top_rpn_reg_weights,
        int item_id,
        int sc_map_width, int sc_map_height,
        int truncated_width, int truncated_height,
        const vector<JointList> &all_people,
        float rpn_distance_threshold,
        float scale);

template
void prepareRPNtargets<float>(std::string filename,
        MyRandGen *rand_gen,
        float* top_rpn_cls_label,
        float* top_rpn_reg_targets,
        float* top_rpn_reg_weights,
        int item_id,
        int sc_map_width, int sc_map_height,
        int truncated_width, int truncated_height,
        const vector<JointList> &all_people,
        float rpn_distance_threshold,
        float scale);

