#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/pose/misc.hpp"

using namespace std;
using namespace cv;

cv::Mat copyMakeBorderSmooth(cv::Mat img, int bottom_padding, int right_padding, cv::Vec3f zero_pix)
{
    cv::Mat res(img.rows+bottom_padding*2, img.cols+right_padding*2, CV_8UC3);
    res = cv::Scalar(zero_pix);

    cv::Rect orig_rect(right_padding, bottom_padding, img.cols, img.rows);
    cv::Mat dst_roi = res(orig_rect);
    img.copyTo(dst_roi);

    const double w_3 = 1.0/3;

    // fill the top
    for(int j = 0; j < bottom_padding; ++j)
    {
        for(int ii = 0; ii < img.cols; ++ii)
        {
            Vec3f pix;
            const int prev_idx = orig_rect.y-j;
            const int i = orig_rect.x + ii;
            if(ii == 0)
                pix = 0.5*res.at<cv::Vec3b>(prev_idx, i) + 0.5*res.at<cv::Vec3b>(prev_idx, i+1);
            else if(ii == img.cols-1)
                pix = 0.5*res.at<cv::Vec3b>(prev_idx, i) + 0.5*res.at<cv::Vec3b>(prev_idx, i-1);
            else
                pix = w_3*res.at<cv::Vec3b>(prev_idx, i) + w_3*res.at<cv::Vec3b>(prev_idx, i-1) +
                      w_3*res.at<cv::Vec3b>(prev_idx, i+1);
            int len = bottom_padding*3;
            pix = (pix*(len-j)+zero_pix*j)*(1.0/float(len));
            res.at<cv::Vec3b>(prev_idx-1, i) = pix;
        }
    }

    // fill the bottom
    for(int j = 0; j < bottom_padding; ++j)
    {
        for(int ii = 0; ii < img.cols; ++ii)
        {
            Vec3f pix;
            const int prev_idx = orig_rect.y+img.rows-1+j;
            const int i = orig_rect.x + ii;
            if(ii == 0)
                pix = 0.5*res.at<cv::Vec3b>(prev_idx, i) + 0.5*res.at<cv::Vec3b>(prev_idx, i+1);
            else if(ii == img.cols-1)
                pix = 0.5*res.at<cv::Vec3b>(prev_idx, i) + 0.5*res.at<cv::Vec3b>(prev_idx, i-1);
            else
                pix = w_3*res.at<cv::Vec3b>(prev_idx, i) + w_3*res.at<cv::Vec3b>(prev_idx, i-1) +
                      w_3*res.at<cv::Vec3b>(prev_idx, i+1);
            int len = bottom_padding*3;
            pix = (pix*(len-j)+zero_pix*j)*(1.0/float(len));
            res.at<cv::Vec3b>(prev_idx+1, i) = pix;
        }
    }

    // fill the left
    for(int i = 0; i < right_padding; ++i)
    {
        for(int jj = 0; jj < img.rows; ++jj)
        {
            Vec3f pix;
            const int prev_idx = orig_rect.x - i;
            const int j = orig_rect.y + jj;

            if(j == 0)
                pix = 0.5*res.at<cv::Vec3b>(j, prev_idx) + 0.5*res.at<cv::Vec3b>(j+1, prev_idx);
            else if(j == img.rows-1)
                pix = 0.5*res.at<cv::Vec3b>(j, prev_idx) + 0.5*res.at<cv::Vec3b>(j-1, prev_idx);
            else
                pix = w_3*res.at<cv::Vec3b>(j, prev_idx) + w_3*res.at<cv::Vec3b>(j-1, prev_idx) +
                      w_3*res.at<cv::Vec3b>(j+1, prev_idx);
            int len = right_padding*3;
            pix = (pix*(len-i)+zero_pix*i)*(1.0/float(len));
            res.at<cv::Vec3b>(j, prev_idx-1) = pix;
        }
    }

    // fill the right
    for(int i = 0; i < right_padding; ++i)
    {
        for(int jj = 0; jj < img.rows; ++jj)
        {
            Vec3f pix;
            const int prev_idx = orig_rect.x + img.cols-1+i;
            const int j = orig_rect.y + jj;

            if(j == 0)
                pix = 0.5*res.at<cv::Vec3b>(j, prev_idx) + 0.5*res.at<cv::Vec3b>(j+1, prev_idx);
            else if(j == img.rows-1)
                pix = 0.5*res.at<cv::Vec3b>(j, prev_idx) + 0.5*res.at<cv::Vec3b>(j-1, prev_idx);
            else
                pix = w_3*res.at<cv::Vec3b>(j, prev_idx) + w_3*res.at<cv::Vec3b>(j-1, prev_idx) +
                      w_3*res.at<cv::Vec3b>(j+1, prev_idx);
            int len = right_padding*3;
            pix = (pix*(len-i)+zero_pix*i)*(1.0/float(len));
            res.at<cv::Vec3b>(j, prev_idx+1) = pix;
        }
    }

    return res;
}

cv::Rect myBoundingRect(JointList joints, float margin = 0.0)
{
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::min();
    float max_y = std::numeric_limits<float>::min();
    for(int k = 0; k < joints.size(); ++k)
    {
        cv::Point2f pt(joints[k].second.first, joints[k].second.second);
        min_x = std::min(pt.x, min_x);
        min_y = std::min(pt.y, min_y);
        max_x = std::max(pt.x, max_x);
        max_y = std::max(pt.y, max_y);
    }
    cv::Rect target_rect(cv::Point(min_x-margin, min_y-margin),
                      cv::Point(max_x+margin, max_y+margin));
    return target_rect;
}

std::vector<cv::Point2f> jointlist_to_points(const JointList &joints)
{
    std::vector<cv::Point2f> res;
    for(int k = 0; k < joints.size(); ++k)
    {
        auto pt = joints[k].second;
        res.push_back(cv::Point2f(pt.first, pt.second));
    }
    return res;
}

cv::Mat transl_tr(cv::Point2f tr)
{
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, tr.x, 0, 1, tr.y);
    return trans_mat;
}

cv::Mat scale_tr(double scale)
{
    Mat mat = (Mat_<double>(2,3) << scale, 0, 0,
                                    0, scale, 0);
    return mat;
}

cv::Mat pad_to_3x3(cv::Mat src)
{
   cv::Mat t(3,3,CV_64F, cvScalar(0.0));
   Mat dst_roi = t(cv::Rect(0,0,3,2));
   src.copyTo(dst_roi);
   t.at<double>(2,2) = 1;
   return t;
}

cv::Mat mul(cv::Mat t1, cv::Mat t2)
{
    auto t1_p = pad_to_3x3(t1);
    auto t2_p = pad_to_3x3(t2);
    cv::Mat res = t1_p * t2_p;
    return res(cv::Rect(0,0,3,2)).clone();
}

std::pair<cv::Mat, cv::Mat> transform_image(cv::Mat image, JointList joints_list,
                                            float angle, float scale, cv::Vec3b mean_pix)
{
    typedef cv::Rect_<float> RectF;

    const int image_height = image.rows;
    const int image_width = image.cols;
    const int pad_y = image_width/2;
    const int pad_x = image_height/2;
    const float margin = 100;
    //cv::Mat bordered;
    //cv::copyMakeBorder(image, bordered, pad_y, pad_y, pad_x, pad_x, cv::BORDER_REPLICATE);
    auto bordered = copyMakeBorderSmooth(image, pad_y, pad_x, mean_pix);

    const RectF b_rect = myBoundingRect(joints_list);

    cv::Point2f pad(pad_x, pad_y);
    cv::Point2f center = pad + (b_rect.tl() + b_rect.br()) * 0.5;
    auto matrix = getRotationMatrix2D(center, angle, scale);

    cv::Mat rotated;
    cv::warpAffine(bordered, rotated, matrix, bordered.size(), INTER_CUBIC);

    std::vector<cv::Point2f> corners_image;
    corners_image.push_back(cv::Point2f(0, 0));
    corners_image.push_back(cv::Point2f(image_width, 0));
    corners_image.push_back(cv::Point2f(0, image_height));
    corners_image.push_back(cv::Point2f(image_width, image_height));

    auto first_transf = mul(matrix, transl_tr(pad));
    std::vector<cv::Point2f> corners_tr;
    cv::transform(corners_image, corners_tr, first_transf);

    auto joint_points = jointlist_to_points(joints_list);

    RectF transformed_image_box = cv::boundingRect(corners_tr);
    auto image_box_tl = transformed_image_box.tl();
    auto image_box_br = transformed_image_box.br();

    std::vector<cv::Point2f> transformed_joints;
    cv::transform(joint_points, transformed_joints, first_transf);

    RectF joint_bbox = cv::boundingRect(transformed_joints);

    float left = std::max(joint_bbox.tl().x-margin, image_box_tl.x);
    float top = std::max(joint_bbox.tl().y-margin, image_box_tl.y);
    float right = std::min(joint_bbox.br().x+margin, image_box_br.x);
    float bottom = std::min(joint_bbox.br().y+margin, image_box_br.y);

    RectF final_rect(cv::Point2f(left, top), cv::Point2f(right, bottom));

    //cv::Mat rotated_truncated = rotated(transformed_image_box);
    cv::Mat rotated_truncated = rotated(final_rect);

    auto total_transform = mul(transl_tr(-final_rect.tl()), first_transf);

    //imshow("bordered", bordered);
    //imshow("rotated", rotated);
    //imshow("rotate truncate", rotated_truncated);

    return std::make_pair(rotated_truncated.clone(), total_transform);
}
