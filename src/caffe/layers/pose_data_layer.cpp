#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <numeric>
#include <limits>       // std::numeric_limits
#include <functional>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/pose_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/util/SimpleMatrix.hpp"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>

// caffe.proto > LayerParameter > WindowDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

namespace caffe {

static const int ignore_value = 1000;

typedef boost::variate_generator<boost::mt19937&, boost::uniform_real<> > my_variate_gen;

struct UniformGenerator
{
  my_variate_gen *uniform_real_gen;
  boost::uniform_real<> dist;
  boost::mt19937 seed;
  
  UniformGenerator() :
    dist(0.0, 1.0)
  {
    timeval t;
    gettimeofday(&t,NULL);
    seed = boost::mt19937( (int)t.tv_sec );
  
    uniform_real_gen = new my_variate_gen(seed,dist);
  }
  
  ~UniformGenerator()
  {
     delete uniform_real_gen; 
  }
  
  double generate()
  {
      return (*uniform_real_gen)();
  }
};

void add_reverted_edges(const int *start, const int *end, int num,
                        vector<int> &starts, vector<int> &ends)
{
    starts.resize(num*2);
    ends.resize(num*2);
    for(int k = 0; k < num; ++k)
    {
        starts[k] = start[k];
        ends[k] = end[k];
    }
    for(int k=0; k < num; ++k)
    {
        starts[num+k] = end[k];
        ends[num+k] = start[k];
    }
}

int joint_from_class(int cls)
{
    return cls - 1;
}

template <typename Dtype>
PoseDataLayer<Dtype>::~PoseDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void PoseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LayerSetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2

  LOG(INFO) << "Pose data layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.pose_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.pose_data_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.pose_data_param().fg_fraction() << std::endl
      << "  cache_images: "
      << this->layer_param_.pose_data_param().cache_images() << std::endl
      << "  fg_distance_threshold: "
      << this->layer_param_.pose_data_param().fg_threshold() << std::endl
      << "  multi_label: "
      << this->layer_param_.pose_data_param().multi_label() << std::endl
      << "  root_folder: "
      << this->layer_param_.pose_data_param().root_folder();

  cache_images_ = this->layer_param_.pose_data_param().cache_images();
  string root_folder = this->layer_param_.pose_data_param().root_folder();

  const bool prefetch_needs_rand = true;
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }

  rand_gen_ = new MyRandGen();
  
  std::ifstream infile(this->layer_param_.pose_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open pose file "
      << this->layer_param_.pose_data_param().source() << std::endl;

  multiperson_ = false;
  string hashtag;
  int image_index, channels;
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;

    int num_persons = 1;
    std::string prefix = "multi";
    //LOG(WARNING) << "string is: " << image_path;
    if(std::equal(prefix.begin(), prefix.end(), image_path.begin()))
    {
        multiperson_ = true;
        infile >> num_persons;
        infile >> image_path;
    }

    image_path = root_folder + image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    channels = image_size[0];
    image_database_.push_back(std::make_pair(image_path, image_size));

    if (cache_images_) {
      Datum datum;
      if (!ReadFileToDatum(image_path, &datum)) {
        LOG(ERROR) << "Could not open or find file " << image_path;
        return;
      }
      image_database_cache_.push_back(std::make_pair(image_path, datum));
    }

    std::vector<JointList> all_people;
    for(int k = 0; k < num_persons; ++k)
    {
        int num_joints;
        infile >> num_joints;

        //LOG(WARNING) << "--- num joints ---- " << num_joints;

        JointList joints(num_joints);

        for (int i = 0; i < num_joints; ++i) {
            int cls;
            infile >> cls;
            pair<float, float> jnt;
            infile >> jnt.first >> jnt.second;
            joints[i] = std::make_pair(cls, jnt);
        }

        all_people.push_back(joints);
    }
    
    joint_database_.push_back(all_people);

  } while (infile >> hashtag >> image_index);


  img_index_ = 0;
  
  // fake some size for the first iteration
  int input_height = 160; // this->layer_param_.pose_data_param().input_height();
  int input_width  = 160; // this->layer_param_.pose_data_param().input_width();
  const int stride = 8;
  const int segm_stride = this->layer_param().pose_data_param().segmentation_stride();

  // FIX STRIDE
  int sc_map_height = ceil(input_height/stride)/*+1*/;
  int sc_map_width = ceil(input_width/stride)/*+1*/;
  int segm_scmap_height = ceil(input_height / segm_stride);
  int segm_scmap_width = ceil(input_width / segm_stride);

  bool no_bg_class = this->layer_param_.pose_data_param().no_bg_class();
    
  const int batch_size = this->layer_param_.pose_data_param().batch_size();
  top[0]->Reshape(batch_size, channels, input_height, input_width);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i)
    this->prefetch_[i].data_.Reshape(batch_size, channels, input_height, input_width);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // -------------------- top blob indexes ----------------------
  bool locref = this->layer_param_.pose_data_param().location_refinement();
  bool allreg = this->layer_param_.pose_data_param().regress_to_other();
  bool rpn = this->layer_param_.pose_data_param().rpn();
  bool segmentation = this->layer_param_.pose_data_param().segmentation();
  bool weight_targets = this->layer_param().pose_data_param().weight_targets();

  int idx_cls = 0;
  int idx_cls_weights = 0;
  int idx_locref_targets = 0;
  int idx_locref_weights = 0;
  int idx_allreg_targets = 0;
  int idx_allreg_weights = 0;
  int idx_rpn_cls_targets = 0;
  int idx_rpn_reg_targets = 0;
  int idx_rpn_reg_weights = 0;
  int idx_segm_cls_targets = 0;

  int label_idx = 0;
  // TODO fix to label_idx++
  idx_cls = label_idx;
  label_idx += 1;
  if(weight_targets)
  {
    idx_cls_weights = label_idx;
    label_idx += 1;
  }
  if(locref)
  {
    idx_locref_targets = label_idx;
    idx_locref_weights = label_idx + 1;
    label_idx += 2;
  }
  if(allreg)
  {
    idx_allreg_targets = label_idx;
    idx_allreg_weights = label_idx + 1;
    label_idx += 2;
  }
  if(rpn)
  {
    idx_rpn_cls_targets = label_idx;
    idx_rpn_reg_targets = label_idx + 1;
    idx_rpn_reg_weights = label_idx + 2;
    label_idx += 3;
  }
  if(segmentation)
  {
    idx_segm_cls_targets = label_idx;
    label_idx += 1;
  }
  // ---------------------------------------------------------


  // label
  int num_labels = this->num_labels_;
  const int num_classes = this->layer_param_.pose_data_param().num_classes();
  const int NUM_JOINTS = num_classes;

  auto joint_stats_file = this->layer_param().pose_data_param().has_joint_pairs_stats() ?
              this->layer_param().pose_data_param().joint_pairs_stats() :
              "/BS/eldar/work/pose/exp/mpii-all-joints-reg/data/all_stats.txt";

  neighbour_stats_ = readMatricesFromFile(joint_stats_file);
  SimpleMatrix *regr_edges = neighbour_stats_[0];
  const int num_regr_targets = regr_edges->rows();

  int label_channels = num_classes + (no_bg_class ? 0 : 1);
  int num_locs = NUM_JOINTS*2;
  int num_next_channels = num_regr_targets*2;

  top[idx_cls+1]->Reshape(batch_size, label_channels, sc_map_height, sc_map_width);
  if(weight_targets) {
    top[idx_cls_weights+1]->Reshape(batch_size, label_channels, sc_map_height, sc_map_width);
  }
  if(locref) {
    top[idx_locref_targets+1]->Reshape(batch_size, num_locs, sc_map_height, sc_map_width);
    top[idx_locref_weights+1]->Reshape(batch_size, num_locs, sc_map_height, sc_map_width);
  }
  if(allreg) {
      top[idx_allreg_targets+1]->Reshape(batch_size, num_next_channels, sc_map_height, sc_map_width);
      top[idx_allreg_weights+1]->Reshape(batch_size, num_next_channels, sc_map_height, sc_map_width);
  }

  const int num_rpn_cls_labels = num_anchors;
  const int num_rpn_regr_targets = num_anchors * num_reg_targs;

  if(rpn)
  {
      top[idx_rpn_cls_targets+1]->Reshape(batch_size, num_anchors, sc_map_height, sc_map_width);
      top[idx_rpn_reg_targets+1]->Reshape(batch_size, num_rpn_regr_targets, sc_map_height, sc_map_width);
      top[idx_rpn_reg_weights+1]->Reshape(batch_size, num_rpn_regr_targets, sc_map_height, sc_map_width);
  }

  if(segmentation) {
      top[idx_segm_cls_targets+1]->Reshape(batch_size, NUM_SEGM_CLASSES, segm_scmap_height, segm_scmap_width);
  }

  LOG(WARNING) << "number of top blobs " << top.size();

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].labels_[idx_cls].Reshape(batch_size, label_channels, sc_map_height, sc_map_width);
    if(weight_targets) {
      this->prefetch_[i].labels_[idx_cls_weights].Reshape(batch_size, label_channels, sc_map_height, sc_map_width);
    }
    if(locref) {
      this->prefetch_[i].labels_[idx_locref_targets].Reshape(batch_size, num_locs, sc_map_height, sc_map_width);
      this->prefetch_[i].labels_[idx_locref_weights].Reshape(batch_size, num_locs, sc_map_height, sc_map_width);
    }
    if(allreg) {
      this->prefetch_[i].labels_[idx_allreg_targets].Reshape(batch_size, num_next_channels, sc_map_height, sc_map_width);
      this->prefetch_[i].labels_[idx_allreg_weights].Reshape(batch_size, num_next_channels, sc_map_height, sc_map_width);
    }
    if(rpn) {
      this->prefetch_[i].labels_[idx_rpn_cls_targets].Reshape(batch_size, num_rpn_cls_labels, sc_map_height, sc_map_width);
      this->prefetch_[i].labels_[idx_rpn_reg_targets].Reshape(batch_size, num_rpn_regr_targets, sc_map_height, sc_map_width);
      this->prefetch_[i].labels_[idx_rpn_reg_weights].Reshape(batch_size, num_rpn_regr_targets, sc_map_height, sc_map_width);
    }
    if(segmentation) {
      this->prefetch_[i].labels_[idx_segm_cls_targets].Reshape(batch_size, NUM_SEGM_CLASSES, segm_scmap_height, segm_scmap_width);
    }
  }

  this->min_distance_.Reshape(batch_size, 1, sc_map_height, sc_map_width);
  this->sample_mask_.Reshape(batch_size, 1, sc_map_height, sc_map_width);

  // data mean
  has_mean_values_ = this->transform_param_.mean_value_size() > 0;
  if (has_mean_values_) {
    for (int c = 0; c < this->transform_param_.mean_value_size(); ++c) {
      mean_values_.push_back(this->transform_param_.mean_value(c));
    }
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
     "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  const int max_input_size = this->layer_param_.pose_data_param().max_input_size();
  LOG(INFO) << "Maximum input size: " << max_input_size << " * " << max_input_size;


  uniform_real_gen = new UniformGenerator();

  total_num_images_ = 0;
  rejected_images_ = 0;
}

template <typename Dtype>
unsigned int PoseDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// Thread fetching the data
template <typename Dtype>
void PoseDataLayer<Dtype>::load_batch(MultiBatch<Dtype>* batch) {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  //LOG(WARNING) << "--------------------- load_batch ----------------------------";

  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  const double orig_scale = this->layer_param_.pose_data_param().scale();
  const double scale_jitter_up = this->layer_param_.pose_data_param().scale_jitter_up();
  const double scale_jitter_lo = this->layer_param_.pose_data_param().scale_jitter_lo();
  const bool jitter_scale = this->layer_param_.pose_data_param().has_scale_jitter_up() &&
                            this->layer_param_.pose_data_param().has_scale_jitter_lo();
  
  const int batch_size = this->layer_param_.pose_data_param().batch_size();
  // limit batch size by 1 for now
  CHECK_EQ(batch_size, 1);

  const float fg_fraction =
      this->layer_param_.pose_data_param().fg_fraction();
  const bool sequential  = this->layer_param_.pose_data_param().sequential();
  const bool soft_labels = this->layer_param_.pose_data_param().soft_labels();
  const bool multi_label = this->layer_param_.pose_data_param().multi_label();
  const bool no_bg_class = this->layer_param_.pose_data_param().no_bg_class();
  const double gauss_blob_sigma = this->layer_param_.pose_data_param().gauss_blob_sigma();
  const int num_classes  = this->layer_param_.pose_data_param().num_classes();
  const int NUM_JOINTS = num_classes;

  const float fg_distance_threshold = this->layer_param_.pose_data_param().fg_threshold();
  const float bg_distance_threshold = this->layer_param_.pose_data_param().bg_threshold();
  const int label_dim = batch->labels_[0].shape(1);

  bool use_bg_threshold = this->layer_param_.pose_data_param().has_bg_threshold();

  int label_channels = num_classes + (no_bg_class ? 0 : 1);
  const int skip_class = num_classes + 1;
  const int stride = 8;
  const int half_stride = 4;

  bool use_fg_fraction = this->layer_param_.pose_data_param().has_fg_fraction();

  bool locref = this->layer_param_.pose_data_param().location_refinement();
  bool allreg = this->layer_param_.pose_data_param().regress_to_other();
  bool rpn = this->layer_param_.pose_data_param().rpn();
  bool segmentation = this->layer_param_.pose_data_param().segmentation();
  bool weight_targets = this->layer_param().pose_data_param().weight_targets();

  const int max_input_size = this->layer_param_.pose_data_param().max_input_size();

  UniformGenerator *real_gen = (UniformGenerator*)uniform_real_gen;

  SimpleMatrix *regr_edges = neighbour_stats_[0];
  SimpleMatrix *regr_means = neighbour_stats_[1];
  SimpleMatrix *regr_std_devs = neighbour_stats_[2];

  // -------------------- top blob indexes ----------------------
  int idx_cls = 0;
  int idx_cls_weights = 0;
  int idx_locref_targets = 0;
  int idx_locref_weights = 0;
  int idx_allreg_targets = 0;
  int idx_allreg_weights = 0;
  int idx_rpn_cls_targets = 0;
  int idx_rpn_reg_targets = 0;
  int idx_rpn_reg_weights = 0;
  int idx_segm_cls_targets = 0;

  int label_idx = 0;

  idx_cls = label_idx;
  label_idx += 1;
  if(weight_targets)
  {
    idx_cls_weights = label_idx;
    label_idx += 1;
  }
  if(locref)
  {
    idx_locref_targets = label_idx;
    idx_locref_weights = label_idx + 1;
    label_idx += 2;
  }
  if(allreg)
  {
    idx_allreg_targets = label_idx;
    idx_allreg_weights = label_idx + 1;
    label_idx += 2;
  }
  if(rpn)
  {
      idx_rpn_cls_targets = label_idx;
      idx_rpn_reg_targets = label_idx + 1;
      idx_rpn_reg_weights = label_idx + 2;
      label_idx += 3;
  }
  if(segmentation)
  {
    idx_segm_cls_targets = label_idx;
    label_idx += 1;
  }
  // ---------------------------------------------------------

  CHECK_LE(label_idx, MultiBatch<Dtype>::MAX_LABELS) << "number of labels exceeds MAX_LABELS";

  int num_images = image_database_.size();
  int item_id = 0;
  do {
    unsigned int img_index = PrefetchRand() % num_images;
    if (sequential)
    {
      img_index = img_index_;
      img_index_ = (img_index_ + 1) % num_images;
    }

    double scale = orig_scale;
    if (jitter_scale)
    {
        double random_real = real_gen->generate();
        scale *= scale_jitter_lo + (scale_jitter_up-scale_jitter_lo)*random_real;
    }
    
    std::pair<std::string, vector<int> > image_entry = image_database_[img_index];
    vector<JointList> all_people = joint_database_[img_index];
    const int num_people = all_people.size();

    // compute maps from joint id(joint number) to index in the JointList
    vector<vector<int> > joint_indexes_people;
    for(int i = 0; i < num_people; ++i)
    {
        JointList joints = all_people[i];
        std::vector<int> joint_all(NUM_JOINTS);
        std::fill(joint_all.begin(), joint_all.end(), -1);
        for(int k = 0; k < joints.size(); ++k)
        {
            const int cls = joints[k].first;
            joint_all[cls-1] = k;
        }
        joint_indexes_people.push_back(joint_all);
    }

    //LOG(WARNING) << "Image: " << image_entry.first << std::endl;
    
    vector<int> image_size = image_entry.second;
    const int orig_width = image_size[2];
    const int orig_height = image_size[1];
    if(orig_height < 100 || orig_width < 100)
        continue;

    // FIXSTRIDE
    const int sc_map_height = ceil(orig_height * scale / stride)/*+1*/;
    const int sc_map_width = ceil(orig_width * scale / stride)/*+1*/;
    const int input_height = (sc_map_height/*-1*/) * stride;
    const int input_width = (sc_map_width/*-1*/) * stride;

    //total_num_images_ += 1;
    // some images don't fit to GPU's memory so we have to limit ourselves
    const int max_allowed_size = max_input_size; //852;
    if(input_height*input_width > max_allowed_size*max_allowed_size)
    {
        //rejected_images_ += 1;
        continue;
    }

    //LOG(INFO) << "rejected/total " << rejected_images_ << "/" << total_num_images_;

    int num_labels = this->num_labels_;

    const int num_locs = NUM_JOINTS*2;
    const int num_regr_targets = regr_edges->rows();
    CHECK_EQ(num_regr_targets, 182);
    int num_next_channels = num_regr_targets * 2;

    Blob<Dtype> *labels = batch->labels_;

    batch->data_.Reshape(batch_size, image_size[0], input_height, input_width);
    Dtype* top_label =
            prepareLabel(labels[idx_cls], batch_size, label_channels, sc_map_width, sc_map_height,  Dtype(ignore_value));
    Dtype* top_weights_label = 0;
    if(weight_targets)
        top_weights_label = prepareLabel(labels[idx_cls_weights], batch_size, label_channels, sc_map_width, sc_map_height, Dtype(1));
    Dtype* top_loc_label = 0;
    Dtype* top_loc_weights = 0;
    if(locref) {
        top_loc_label = prepareLabel(labels[idx_locref_targets], batch_size, num_locs, sc_map_width, sc_map_height, Dtype(0));
        top_loc_weights = prepareLabel(labels[idx_locref_weights], batch_size, num_locs, sc_map_width, sc_map_height, Dtype(0));
    }
    Dtype* top_next_label = 0;
    Dtype* top_next_weights = 0;
    if(allreg) {
        top_next_label = prepareLabel(labels[idx_allreg_targets], batch_size, num_next_channels, sc_map_width, sc_map_height, Dtype(0));
        top_next_weights = prepareLabel(labels[idx_allreg_weights], batch_size, num_next_channels, sc_map_width, sc_map_height, Dtype(0));
    }
    this->min_distance_.Reshape(batch_size, 1, sc_map_height, sc_map_width);
    this->sample_mask_.Reshape(batch_size, 1, sc_map_height, sc_map_width);

    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* min_distance_data = this->min_distance_.mutable_cpu_data();
    Dtype* sample_mask_data = this->sample_mask_.mutable_cpu_data();
    // zero out batch
    caffe_set(batch->data_.count(), Dtype(0), top_data);
    caffe_set(this->sample_mask_.count(), Dtype(0), sample_mask_data);
    
    timer.Start();
    cv::Mat image;
    if (this->cache_images_) {
      pair<std::string, Datum> image_cached =
          image_database_cache_[img_index];
      image = DecodeDatumToCVMat(image_cached.second, true);
    } else {
      image = cv::imread(image_entry.first, CV_LOAD_IMAGE_COLOR);
      if (!image.data) {
          LOG(ERROR) << "Could not open or find file " << image_entry.first;
          return;
      }
    }
    read_time += timer.MicroSeconds();
    timer.Start();
    const int channels = image.channels();
    
    cv::Size cv_new_size(round(orig_width*scale), round(orig_height*scale));
    cv::resize(image, image,
               cv_new_size, 0, 0, cv::INTER_LINEAR);
    
    const int image_height = image.rows;
    const int image_width = image.cols;
    const int truncated_height = ceil(double(image_height)/stride);
    const int truncated_width = ceil(double(image_width)/stride);

    cv::Mat final_img;
    const int border_padding = 64;
    cv::copyMakeBorder(image, final_img, 0, border_padding, 0, border_padding, cv::BORDER_REPLICATE);

    cv::Mat cv_cropped_img(input_height,input_width,CV_8UC3);
    cv_cropped_img.setTo(cv::Scalar(mean_values_[0], mean_values_[1], mean_values_[2]));

    cv::Rect roi(0, 0,
                 std::min(final_img.cols, input_width),
                 std::min(final_img.rows, input_height));
    cv::Mat tmp = final_img(roi);
    cv::Mat dst_roi = cv_cropped_img(roi);
    tmp.copyTo(dst_roi);

    // copy image into top_data
    for (int h = 0; h < cv_cropped_img.rows; ++h) {
      const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < cv_cropped_img.cols; ++w) {
        for (int c = 0; c < channels; ++c) {
          int top_index = ((item_id * channels + c) * input_height + h)
                           * input_width + w;
          // int top_index = (c * height + h) * width + w;
          Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
          if (this->has_mean_values_) {
            top_data[top_index] = pixel - this->mean_values_[c];
          } else {
            top_data[top_index] = pixel;
          }
        }
      }
    }

    const int first_class_idx = no_bg_class ? 1 : 0;
    const float loc_refine_std = sqrt(53);

    // construct label
    int num_positives = 0;
    std::vector<Dtype> scores(num_classes+1);
    std::vector<cv::Point2f> diffs(NUM_JOINTS);
    for (int j = 0; j < truncated_height; ++j) {
      for (int i = 0; i < truncated_width; ++i) {

        std::fill(scores.begin(), scores.end(), 0);
        // distance to the closest joint for each type
        vector<Dtype> dists(NUM_JOINTS);
        std::fill(dists.begin(), dists.end(), FLT_MAX);
        // for each joint stores person ID that has closest joint to the current location
        vector<int> person_dists(NUM_JOINTS);
        std::fill(person_dists.begin(), person_dists.end(), -1);

        cv::Point2f pt = cv::Point2f(i*stride+half_stride, j*stride+half_stride) * (1.0/scale);
        float min_dist = FLT_MAX;
        int closest_joint = -1;
        bool skip_sample = false;
        for(int p = 0; p < num_people; ++p)
        {
            const JointList joints = all_people[p];
            for(int k = 0; k < joints.size(); ++k)
            {
                const int cls = joints[k].first;
                cv::Point2f diff = cv::Point2f(joints[k].second.first, joints[k].second.second) - pt;
                float dist = sqrt(diff.dot(diff));
                const int joint_id = joint_from_class(cls);
                if(dist < dists[joint_id])
                {
                    if(soft_labels)
                        scores[cls] = exp(-dist*dist/(2*gauss_blob_sigma*gauss_blob_sigma));
                    else
                        scores[cls] = (dist <= fg_distance_threshold) ? 1 : 0;
                    dists[joint_id] = dist;
                    person_dists[joint_id] = p;
                    if(cls != skip_class)
                        diffs[joint_id] = diff*scale;
                }
                if(dist < min_dist)
                {
                    min_dist = dist;
                    closest_joint = cls;
                }
                if(cls == skip_class && scores[cls] > 0.05)
                    skip_sample = true;
            }
        }
        int short_index = ((item_id) * sc_map_height + j)
                            * sc_map_width + i;
        min_distance_data[short_index] = min_dist;
        // assign background score for soft_labels
        scores[0] = 1 - scores[closest_joint];

        float fg_score_thresh = 0.05;
        bool is_foreground = soft_labels ?
                               scores[0] <= (1-fg_score_thresh) :
                               min_dist <= fg_distance_threshold;
        if(is_foreground)
            num_positives += 1;
        if(is_foreground || skip_sample)
            sample_mask_data[short_index] = 1;
        if(skip_sample)
            continue;
        if(use_fg_fraction && !is_foreground)
            continue;

        if(!soft_labels && !multi_label) // mutually exclusive classes, softmax
        {
            int curr_class = is_foreground ? closest_joint : 0;
            for(int cls = 0; cls <= num_classes; ++cls)
                scores[cls] = cls == curr_class ? 1 : 0;
        }
        for(int cls = first_class_idx; cls <= num_classes; ++cls)
        {
            int top_index = ((item_id * label_dim + cls-first_class_idx) * sc_map_height + j)
                * sc_map_width + i;
            top_label[top_index] = scores[cls];
        }
        if(is_foreground && locref)
        {
            for(int cls = 1; cls <= num_classes; ++cls)
            {
                if(scores[cls] < fg_score_thresh)
                    continue;
                float diff[2];
                const int joint_id = joint_from_class(cls);
                diff[0] = diffs[joint_id].x;
                diff[1] = diffs[joint_id].y;
                for(int k = 0; k < 2; ++k) {
                    int top_index = ((item_id * num_classes + joint_id*2+k) * sc_map_height + j) * sc_map_width + i;
                    top_loc_label[top_index] = diff[k]/loc_refine_std;
                    top_loc_weights[top_index] = 1;
                }
            }
        }
        if(is_foreground && allreg)
        {
            for(int l = 0; l < num_regr_targets; ++l)
            {
                int cls = regr_edges->val(l, 0);
                int next_cls = regr_edges->val(l, 1);
                if(scores[cls] < fg_score_thresh)
                    continue;

                const int this_joint_idx = joint_from_class(cls);
                const int person = person_dists[this_joint_idx];
                const JointList &joints = all_people[person];
                const std::vector<int> &joint_all = joint_indexes_people[person];

                int next_joint_idx = joint_all[next_cls-1];
                if(next_joint_idx == -1) // no next joint in the image
                    continue;

                cv::Point2f next(joints[next_joint_idx].second.first,
                                 joints[next_joint_idx].second.second);
                cv::Point2f diff_ = (next - pt)*scale;

                float diff[2];
                diff[0] = (diff_.x - regr_means->val(l, 0)) / regr_std_devs->val(l, 0);
                diff[1] = (diff_.y - regr_means->val(l, 1)) / regr_std_devs->val(l, 1);
                //LOG(WARNING) << "(" << j << ", " << i << ") " << cls << " " << next_cls << " " << diff[0] << " " << diff[1] << " ";


                for(int k = 0; k < 2; ++k) {
                    int top_index = ((item_id * num_next_channels + l*2+k) * sc_map_height + j) * sc_map_width + i;
                    top_next_label[top_index] = diff[k];
                    top_next_weights[top_index] = 1;
                }
            }
        }
      }
    }
    
    if(weight_targets) // downweight negatives
    {
        const int total_num_samples = sc_map_height * sc_map_width;
        Dtype weight = (1-fg_fraction)/fg_fraction*(num_positives)/(total_num_samples-num_positives);

        for (int j = 0; j < sc_map_height; ++j) {
          for (int i = 0; i < sc_map_width; ++i) {

              int short_index = ((item_id) * sc_map_height + j)
                                 * sc_map_width + i;
              if (sample_mask_data[short_index] == 1)
                  continue;

              for(int c = first_class_idx; c <= num_classes; ++c) {
                  int top_index = ((item_id * label_dim + c-first_class_idx) * sc_map_height + j)
                      * sc_map_width + i;
                  top_label[top_index] = c == 0 ? 1 : 0;
                  top_weights_label[top_index] = weight;
              }
          }
        }
    }
    else if(use_fg_fraction) // or sample negatives
    {
        const int max_negatives = num_positives * (1.0-fg_fraction) / fg_fraction;
        int num_negatives = 0;
        const int max_iter = max_negatives * 10;
        
        for(int k = 0; k < max_iter; ++k)
        {
            int j = PrefetchRand() % truncated_height;
            int i = PrefetchRand() % truncated_width;
            int short_index = ((item_id) * sc_map_height + j)
                               * sc_map_width + i;
            if (sample_mask_data[short_index] == 1)
                continue;
            if (use_bg_threshold && min_distance_data[short_index] <= bg_distance_threshold)
                continue;
            for(int c = first_class_idx; c <= num_classes; ++c) {
                int top_index = ((item_id * label_dim + c-first_class_idx) * sc_map_height + j)
                    * sc_map_width + i;
                top_label[top_index] = c == 0 ? 1 : 0;
            }
            sample_mask_data[short_index] = 1;
            num_negatives += 1;
            if(num_negatives == max_negatives)
                break;
        }
        //LOG(WARNING) << "positives, negatives: " << num_positives << " " << num_negatives << std::endl;
    }

    if(rpn) {
        const int num_rpn_cls_labels = num_anchors;
        const int num_rpn_regr_targets = num_anchors * num_reg_targs;
        const float rpn_dist_thresh = this->layer_param_.pose_data_param().rpn_distance_threshold();

        Dtype *top_rpn_cls_label = prepareLabel(labels[idx_rpn_cls_targets], batch_size, num_rpn_cls_labels, sc_map_width, sc_map_height, Dtype(ignore_value));
        Dtype *top_rpn_reg_targets = prepareLabel(labels[idx_rpn_reg_targets], batch_size, num_rpn_regr_targets, sc_map_width, sc_map_height, Dtype(0));
        Dtype *top_rpn_reg_weights = prepareLabel(labels[idx_rpn_reg_weights], batch_size, num_rpn_regr_targets, sc_map_width, sc_map_height, Dtype(0));
        prepareRPNtargets(image_entry.first,
                rand_gen_,
                top_rpn_cls_label,
                top_rpn_reg_targets,
                top_rpn_reg_weights,
                item_id,
                sc_map_width, sc_map_height,
                truncated_width, truncated_height,
                all_people, rpn_dist_thresh,
                scale);
    }
    if(segmentation) {
        const int segm_stride = this->layer_param().pose_data_param().segmentation_stride();
        const int segm_scmap_height = ceil(input_height / segm_stride);
        const int segm_scmap_width = ceil(input_width / segm_stride);
        Dtype *top_segm_cls_label = prepareLabel(labels[idx_segm_cls_targets], batch_size,
                                                 NUM_SEGM_CLASSES, segm_scmap_width, segm_scmap_height,
                                                 Dtype(ignore_value));

        sticks_segmentation(top_segm_cls_label, scale, segm_stride,
                            segm_scmap_width, segm_scmap_height,
                            item_id, all_people[0],
                            rand_gen_);

        /*
        int num_segm_pixels = NUM_SEGM_CLASSES * segm_scmap_width * segm_scmap_height;
        int num_zero = 0;
        int num_ignore = 0;
        int num_pos = 0;
        for(int k = 0; k < num_segm_pixels; ++k)
        {
            Dtype val = top_segm_cls_label[k];
            if(val == Dtype(ignore_value))
                num_ignore++;
            else if(val == 0.0)
                num_zero++;
            else if(val == 1.0)
                num_pos++;
        }
        LOG(INFO) << "SEGM STATS "<< num_pos << " " << num_zero << " " << num_ignore;
        */
    }

    trans_time += timer.MicroSeconds();
    
    item_id++;
  } while(item_id < batch_size);
  
  /*
  size_t n = xs.size();
  float var_x = caffe_cpu_dot(n, xs.data(), xs.data())/n;
  float var_y = caffe_cpu_dot(n, ys.data(), ys.data())/n;
  LOG(WARNING) << var_x << " " << var_y << " " << n << std::endl;
  */

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
Dtype *prepareLabel(Blob<Dtype> &label,
                  int batch_size, int num_channels, int width, int height,
                  Dtype init_value)
{
    label.Reshape(batch_size, num_channels, height, width);
    Dtype *ptr = label.mutable_cpu_data();
    caffe_set(label.count(), init_value, ptr);
    return ptr;
}

INSTANTIATE_CLASS(PoseDataLayer);
REGISTER_LAYER_CLASS(PoseData);

}  // namespace caffe
