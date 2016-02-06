#ifndef POSE_LAYERS_HPP
#define POSE_LAYERS_HPP

//#include "caffe/data_layers.hpp"
//#include "caffe/loss_layers.hpp"

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/loss_layer.hpp"

class SimpleMatrix;

namespace caffe {

template <typename Dtype>
class MultiBatch {
 public:
  static const int MAX_LABELS = 6;
  Blob<Dtype> data_;
  Blob<Dtype> labels_[MAX_LABELS];
};

template <typename Dtype>
class MultiBasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit MultiBasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(MultiBatch<Dtype>* batch) = 0;

  MultiBatch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<MultiBatch<Dtype>*> prefetch_free_;
  BlockingQueue<MultiBatch<Dtype>*> prefetch_full_;

  Blob<Dtype> transformed_data_;
  int num_labels_;
};

/**
 * @brief Provides data to the Net for articulated pose training
 *        Data input is a text file with image path and coordinates
 *        of joints
 *
 */
template <typename Dtype>
class PoseDataLayer : public MultiBasePrefetchingDataLayer<Dtype> {
 public:
  explicit PoseDataLayer(const LayerParameter& param)
      : MultiBasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~PoseDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PoseData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  //virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  typedef vector<pair<int, pair<float, float> > > JointList;
  virtual unsigned int PrefetchRand();
  virtual void load_batch(MultiBatch<Dtype>* batch);

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  vector<vector<JointList> > joint_database_;
  bool multiperson_;
  vector<Dtype> mean_values_;
  bool has_mean_values_;
  bool cache_images_;
  vector<std::pair<std::string, Datum > > image_database_cache_;
  int img_index_;
  vector<SimpleMatrix*> neighbour_stats_;
  Blob<Dtype> min_distance_;
  Blob<Dtype> sample_mask_;
  void *uniform_real_gen;
  std::vector<float> xs,ys;
};

template <typename Dtype>
class SoftmaxWithLossVecLayer : public LossLayer<Dtype> {
 public:
  explicit SoftmaxWithLossVecLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SoftmaxWithLossVec"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  /// The internal SoftmaxLayer used to map predictions to a distribution.
  shared_ptr<Layer<Dtype> > softmax_layer_;
  shared_ptr<Layer<Dtype> > sigmoid_layer_;

  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> prob_;
  /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  /// top vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_top_vec_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// Whether to normalize the loss by the total number of values present
  /// (otherwise just by the batch size).
  bool normalize_;
  bool cross_entropy_;

  int softmax_axis_, outer_num_, inner_num_;
};

/*
template <typename Dtype>
class SmoothL1LossLayer : public LossLayer<Dtype> {
 public:
  explicit SmoothL1LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SmoothL1Loss"; }

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  // Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
  // to both inputs -- override to return true and always allow force_backward.
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> errors_;
  bool has_weights_;
  int sample_count_;
};
*/

/**
 * @brief A helper for image operations that rearranges image regions into
 *        column vectors.  Used by ConvolutionLayer to perform convolution
 *        by matrix multiplication.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
/*
template <typename Dtype>
class Im2colSampleLayer : public Layer<Dtype> {
 public:
  explicit Im2colSampleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Im2colSample"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int channels_;
  int height_, width_;
  int pad_h_, pad_w_;
  int hole_h_, hole_w_;
  Blob<Dtype> locations_;
};
*/

}

#endif // POSE_LAYERS_HPP
