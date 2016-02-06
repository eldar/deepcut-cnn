#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/pose_layers.hpp"

namespace caffe {

static const int ignore_value = 1000;
    
template <typename Dtype>
void SoftmaxWithLossVecLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);

  cross_entropy_ = this->layer_param_.softmax_with_loss_vec_param().cross_entropy();
  
  if(cross_entropy_)
  {
    LayerParameter sigmoid_param(this->layer_param_);
    sigmoid_param.set_type("Sigmoid");
    sigmoid_layer_ = LayerRegistry<Dtype>::CreateLayer(sigmoid_param);
    sigmoid_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  }
  else
  {
    LayerParameter softmax_param(this->layer_param_);
    softmax_param.set_type("Softmax");
    softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
    softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  }
  
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void SoftmaxWithLossVecLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  if(cross_entropy_)
    sigmoid_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  else
    softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  int num_classes = bottom[0]->shape(1);
  CHECK_EQ(outer_num_ * inner_num_ * num_classes, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLossVecLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  bool use_softmax = !this->layer_param_.softmax_with_loss_vec_param().no_softmax();
  bool smooth_l1 = this->layer_param_.softmax_with_loss_vec_param().smooth_l1();
  if(cross_entropy_) {
    sigmoid_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  } else if(use_softmax) {
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  } else {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* prob_data = prob_.mutable_cpu_data();
    caffe_copy(bottom[0]->count(), bottom_data, prob_data);
  }

  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* input_data = bottom[0]->cpu_data();

  int dim = prob_.count() / outer_num_;
  int num_classes = prob_.shape(1);
  int count = 0;
  Dtype loss = 0;
  
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      bool ignore = label[i * dim + j] == ignore_value;
      if (ignore) {
        continue;
      }
      if(cross_entropy_) {
        for (int c = 0; c < num_classes; ++c) {
          int idx = i * dim + c * inner_num_ + j;
          loss -= input_data[idx] * (label[idx] - (input_data[idx] >= 0)) -
                     log(1 + exp(input_data[idx] - 2 * input_data[idx] * (input_data[idx] >= 0)));
        }
      } else if(use_softmax) {
        int label_value = -1;
        Dtype label_prob = FLT_MIN;
        for (int c = 0; c < num_classes; ++c) {
            Dtype val = label[i * dim + c * inner_num_ + j];
            if(val > label_prob) {
                label_value = c;
                label_prob = val;
            }
        }
        loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                            Dtype(FLT_MIN)));
      } else if(smooth_l1) {
        for (int c = 0; c < num_classes; ++c) {
          int idx = i * dim + c * inner_num_ + j;
          Dtype diff = prob_data[idx] - label[idx];
          Dtype abs_diff = abs(diff); 
          if (abs_diff < 1) {
            loss += 0.5 * diff * diff;
          } else {
            loss += abs_diff - 0.5;
          }
        }
      } else { // regression loss
        for (int c = 0; c < num_classes; ++c) {
          int idx = i * dim + c * inner_num_ + j;
          Dtype diff = prob_data[idx] - label[idx];
          loss += diff*diff;
        }
      }
      ++count;
    }
  }
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossVecLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bool smooth_l1 = this->layer_param_.softmax_with_loss_vec_param().smooth_l1();
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    caffe_axpy(prob_.count(), Dtype(-1.0), label, bottom_diff);
    int dim = prob_.count() / outer_num_;
    int num_classes = bottom[0]->shape(softmax_axis_);
    int count = 0;

    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        bool ignore = label[i * dim + j] == ignore_value;
        if (ignore) {
          for (int c = 0; c < num_classes; ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          if(smooth_l1) {
            for (int c = 0; c < num_classes; ++c) {
              const int idx = i * dim + c * inner_num_ + j;
              const Dtype diff = bottom_diff[idx];
              const Dtype abs_diff = abs(diff);
              if (abs_diff >= 1) {
                bottom_diff[idx] = (Dtype(0) < diff) - (diff < Dtype(0));
              }
            }
          }
          ++count;
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossVecLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    //LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode.";
    Forward_cpu(bottom, top);
}

template <typename Dtype>
void SoftmaxWithLossVecLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
    //LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode.";
    Backward_cpu(top, propagate_down, bottom);
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossVecLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossVecLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLossVec);

}  // namespace caffe
