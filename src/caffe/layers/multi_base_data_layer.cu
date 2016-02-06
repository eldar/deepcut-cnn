#include <vector>

#include "caffe/pose_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiBasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  MultiBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  for (int k = 0; k < num_labels_; ++k) {
    // Reshape to loaded labels.
    top[1+k]->ReshapeLike(batch->labels_[k]);
    // Copy the labels.
    caffe_copy(batch->labels_[k].count(), batch->labels_[k].cpu_data(),
        top[1+k]->mutable_cpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(MultiBasePrefetchingDataLayer);

}  // namespace caffe
