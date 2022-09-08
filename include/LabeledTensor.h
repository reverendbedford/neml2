#pragma once

#include "types.h"
#include "BatchTensor.h"

#include <string>
#include <map>

template <TorchSize N>
class LabeledTensor : public BatchTensor<N> {
 public:
  /// Default constructor
  LabeledTensor();
  
  /// Construct from a tensor without labels
  LabeledTensor(const torch::Tensor & tensor);

  /// Construct from a tensor with labels
  LabeledTensor(const torch::Tensor & tensor, 
                std::map<std::string,TorchSlice> labels);
  
  /// Associate a label with a slice of the tensor
  void add_label(std::string label, TorchSlice indices);

  /// Return a labeled view into the tensor
  BatchTensor<N> get(std::string label);

 protected:
  std::map<std::string,TorchSlice> labels_; 
};

template <TorchSize N>
LabeledTensor<N>::LabeledTensor() :
    BatchTensor<N>()
{

}

template <TorchSize N>
LabeledTensor<N>::LabeledTensor(const torch::Tensor & tensor) :
    BatchTensor<N>(tensor)
{

}

template <TorchSize N>
LabeledTensor<N>::LabeledTensor(const torch::Tensor & tensor, 
                             std::map<std::string,TorchSlice> labels) :
    BatchTensor<N>(tensor), labels_(labels)
{

}

template <TorchSize N>
void LabeledTensor<N>::add_label(std::string label, TorchSlice indices)
{
  labels_.insert({label, indices});
}


template <TorchSize N>
BatchTensor<N> LabeledTensor<N>::get(std::string label)
{
  return BatchTensor<N>::base_index(labels_.at(label));
}
