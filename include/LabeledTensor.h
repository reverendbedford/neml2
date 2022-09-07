#pragma once

#include "types.h"

#include <torch/torch.h>

#include <string>
#include <map>

class LabeledTensor : public torch::Tensor {
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
  torch::Tensor get(std::string label);

 protected:
  std::map<std::string,TorchSlice> labels_; 
};
