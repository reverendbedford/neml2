#pragma once

#include "StandardBatchedLabeledTensor.h"

class State : public StandardBatchedLabeledTensor
{
 public:
  /// Temporary, will remove
  State(const torch::Tensor & tensor);

  /// Temporary, will remove
  State(const torch::Tensor & tensor, std::map<std::string,TorchSlice> labels);

  template<typename T>
  struct item_type{ typedef T type; };

  /// As view but also interpret the view as an object
  template <typename T>
  typename item_type<T>::type get(std::string name)
  {
    return T(view(name));
  }
};
