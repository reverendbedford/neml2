#pragma once

#include "types.h"
#include "BatchTensor.h"

#include <string>
#include <map>

template <TorchSize N>
class LabeledTensor : public BatchTensor<N>
{
public:
  /// Construct from a tensor without labels
  LabeledTensor(const torch::Tensor & tensor);

  /// Construct from a tensor with labels
  LabeledTensor(const torch::Tensor & tensor, std::map<std::string, TorchSlice> labels);

  /// Associate a label with a slice of the tensor
  virtual void add_label(std::string label, TorchSlice indices);

  /// Return a labeled view into the tensor
  virtual BatchTensor<N> get_view(std::string label) const;

  /// Set a labeled view into the tensor
  virtual void set_view(std::string label, const torch::Tensor & tensor);

protected:
  std::map<std::string, TorchSlice> labels_;
};

template <TorchSize N>
LabeledTensor<N>::LabeledTensor(const torch::Tensor & tensor)
  : BatchTensor<N>(tensor)
{
}

template <TorchSize N>
LabeledTensor<N>::LabeledTensor(const torch::Tensor & tensor,
                                std::map<std::string, TorchSlice> labels)
  : BatchTensor<N>(tensor),
    labels_(labels)
{
}

template <TorchSize N>
void
LabeledTensor<N>::add_label(std::string label, TorchSlice indices)
{
  auto p = labels_.insert({label, indices});
  if (!p.second)
    throw std::runtime_error("Attempted to insert duplicate key " + label +
                             " into a LabeledTensor");
}

template <TorchSize N>
BatchTensor<N>
LabeledTensor<N>::get_view(std::string label) const
{
  return BatchTensor<N>::base_index(labels_.at(label));
}

template <TorchSize N>
void
LabeledTensor<N>::set_view(std::string label, const torch::Tensor & tensor)
{
  BatchTensor<N>::base_index_put(labels_.at(label), tensor);
}
