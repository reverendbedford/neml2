#pragma once

#include "types.h"
#include "BatchTensor.h"

#include <string>
#include <unordered_map>

template <TorchSize N>
class LabeledTensor
{
public:
  /// Construct from a tensor without labels
  LabeledTensor(const torch::Tensor & tensor);

  /// Construct from a tensor with labels
  LabeledTensor(const torch::Tensor & tensor,
                const std::unordered_map<std::string, TorchSlice> & labels);

  /// Associate a label with a slice of the tensor
  void add_label(const std::string & label, const TorchSlice & indices);

  /// Return a labeled view into the tensor
  BatchTensor<N> operator[](const std::string & label) const;

  const BatchTensor<N> & tensor() const { return tensor_; }
  BatchTensor<N> & tensor() { return tensor_; }

  const std::unordered_map<std::string, TorchSlice> & labels() const { return labels_; }

protected:
  /// The underlying tensor (without the labels)
  BatchTensor<N> tensor_;

  /// The label-to-view map
  std::unordered_map<std::string, TorchSlice> labels_;
};

template <TorchSize N>
LabeledTensor<N>::LabeledTensor(const torch::Tensor & tensor)
  : tensor_(tensor)
{
}

template <TorchSize N>
LabeledTensor<N>::LabeledTensor(const torch::Tensor & tensor,
                                const std::unordered_map<std::string, TorchSlice> & labels)
  : tensor_(tensor),
    labels_(labels)
{
}

template <TorchSize N>
void
LabeledTensor<N>::add_label(const std::string & label, const TorchSlice & indices)
{
  auto p = labels_.insert({label, indices});
  if (!p.second)
    throw std::runtime_error("Attempted to insert duplicate key " + label +
                             " into a LabeledTensor");
}

template <TorchSize N>
BatchTensor<N>
LabeledTensor<N>::operator[](const std::string & label) const
{
  return tensor_.base_index(labels_.at(label));
}
