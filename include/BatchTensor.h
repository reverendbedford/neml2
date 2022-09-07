#pragma once

#include "types.h"

#include <stdexcept>

#include <torch/torch.h>

/// Tensor where the first index is a batch dimension
template <TorchSize N>
class BatchTensor : public torch::Tensor {
 public:
  /// Default constructor, empty with no batch dimensions
  BatchTensor();

  /// Construct from existing tensor, no batch dimensions
  BatchTensor(const torch::Tensor & tensor);

  /// Construct from existing tensor, specify batch dimensions
  BatchTensor(const torch::Tensor & tensor, TorchSize nbatch);

  /// Return the number of batch dimensions
  TorchSize nbatch() const;

  /// Return the batch size
  TorchShape batch_sizes() const;

  /// Return the base size
  TorchShape base_sizes() const;
};

template <TorchSize N>
BatchTensor<N>::BatchTensor() :
    torch::Tensor()
{

}

template <TorchSize N>
BatchTensor<N>::BatchTensor(const torch::Tensor & tensor) :
    torch::Tensor(tensor)
{
  // Check to make sure we can actually do this
  if (sizes().size() < N)
    throw std::runtime_error("Tensor dimension is smaller than the requested "
                             "number of batch dimensions");
}

template <TorchSize N>
TorchSize BatchTensor<N>::nbatch() const
{
  return N;
}

template <TorchSize N>
TorchShape BatchTensor<N>::batch_sizes() const
{
  return TorchShape(sizes().begin(), sizes().begin() + N);
}

template <TorchSize N>
TorchShape BatchTensor<N>::base_sizes() const
{
  return TorchShape(sizes().begin() + N, sizes().end());
}
