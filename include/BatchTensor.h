#pragma once

#include "types.h"

#include <stdexcept>
#include <array>
#include <numeric>

#include <torch/torch.h>

/// Tensor where the first index is a batch dimension
template <TorchSize N>
class BatchTensor : public torch::Tensor {
 public:
  /// Default constructor, empty with no batch dimensions
  BatchTensor();

  /// Construct from existing tensor, no batch dimensions
  BatchTensor(const torch::Tensor & tensor);

  /// Return the number of base dimensions
  virtual TorchSize nbase() const;

  /// Return the number of batch dimensions
  virtual TorchSize nbatch() const;

  /// Return the batch size
  virtual TorchShape batch_sizes() const;

  /// Return the base size
  virtual TorchShape base_sizes() const;

  /// Return an index sliced on the batch dimensions
  virtual torch::Tensor base_index(TorchSlice indices);

 private:
  /// Add a slice on the batch dimensions to an index
  TorchSlice make_slice(TorchSlice base) const;

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
TorchSize BatchTensor<N>::nbase() const
{
  return sizes().size() - nbatch();
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

template <TorchSize N>
torch::Tensor BatchTensor<N>::base_index(TorchSlice indices)
{
  return torch::Tensor::index(make_slice(indices));
}

template <TorchSize N>
TorchSlice BatchTensor<N>::make_slice(TorchSlice base) const
{
  TorchSlice front(N, torch::indexing::Slice());
  front.insert(front.end(), base.begin(), base.end());
  return front;
}

/// Helper to get the total storage required from a TorchShape
inline TorchSize storage(const TorchShape & shape)
{
  TorchSize sz = 1;
  return std::accumulate(shape.begin(), shape.end(), sz,
                         std::multiplies<TorchSize>());
}
