#pragma once

#include "neml2/misc/types.h"
#include "neml2/tensors/BatchTensor.h"

#include <array>

namespace neml2
{
/// Tensor with a dynamic batch dimension and fixed logical dimensions
template <TorchSize N, TorchSize... D>
class FixedDimTensor : public BatchTensor<N>
{
public:
  /// Default constructor
  FixedDimTensor();

  /// Make from another tensor
  FixedDimTensor(const torch::Tensor & tensor);

  /// Make a batched tensor filled with default base tensor
  FixedDimTensor(const torch::Tensor & tensor, TorchShapeRef batch_size);

  /// The base shape
  static inline const TorchShape _base_sizes = TorchShape({D...});
};

template <TorchSize N, TorchSize... D>
FixedDimTensor<N, D...>::FixedDimTensor()
  : BatchTensor<N>(TorchShapeRef{std::vector<TorchSize>(N, 1)}, TorchShapeRef({D...}))
{
}

template <TorchSize N, TorchSize... D>
FixedDimTensor<N, D...>::FixedDimTensor(const torch::Tensor & tensor)
  : BatchTensor<N>(tensor)
{
  // Check to make sure we got the correct base_sizes()
  if (_base_sizes != this->base_sizes())
    throw std::runtime_error("Base size of the supplied tensor "
                             "does not match the templated "
                             "base size");
}

template <TorchSize N, TorchSize... D>
FixedDimTensor<N, D...>::FixedDimTensor(const torch::Tensor & tensor, TorchShapeRef batch_size)
  : BatchTensor<N>(tensor, batch_size)
{
  // Check to make sure we got the correct base_sizes()
  if (_base_sizes != this->base_sizes())
    throw std::runtime_error("Base size of the supplied tensor "
                             "does not match the templated "
                             "base size");
}
} // namespace neml2
