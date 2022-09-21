#pragma once

#include "types.h"
#include "BatchTensor.h"

#include <array>

/// Tensor with a dynamic batch dimension and fixed logical dimensions
template <TorchSize N, TorchSize  ... D>
class FixedDimTensor : public BatchTensor<N> {
 public:
  /// Default constructor
  FixedDimTensor();

  /// Make an empty tensor with given batch size
  FixedDimTensor(TorchShapeRef batch_size);

  /// Make from another tensor
  FixedDimTensor(const torch::Tensor & tensor);

  /// The actual (static) base shape 
  static inline const TorchShape base_shape{ {D...} };

 protected:
  /// Return what the full shape of the tensor should be, given the batch size
  std::vector<TorchSize> construct_sizes(TorchShapeRef batch_size) const;
};

template <TorchSize N, TorchSize ... D>
FixedDimTensor<N, D...>::FixedDimTensor() : 
    BatchTensor<N>()
{

}

template <TorchSize N, TorchSize ... D>
FixedDimTensor<N, D...>::FixedDimTensor(TorchShapeRef batch_size) :
    BatchTensor<N>(std::move(torch::empty(construct_sizes(batch_size),
                                          TorchDefaults)))
{
}

template <TorchSize N, TorchSize ... D>
FixedDimTensor<N, D...>::FixedDimTensor(const torch::Tensor & tensor) :
    BatchTensor<N>(tensor)
{
  // Check to make sure we got the correct base_sizes()
  if (base_shape != BatchTensor<N>::base_sizes())
    throw std::runtime_error("Base size of the supplied tensor "
                             "does not match the templated "
                             "base size");
}


template <TorchSize N, TorchSize ... D>
TorchShape FixedDimTensor<N, D...>::construct_sizes(TorchShapeRef batch_size) const
{
  // Quick check to make sure batch_size is consistent with N, this could become
  // a static assertion
  if (batch_size.size() != N)
    throw std::runtime_error("Proposed batch shape does not match "
                             "the number of templated batch dimensions");
  
  TorchShape total(batch_size.vec());
  total.insert(total.end(), base_shape.begin(), base_shape.end());
  return total;
}
