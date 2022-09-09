#pragma once

#include "FixedDimTensor.h"

// I'm still making my mind up on this, but for now let's just work
// with "batched" (single batch index) or unbatched (no batch index)
// templates
template <TorchSize ... D>
using StandardBatchedTensorBase = FixedDimTensor<1, D...>;

/// A specific implementation of the standard batched tensor
// Only goal here is to fill in a nice constructor on the batch size
template <TorchSize ... D>
class StandardBatchedTensor: public StandardBatchedTensorBase<D...> {
 public:
  StandardBatchedTensor(TorchSize nbatch);
  StandardBatchedTensor(const torch::Tensor & tensor);
  
  /// Helper to return the (now scalar) batch size
  TorchSize batch_size() const;
};

template <TorchSize ... D>
StandardBatchedTensor<D...>::StandardBatchedTensor(TorchSize nbatch) :
    StandardBatchedTensorBase<D...>(TorchShapeRef({nbatch}))
{

}

template <TorchSize ... D>
StandardBatchedTensor<D...>::StandardBatchedTensor(
    const torch::Tensor & tensor) : 
    StandardBatchedTensorBase<D...>(tensor)
{

}

template <TorchSize ... D>
TorchSize StandardBatchedTensor<D...>::batch_size() const
{
  return torch::Tensor::sizes()[0];
}
