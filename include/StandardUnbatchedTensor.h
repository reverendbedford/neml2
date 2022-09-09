#pragma once

#include "FixedDimTensor.h"

// A standard  tensor with no batch dimensions
template <TorchSize ...D>
using StandardUnbatchedTensorBase = FixedDimTensor<0, D...>;

/// A specific implementation to define a nice constructor
template <TorchSize ... D>
class StandardUnbatchedTensor: public StandardUnbatchedTensorBase<D...> {
 public:
  StandardUnbatchedTensor();
  StandardUnbatchedTensor(const torch::Tensor & tensor);
};

template <TorchSize ... D>
StandardUnbatchedTensor<D...>::StandardUnbatchedTensor() :
    StandardUnbatchedTensorBase<D...>(TorchShapeRef({}))
{

}

template <TorchSize ... D>
StandardUnbatchedTensor<D...>::StandardUnbatchedTensor(const torch::Tensor &
                                                       tensor) :
    StandardUnbatchedTensorBase<D...>(tensor)
{

}
