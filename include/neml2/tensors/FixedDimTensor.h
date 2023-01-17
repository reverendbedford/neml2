// Copyright 2023, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy 
// of this software and associated documentation files (the "Software"), to deal 
// in the Software without restriction, including without limitation the rights 
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
// copies of the Software, and to permit persons to whom the Software is 
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
// THE SOFTWARE.

#pragma once

#include "neml2/misc/types.h"
#include "neml2/tensors/BatchTensor.h"

#include <array>

namespace neml2
{
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
