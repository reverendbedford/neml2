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

#include "neml2/tensors/FixedDimTensor.h"

// Derived classes to be instantiated
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R3.h"
#include "neml2/tensors/SFR3.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/SFFR4.h"
#include "neml2/tensors/R5.h"
#include "neml2/tensors/SSFR5.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/WR2.h"
#include "neml2/tensors/Quaternion.h"
#include "neml2/tensors/SWR4.h"
#include "neml2/tensors/WSR4.h"
#include "neml2/tensors/WWR4.h"
#include "neml2/models/crystallography/MillerIndex.h"

namespace neml2
{
template <class Derived, TorchSize... S>
FixedDimTensor<Derived, S...>::FixedDimTensor(const torch::Tensor & tensor, TorchSize batch_dim)
  : BatchTensorBase<Derived>(tensor, batch_dim)
{
  neml_assert_dbg(this->base_sizes() == const_base_sizes,
                  "Base shape mismatch: trying to create a tensor with base shape ",
                  const_base_sizes,
                  " from a tensor with base shape ",
                  this->base_sizes());
}

template <class Derived, TorchSize... S>
FixedDimTensor<Derived, S...>::FixedDimTensor(const torch::Tensor & tensor)
  : BatchTensorBase<Derived>(tensor, tensor.dim() - const_base_dim)
{
  neml_assert_dbg(this->base_sizes() == const_base_sizes,
                  "Base shape mismatch: trying to create a tensor with base shape ",
                  const_base_sizes,
                  " from a tensor with shape ",
                  tensor.sizes());
}

template <class Derived, TorchSize... S>
FixedDimTensor<Derived, S...>::operator BatchTensor() const
{
  return BatchTensor(*this, this->batch_dim());
}

template <class Derived, TorchSize... S>
Derived
FixedDimTensor<Derived, S...>::empty(const torch::TensorOptions & options)
{
  return Derived(torch::empty(const_base_sizes, options), 0);
}

template <class Derived, TorchSize... S>
Derived
FixedDimTensor<Derived, S...>::empty(TorchShapeRef batch_shape,
                                     const torch::TensorOptions & options)
{
  return Derived(torch::empty(utils::add_shapes(batch_shape, const_base_sizes), options),
                 batch_shape.size());
}

template <class Derived, TorchSize... S>
Derived
FixedDimTensor<Derived, S...>::zeros(const torch::TensorOptions & options)
{
  return Derived(torch::zeros(const_base_sizes, options), 0);
}

template <class Derived, TorchSize... S>
Derived
FixedDimTensor<Derived, S...>::zeros(TorchShapeRef batch_shape,
                                     const torch::TensorOptions & options)
{
  return Derived(torch::zeros(utils::add_shapes(batch_shape, const_base_sizes), options),
                 batch_shape.size());
}

template <class Derived, TorchSize... S>
Derived
FixedDimTensor<Derived, S...>::ones(const torch::TensorOptions & options)
{
  return Derived(torch::ones(const_base_sizes, options), 0);
}

template <class Derived, TorchSize... S>
Derived
FixedDimTensor<Derived, S...>::ones(TorchShapeRef batch_shape, const torch::TensorOptions & options)
{
  return Derived(torch::ones(utils::add_shapes(batch_shape, const_base_sizes), options),
                 batch_shape.size());
}

template <class Derived, TorchSize... S>
Derived
FixedDimTensor<Derived, S...>::full(Real init, const torch::TensorOptions & options)
{
  return Derived(torch::full(const_base_sizes, init, options), 0);
}

template <class Derived, TorchSize... S>
Derived
FixedDimTensor<Derived, S...>::full(TorchShapeRef batch_shape,
                                    Real init,
                                    const torch::TensorOptions & options)
{
  return Derived(torch::full(utils::add_shapes(batch_shape, const_base_sizes), init, options),
                 batch_shape.size());
}

template class FixedDimTensor<Scalar>;
template class FixedDimTensor<Vec, 3>;
template class FixedDimTensor<Rot, 3>;
template class FixedDimTensor<WR2, 3>;
template class FixedDimTensor<R2, 3, 3>;
template class FixedDimTensor<SR2, 6>;
template class FixedDimTensor<R3, 3, 3, 3>;
template class FixedDimTensor<SFR3, 6, 3>;
template class FixedDimTensor<R4, 3, 3, 3, 3>;
template class FixedDimTensor<SSR4, 6, 6>;
template class FixedDimTensor<SFFR4, 6, 3, 3>;
template class FixedDimTensor<R5, 3, 3, 3, 3, 3>;
template class FixedDimTensor<SSFR5, 6, 6, 3>;
template class FixedDimTensor<Quaternion, 4>;
template class FixedDimTensor<SWR4, 6, 3>;
template class FixedDimTensor<WSR4, 3, 6>;
template class FixedDimTensor<WWR4, 3, 3>;
template class FixedDimTensor<crystallography::MillerIndex, 3>;

} // namespace neml2
