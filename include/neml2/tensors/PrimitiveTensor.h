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

#include "neml2/tensors/Tensor.h"

namespace neml2
{
/**
 * @brief PrimitiveTensor inherits from TensorBase and additionally templates on the base shape.
 *
 * @tparam S Base shape
 */
template <class Derived, Size... S>
class PrimitiveTensor : public TensorBase<Derived>
{
public:
  /// The base shape
  static inline const TensorShape const_base_sizes = {S...};

  /// The base dim
  static constexpr Size const_base_dim = sizeof...(S);

  /// The base storage
  static inline const Size const_base_storage = utils::storage_size({S...});

  /// Default constructor
  PrimitiveTensor() = default;

  /// Construct from another torch::Tensor given batch dimension
  explicit PrimitiveTensor(const torch::Tensor & tensor, Size batch_dim);

  /// Construct from another torch::Tensor and infer batch dimension
  PrimitiveTensor(const torch::Tensor & tensor);

  /// Implicit conversion to a Tensor and loses information on the fixed base shape
  operator Tensor() const;

  /// Unbatched empty tensor
  [[nodiscard]] static Derived
  empty(const torch::TensorOptions & options = default_tensor_options());
  /// Empty tensor given batch shape
  [[nodiscard]] static Derived
  empty(TensorShapeRef batch_shape,
        const torch::TensorOptions & options = default_tensor_options());
  /// Unbatched zero tensor
  [[nodiscard]] static Derived
  zeros(const torch::TensorOptions & options = default_tensor_options());
  /// Zero tensor given batch shape
  [[nodiscard]] static Derived
  zeros(TensorShapeRef batch_shape,
        const torch::TensorOptions & options = default_tensor_options());
  /// Unbatched unit tensor
  [[nodiscard]] static Derived
  ones(const torch::TensorOptions & options = default_tensor_options());
  /// Unit tensor given batch shape
  [[nodiscard]] static Derived
  ones(TensorShapeRef batch_shape, const torch::TensorOptions & options = default_tensor_options());
  /// Unbatched tensor filled with a given value given base shape
  [[nodiscard]] static Derived
  full(Real init, const torch::TensorOptions & options = default_tensor_options());
  /// Full tensor given batch shape
  [[nodiscard]] static Derived
  full(TensorShapeRef batch_shape,
       Real init,
       const torch::TensorOptions & options = default_tensor_options());

  /// Derived tensor classes should define identity_map where appropriate
  [[nodiscard]] static Tensor identity_map(const torch::TensorOptions &)
  {
    throw NEMLException("Not implemented");
  }
};

///////////////////////////////////////////////////////////////////////////////
// Implementations
///////////////////////////////////////////////////////////////////////////////

template <class Derived, Size... S>
PrimitiveTensor<Derived, S...>::PrimitiveTensor(const torch::Tensor & tensor, Size batch_dim)
  : TensorBase<Derived>(tensor, batch_dim)
{
  neml_assert_dbg(this->base_sizes() == const_base_sizes,
                  "Base shape mismatch: trying to create a tensor with base shape ",
                  const_base_sizes,
                  " from a tensor with base shape ",
                  this->base_sizes());
}

template <class Derived, Size... S>
PrimitiveTensor<Derived, S...>::PrimitiveTensor(const torch::Tensor & tensor)
  : TensorBase<Derived>(tensor, tensor.dim() - const_base_dim)
{
  neml_assert_dbg(this->base_sizes() == const_base_sizes,
                  "Base shape mismatch: trying to create a tensor with base shape ",
                  const_base_sizes,
                  " from a tensor with shape ",
                  tensor.sizes());
}

template <class Derived, Size... S>
PrimitiveTensor<Derived, S...>::operator Tensor() const
{
  return Tensor(*this, this->batch_dim());
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::empty(const torch::TensorOptions & options)
{
  return Derived(torch::empty(const_base_sizes, options), 0);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::empty(TensorShapeRef batch_shape,
                                      const torch::TensorOptions & options)
{
  return Derived(torch::empty(utils::add_shapes(batch_shape, const_base_sizes), options),
                 batch_shape.size());
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::zeros(const torch::TensorOptions & options)
{
  return Derived(torch::zeros(const_base_sizes, options), 0);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::zeros(TensorShapeRef batch_shape,
                                      const torch::TensorOptions & options)
{
  return Derived(torch::zeros(utils::add_shapes(batch_shape, const_base_sizes), options),
                 batch_shape.size());
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::ones(const torch::TensorOptions & options)
{
  return Derived(torch::ones(const_base_sizes, options), 0);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::ones(TensorShapeRef batch_shape,
                                     const torch::TensorOptions & options)
{
  return Derived(torch::ones(utils::add_shapes(batch_shape, const_base_sizes), options),
                 batch_shape.size());
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::full(Real init, const torch::TensorOptions & options)
{
  return Derived(torch::full(const_base_sizes, init, options), 0);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::full(TensorShapeRef batch_shape,
                                     Real init,
                                     const torch::TensorOptions & options)
{
  return Derived(torch::full(utils::add_shapes(batch_shape, const_base_sizes), init, options),
                 batch_shape.size());
}
} // namespace neml2
