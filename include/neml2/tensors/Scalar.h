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

#include "neml2/tensors/FixedDimTensor.h"

namespace neml2
{
/**
 * @brief The (logical) scalar
 *
 * The logical storage space is (), i.e., scalar.
 *
 */
class Scalar : public FixedDimTensor<Scalar>
{
public:
  using FixedDimTensor<Scalar>::FixedDimTensor;

  Scalar(Real init, const torch::TensorOptions & options);

  /// The derivative of a Scalar with respect to itself
  [[nodiscard]] static Scalar
  identity_map(const torch::TensorOptions & options = default_tensor_options);
};

template <
    class Derived,
    typename = typename std::enable_if_t<!std::is_same_v<Derived, Scalar>>,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator+(const Derived & a, const Scalar & b)
{
  neml_assert_batch_broadcastable_dbg(a, b);
  TorchSlice net{torch::indexing::Ellipsis};
  net.insert(net.end(), a.base_dim(), torch::indexing::None);
  return Derived(torch::operator+(a, b.index(net)), broadcast_batch_dim(a, b));
}

template <
    class Derived,
    typename = typename std::enable_if_t<!std::is_same_v<Derived, Scalar>>,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator+(const Scalar & a, const Derived & b)
{
  return b + a;
}

template <
    class Derived,
    typename = typename std::enable_if_t<!std::is_same_v<Derived, Scalar>>,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator-(const Derived & a, const Scalar & b)
{
  neml_assert_batch_broadcastable_dbg(a, b);
  TorchSlice net{torch::indexing::Ellipsis};
  net.insert(net.end(), a.base_dim(), torch::indexing::None);
  return Derived(torch::operator-(a, b.index(net)), broadcast_batch_dim(a, b));
}

template <
    class Derived,
    typename = typename std::enable_if_t<!std::is_same_v<Derived, Scalar>>,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator-(const Scalar & a, const Derived & b)
{
  return -b + a;
}

template <
    class Derived,
    typename = typename std::enable_if_t<!std::is_same_v<Derived, Scalar>>,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator*(const Derived & a, const Scalar & b)
{
  neml_assert_batch_broadcastable_dbg(a, b);
  TorchSlice net{torch::indexing::Ellipsis};
  net.insert(net.end(), a.base_dim(), torch::indexing::None);
  return Derived(torch::operator*(a, b.index(net)), broadcast_batch_dim(a, b));
}

template <
    class Derived,
    typename = typename std::enable_if_t<!std::is_same_v<Derived, Scalar>>,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator*(const Scalar & a, const Derived & b)
{
  return b * a;
}

template <
    class Derived,
    typename = typename std::enable_if_t<!std::is_same_v<Derived, Scalar>>,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator/(const Derived & a, const Scalar & b)
{
  neml_assert_batch_broadcastable_dbg(a, b);
  TorchSlice net{torch::indexing::Ellipsis};
  net.insert(net.end(), a.base_dim(), torch::indexing::None);
  return Derived(torch::operator/(a, b.index(net)), broadcast_batch_dim(a, b));
}

template <
    class Derived,
    typename = typename std::enable_if_t<!std::is_same_v<Derived, Scalar>>,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator/(const Scalar & a, const Derived & b)
{
  neml_assert_batch_broadcastable_dbg(a, b);
  TorchSlice net{torch::indexing::Ellipsis};
  net.insert(net.end(), b.base_dim(), torch::indexing::None);
  return Derived(torch::operator/(a.index(net), b), broadcast_batch_dim(a, b));
}

namespace math
{
template <
    class Derived,
    typename = typename std::enable_if_t<!std::is_same_v<Derived, Scalar>>,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
pow(const Derived & a, const Scalar & n)
{
  neml_assert_batch_broadcastable_dbg(a, n);
  TorchSlice net{torch::indexing::Ellipsis};
  net.insert(net.end(), a.base_dim(), torch::indexing::None);
  return Derived(torch::pow(a, n.index(net)), broadcast_batch_dim(a, n));
}

template <
    class Derived,
    typename = typename std::enable_if_t<!std::is_same_v<Derived, Scalar>>,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
pow(const Scalar & a, const Derived & n)
{
  neml_assert_batch_broadcastable_dbg(a, n);
  TorchSlice net{torch::indexing::Ellipsis};
  net.insert(net.end(), n.base_dim(), torch::indexing::None);
  return Derived(torch::pow(a.index(net), n), broadcast_batch_dim(a, n));
}
}
} // namespace neml2
