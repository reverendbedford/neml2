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
#include "neml2/misc/utils.h"
#include "neml2/misc/math.h"
#include "neml2/misc/error.h"

#include <stdexcept>
#include <array>
#include <numeric>

#include <torch/torch.h>

namespace neml2
{
inline torch::Tensor
einsum(const std::initializer_list<torch::Tensor> & tensors,
       const std::initializer_list<c10::string_view> & indices,
       c10::string_view out_indices = "")
{
  neml_assert_dbg(tensors.size() != 0, "You must provide tensors in einsum.");
  neml_assert_dbg(indices.size() == tensors.size(),
                  "Number of tensor indices must match number of tensors in einsum.\n",
                  "You provided ",
                  indices.size(),
                  " indices, and ",
                  tensors.size(),
                  " tensors.");

  std::string equation = "";

  // Form the equation in Einstein notation
  for (const auto & i : indices)
    equation += "..." + std::string(i) + ",";
  equation.pop_back();
  if (!out_indices.empty())
    equation += "->..." + std::string(out_indices);

  return torch::einsum(equation, tensors);
}

/**
 * @brief NEML2's enhanced tensor type.
 *
 * BatchTensor derives from torch::Tensor and clearly distinguishes between "batched" dimensions
 * from other dimensions. The shape of the "batched" dimensions is called the batch size, and the
 * shape of the rest dimensions is called the base size.
 *
 * @tparam N The number of batch dimensions.
 */
template <TorchSize N>
class BatchTensor : public torch::Tensor
{
public:
  /// Default constructor, empty with no batch dimensions
  BatchTensor(const torch::TensorOptions & options = default_tensor_options);

  /// Construct from existing tensor
  BatchTensor(const torch::Tensor & tensor);

  BatchTensor(Real) = delete;

  /**
   * @brief Make an empty batched tensor given batch size and base size
   *
   * @param batch_size Batch size
   * @param base_size Base size
   * @param options Tensor options
   */
  BatchTensor(TorchShapeRef batch_size,
              TorchShapeRef base_size,
              const torch::TensorOptions & options = default_tensor_options);

  /**
   * @brief Make an empty single-batch tensor given base size
   *
   * @param base_size Base size
   * @param options Tensor options
   */
  BatchTensor(TorchShapeRef base_size,
              const torch::TensorOptions & options = default_tensor_options);

  /// Return the number of base dimensions
  TorchSize base_dim() const;

  /// Return the number of batch dimensions
  constexpr TorchSize batch_dim() const;

  /// Return the base size
  TorchShape base_sizes() const;

  /// Return the batch size
  TorchShape batch_sizes() const;

  /// Return the flattened storage needed just for the base indices
  TorchSize base_storage() const;

  /// Get a batch
  torch::Tensor batch_index(TorchSlice indices) const;

  /// Set a index sliced on the batch dimensions to a value
  void batch_index_put(TorchSlice indices, const torch::Tensor & other);

  /// Return an index sliced on the batch dimensions
  BatchTensor<N> base_index(TorchSlice indices) const;

  /// Set a index sliced on the batch dimensions to a value
  void base_index_put(TorchSlice indices, const torch::Tensor & other);

  /// Return a new view of the tensor with values broadcast along the batch dimensions.
  BatchTensor<N> batch_expand(TorchShapeRef batch_size) const;

  /// Return a new tensor with values broadcast along the batch dimensions.
  BatchTensor<N> batch_expand_copy(TorchShapeRef batch_size) const;

  /// Negation
  BatchTensor<N> operator-() const;

  /// Raise to a power
  BatchTensor<N> pow(const BatchTensor<N> & n) const;

  /// Identity
  static BatchTensor<N> identity(TorchSize n,
                                 const torch::TensorOptions & options = default_tensor_options);
};

template <TorchSize N>
BatchTensor<N>::BatchTensor(const torch::TensorOptions & options)
  : BatchTensor<N>(TorchShapeRef({}), options)
{
}

template <TorchSize N>
BatchTensor<N>::BatchTensor(const torch::Tensor & tensor)
  : torch::Tensor(tensor)
{
  neml_assert_dbg(sizes().size() >= N,
                  "Tensor dimension ",
                  sizes().size(),
                  " is smaller than the requested number of batch dimensions ",
                  N);
}

template <TorchSize N>
BatchTensor<N>::BatchTensor(TorchShapeRef batch_size,
                            TorchShapeRef base_size,
                            const torch::TensorOptions & options)
  : torch::Tensor(torch::zeros(utils::add_shapes(batch_size, base_size), options))
{
  neml_assert_dbg(batch_size.size() == N,
                  "Proposed batch shape has dimension ",
                  batch_size.size(),
                  ". It does not match the number of templated batch dimensions ",
                  N);
}

template <TorchSize N>
BatchTensor<N>::BatchTensor(TorchShapeRef base_size, const torch::TensorOptions & options)
  : torch::Tensor(torch::zeros(
        utils::add_shapes(TorchShapeRef{std::vector<TorchSize>(N, 1)}, base_size), options))
{
}

template <TorchSize N>
TorchSize
BatchTensor<N>::base_dim() const
{
  return sizes().size() - batch_dim();
}

template <TorchSize N>
constexpr TorchSize
BatchTensor<N>::batch_dim() const
{
  return N;
}

template <TorchSize N>
TorchShape
BatchTensor<N>::base_sizes() const
{
  return TorchShape(sizes().begin() + N, sizes().end());
}

template <TorchSize N>
TorchShape
BatchTensor<N>::batch_sizes() const
{
  return TorchShape(sizes().begin(), sizes().begin() + N);
}

template <TorchSize N>
TorchSize
BatchTensor<N>::base_storage() const
{
  return storage_size(base_sizes());
}

template <TorchSize N>
torch::Tensor
BatchTensor<N>::batch_index(TorchSlice indices) const
{
  indices.insert(indices.end(), torch::indexing::Ellipsis);
  indices.insert(indices.end(), base_dim(), torch::indexing::Slice());
  auto res = this->index(indices);
  return res.dim() == base_dim() ? res.unsqueeze(0) : res;
}

template <TorchSize N>
void
BatchTensor<N>::batch_index_put(TorchSlice indices, const torch::Tensor & other)
{
  indices.insert(indices.end(), torch::indexing::Ellipsis);
  batch_index(indices).copy_(other);
}

template <TorchSize N>
BatchTensor<N>
BatchTensor<N>::base_index(TorchSlice indices) const
{
  indices.insert(indices.begin(), batch_dim(), torch::indexing::Slice());
  auto res = this->index(indices);
  return res.dim() == batch_dim() ? res.unsqueeze(-1) : res;
}

template <TorchSize N>
void
BatchTensor<N>::base_index_put(TorchSlice indices, const torch::Tensor & other)
{
  indices.insert(indices.begin(), torch::indexing::Ellipsis);
  base_index(indices).copy_(other);
}

template <TorchSize N>
BatchTensor<N>
BatchTensor<N>::batch_expand(TorchShapeRef batch_size) const
{
  neml_assert_dbg(batch_size.size() == N,
                  "Proposed batch shape has dimension ",
                  batch_size.size(),
                  ". It does not match the number of templated batch dimensions ",
                  N);

  // We don't want to touch the base dimensions, so put -1 for them.
  TorchShape net(batch_size.vec());
  net.insert(net.end(), base_dim(), -1);
  return expand(net);
}

template <TorchSize N>
BatchTensor<N>
BatchTensor<N>::batch_expand_copy(TorchShapeRef batch_size) const
{
  neml_assert_dbg(batch_size.size() == N,
                  "Proposed batch shape has dimension ",
                  batch_size.size(),
                  ". It does not match the number of templated batch dimensions ",
                  N);

  // We don't want to touch the base dimensions, so put -1 for them.
  auto res = torch::empty(utils::add_shapes(batch_size, base_sizes()), options());
  res.copy_(batch_expand(batch_size));
  return res;
}

template <TorchSize N>
BatchTensor<N>
BatchTensor<N>::operator-() const
{
  return -torch::Tensor(*this);
}

template <TorchSize N>
BatchTensor<N>
BatchTensor<N>::pow(const BatchTensor<N> & n) const
{
  return torch::pow(*this, n);
}

template <TorchSize N>
BatchTensor<N>
BatchTensor<N>::identity(TorchSize n, const torch::TensorOptions & options)
{
  return torch::eye(n, options).index(TorchSlice(N, torch::indexing::None));
}

template <TorchSize N>
BatchTensor<N>
operator+(const BatchTensor<N> & a, const Real & b)
{
  return torch::operator+(a, b);
}

template <TorchSize N>
BatchTensor<N>
operator+(const Real & a, const BatchTensor<N> & b)
{
  return torch::operator+(a, b);
}

template <TorchSize N>
BatchTensor<N>
operator+(const BatchTensor<N> & a, const BatchTensor<N> & b)
{
  return torch::operator+(a, b);
}

template <TorchSize N>
BatchTensor<N>
operator-(const BatchTensor<N> & a, const Real & b)
{
  return torch::operator-(a, b);
}

template <TorchSize N>
BatchTensor<N>
operator-(const Real & a, const BatchTensor<N> & b)
{
  return torch::operator-(a, b);
}

template <TorchSize N>
BatchTensor<N>
operator-(const BatchTensor<N> & a, const BatchTensor<N> & b)
{
  return torch::operator-(a, b);
}

template <TorchSize N>
BatchTensor<N>
operator*(const BatchTensor<N> & a, const Real & b)
{
  return torch::operator*(a, b);
}

template <TorchSize N>
BatchTensor<N>
operator*(const Real & a, const BatchTensor<N> & b)
{
  return torch::operator*(a, b);
}

template <TorchSize N>
BatchTensor<N>
operator*(const BatchTensor<N> & a, const BatchTensor<N> & b)
{
  return torch::operator*(a, b);
}

template <TorchSize N>
BatchTensor<N>
operator/(const BatchTensor<N> & a, const Real & b)
{
  return torch::operator/(a, b);
}

template <TorchSize N>
BatchTensor<N>
operator/(const Real & a, const BatchTensor<N> & b)
{
  return torch::operator/(a, b);
}

template <TorchSize N>
BatchTensor<N>
operator/(const BatchTensor<N> & a, const BatchTensor<N> & b)
{
  return torch::operator/(a, b);
}
} // namespace neml2
