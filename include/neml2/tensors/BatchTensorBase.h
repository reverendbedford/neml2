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

#include "neml2/misc/utils.h"

namespace neml2
{
// Forward declarations
template <class Derived>
class BatchTensorBase;

class BatchTensor;

/**
 * @brief NEML2's enhanced tensor type.
 *
 * BatchTensorBase derives from torch::Tensor and clearly distinguishes between "batched" dimensions
 * from other dimensions. The shape of the "batched" dimensions is called the batch size, and the
 * shape of the rest dimensions is called the base size.
 */
template <class Derived>
class BatchTensorBase : public torch::Tensor
{
public:
  /// Default constructor
  BatchTensorBase() = default;

  /// Construct from another torch::Tensor
  BatchTensorBase(const torch::Tensor & tensor, TorchSize batch_dim);

  /// Copy constructor
  BatchTensorBase(const Derived & tensor);

  BatchTensorBase(Real) = delete;

  /// Unbatched empty tensor given base shape
  [[nodiscard]] static Derived
  empty(TorchShapeRef base_shape, const torch::TensorOptions & options = default_tensor_options());
  /// Empty tensor given batch and base shapes
  [[nodiscard]] static Derived
  empty(TorchShapeRef batch_shape,
        TorchShapeRef base_shape,
        const torch::TensorOptions & options = default_tensor_options());
  /// Empty tensor like another, i.e. same batch and base shapes, same tensor options, etc.
  [[nodiscard]] static Derived empty_like(const BatchTensorBase<Derived> & other);
  /// Unbatched tensor filled with zeros given base shape
  [[nodiscard]] static Derived
  zeros(TorchShapeRef base_shape, const torch::TensorOptions & options = default_tensor_options());
  /// Zero tensor given batch and base shapes
  [[nodiscard]] static Derived
  zeros(TorchShapeRef batch_shape,
        TorchShapeRef base_shape,
        const torch::TensorOptions & options = default_tensor_options());
  /// Zero tensor like another, i.e. same batch and base shapes, same tensor options, etc.
  [[nodiscard]] static Derived zeros_like(const BatchTensorBase<Derived> & other);
  /// Unbatched tensor filled with ones given base shape
  [[nodiscard]] static Derived
  ones(TorchShapeRef base_shape, const torch::TensorOptions & options = default_tensor_options());
  /// Unit tensor given batch and base shapes
  [[nodiscard]] static Derived
  ones(TorchShapeRef batch_shape,
       TorchShapeRef base_shape,
       const torch::TensorOptions & options = default_tensor_options());
  /// Unit tensor like another, i.e. same batch and base shapes, same tensor options, etc.
  [[nodiscard]] static Derived ones_like(const BatchTensorBase<Derived> & other);
  /// Unbatched tensor filled with a given value given base shape
  [[nodiscard]] static Derived
  full(TorchShapeRef base_shape,
       Real init,
       const torch::TensorOptions & options = default_tensor_options());
  /// Full tensor given batch and base shapes
  [[nodiscard]] static Derived
  full(TorchShapeRef batch_shape,
       TorchShapeRef base_shape,
       Real init,
       const torch::TensorOptions & options = default_tensor_options());
  /// Full tensor like another, i.e. same batch and base shapes, same tensor options, etc.,
  /// but filled with a different value
  [[nodiscard]] static Derived full_like(const BatchTensorBase<Derived> & other, Real init);
  /// Unbatched identity tensor
  [[nodiscard]] static Derived
  identity(TorchSize n, const torch::TensorOptions & options = default_tensor_options());
  /// Identity tensor given batch shape and base length
  [[nodiscard]] static Derived
  identity(TorchShapeRef batch_shape,
           TorchSize n,
           const torch::TensorOptions & options = default_tensor_options());
  /**
   * @brief Create a new tensor by adding a new batch dimension with linear spacing between \p
   * start and \p end.
   *
   * \p start and \p end must be broadcastable. The new batch dimension will be added at the
   * user-specified dimension \p dim which defaults to 0.
   *
   * For example, if \p start has shape `(3, 2; 5, 5)` and \p end has shape `(3, 1; 5, 5)`, then
   * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~cpp
   * linspace(start, end, 100, 1);
   * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   * will have shape `(3, 100, 2; 5, 5)`, note the location of the new dimension and the
   * broadcasting.
   *
   * @param start The starting tensor
   * @param end The ending tensor
   * @param nstep The number of steps with even spacing along the new dimension
   * @param dim Where to insert the new dimension
   * @param batch_dim Batch dimension of the output
   * @return BatchTensor Linearly spaced tensor
   */
  [[nodiscard]] static Derived linspace(const Derived & start,
                                        const Derived & end,
                                        TorchSize nstep,
                                        TorchSize dim = 0,
                                        TorchSize batch_dim = -1);
  /// log-space equivalent of the linspace named constructor
  [[nodiscard]] static Derived logspace(const Derived & start,
                                        const Derived & end,
                                        TorchSize nstep,
                                        TorchSize dim = 0,
                                        TorchSize batch_dim = -1,
                                        Real base = 10);

  /// Whether the tensor is batched
  bool batched() const;

  /// Return the number of batch dimensions
  TorchSize batch_dim() const;

  /// Return a writable reference to the batch dimension
  TorchSize & batch_dim();

  /// Return the number of base dimensions
  TorchSize base_dim() const;

  /// Return the batch size
  TorchShapeRef batch_sizes() const;

  /// Return the length of some batch axis
  TorchSize batch_size(TorchSize index) const;

  /// Return the base size
  TorchShapeRef base_sizes() const;

  /// Return the length of some base axis
  TorchSize base_size(TorchSize index) const;

  /// Return the flattened storage needed just for the base indices
  TorchSize base_storage() const;

  /// Get a batch
  Derived batch_index(TorchSlice indices) const;

  /// Return an index sliced on the batch dimensions
  BatchTensor base_index(const TorchSlice & indices) const;

  /// Set a index sliced on the batch dimensions to a value
  void batch_index_put(TorchSlice indices, const torch::Tensor & other);

  /// Set a index sliced on the batch dimensions to a value
  void base_index_put(const TorchSlice & indices, const torch::Tensor & other);

  /// Return a new view of the tensor with values broadcast along the batch dimensions.
  Derived batch_expand(TorchShapeRef batch_size) const;

  /// Return a new view of the tensor with values broadcast along the base dimensions.
  BatchTensor base_expand(TorchShapeRef base_size) const;

  /// Expand the batch to have the same shape as another tensor
  template <class Derived2>
  Derived batch_expand_as(const BatchTensorBase<Derived2> & other) const;

  /// Expand the base to have the same shape as another tensor
  BatchTensor base_expand_as(const BatchTensor & other) const;

  /// Return a new tensor with values broadcast along the batch dimensions.
  Derived batch_expand_copy(TorchShapeRef batch_size) const;

  /// Return a new tensor with values broadcast along the base dimensions.
  BatchTensor base_expand_copy(TorchShapeRef base_size) const;

  /// Reshape batch dimensions
  Derived batch_reshape(TorchShapeRef batch_shape) const;

  /// Reshape base dimensions
  BatchTensor base_reshape(TorchShapeRef base_shape) const;

  /// Unsqueeze a batch dimension
  Derived batch_unsqueeze(TorchSize d) const;

  /// Unsqueeze on the special list batch dimension
  Derived list_unsqueeze() const;

  /// Unsqueeze a base dimension
  BatchTensor base_unsqueeze(TorchSize d) const;

  /// Transpose two batch dimensions
  Derived batch_transpose(TorchSize d1, TorchSize d2) const;

  /// Transpose two base dimensions
  BatchTensor base_transpose(TorchSize d1, TorchSize d2) const;

  /// Move two base dimensions
  BatchTensor base_movedim(TorchSize d1, TorchSize d2) const;

  /// Clone (take ownership)
  Derived clone(torch::MemoryFormat memory_format = torch::MemoryFormat::Contiguous) const;

  /// Discard function graph
  Derived detach() const;

  /// Send to options
  Derived to(const torch::TensorOptions & options) const;

  /// Negation
  Derived operator-() const;

  /// Sum on a batch index
  Derived batch_sum(TorchSize d) const;

  /// Sum on the list index (TODO: replace with class)
  Derived list_sum() const;

private:
  /// Number of batch dimensions. The first `_batch_dim` dimensions are considered batch dimensions.
  TorchSize _batch_dim;
};

template <class Derived>
template <class Derived2>
Derived
BatchTensorBase<Derived>::batch_expand_as(const BatchTensorBase<Derived2> & other) const
{
  return batch_expand(other.batch_sizes());
}

template <class Derived,
          typename = typename std::enable_if<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator+(const Derived & a, const Real & b)
{
  return Derived(torch::operator+(a, b), a.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator+(const Real & a, const Derived & b)
{
  return b + a;
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator+(const Derived & a, const Derived & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Derived(torch::operator+(a, b), broadcast_batch_dim(a, b));
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator-(const Derived & a, const Real & b)
{
  return Derived(torch::operator-(a, b), a.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator-(const Real & a, const Derived & b)
{
  return -b + a;
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator-(const Derived & a, const Derived & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Derived(torch::operator-(a, b), broadcast_batch_dim(a, b));
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator*(const Derived & a, const Real & b)
{
  return Derived(torch::operator*(a, b), a.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator*(const Real & a, const Derived & b)
{
  return b * a;
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator/(const Derived & a, const Real & b)
{
  return Derived(torch::operator/(a, b), a.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator/(const Real & a, const Derived & b)
{
  return Derived(torch::operator/(a, b), b.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
operator/(const Derived & a, const Derived & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Derived(torch::operator/(a, b), broadcast_batch_dim(a, b));
}

namespace math
{
template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
pow(const Derived & a, const Real & n)
{
  return Derived(torch::pow(a, n), a.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
pow(const Real & a, const Derived & n)
{
  return Derived(torch::pow(a, n), n.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
pow(const Derived & a, const Derived & n)
{
  neml_assert_broadcastable_dbg(a, n);
  return Derived(torch::pow(a, n), broadcast_batch_dim(a, n));
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
sign(const Derived & a)
{
  return Derived(torch::sign(a), a.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
cosh(const Derived & a)
{
  return Derived(torch::cosh(a), a.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
sinh(const Derived & a)
{
  return Derived(torch::sinh(a), a.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
where(const torch::Tensor & condition, const Derived & a, const Derived & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Derived(torch::where(condition, a, b), broadcast_batch_dim(a, b));
}

/**
 * This is (almost) equivalent to Torch's heaviside, except that the Torch's version is not
 * differentiable (back-propagatable). I said "almost" because torch::heaviside allows you to set
 * the return value in the case of input == 0. Our implementation always return 0.5 when the input
 * == 0.
 */
template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
heaviside(const Derived & a)
{
  return (sign(a) + 1.0) / 2.0;
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
macaulay(const Derived & a)
{
  return Derived(torch::Tensor(a) * torch::Tensor(heaviside(a)), a.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
dmacaulay(const Derived & a)
{
  return heaviside(a);
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
sqrt(const Derived & a)
{
  return Derived(torch::sqrt(a), a.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
exp(const Derived & a)
{
  return Derived(torch::exp(a), a.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
abs(const Derived & a)
{
  return Derived(torch::abs(a), a.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
diff(const Derived & a, TorchSize n = 1, TorchSize dim = -1)
{
  return Derived(torch::diff(a, n, dim), a.batch_dim());
}

template <
    class Derived,
    typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<Derived>, Derived>>>
Derived
batch_diag_embed(const Derived & a, TorchSize offset = 0, TorchSize d1 = -2, TorchSize d2 = -1)
{
  return Derived(torch::diag_embed(
                     a, offset, d1 < 0 ? d1 - a.base_dim() : d1, d2 < 0 ? d2 - a.base_dim() : d2),
                 a.batch_dim() + 1);
}
} // namespace math
} // namespace neml2
