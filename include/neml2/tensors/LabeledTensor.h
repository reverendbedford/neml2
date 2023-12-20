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
#include "neml2/tensors/LabeledAxis.h"
#include "neml2/tensors/BatchTensor.h"

namespace neml2
{
/**
 * @brief The primary data structure in NEML2 for working with labeled tensor views.
 *
 * Each LabeledTensor consists of one BatchTensor and one or more LabeledAxis. The
 * `LabeledTensor<D>` is templated on the base dimension \f$D\f$. LabeledTensor handles the
 * creation, modification, and accessing of labeled tensors.
 *
 * @tparam D The number of base dimensions
 */
template <class Derived, TorchSize D>
class LabeledTensor
{
public:
  /// Default constructor
  LabeledTensor() = default;

  /// Construct from a Tensor with batch dim and vector of `LabeledAxis`
  LabeledTensor(const torch::Tensor & tensor,
                TorchSize batch_dim,
                const std::vector<const LabeledAxis *> & axes);

  /// Construct from a BatchTensor with vector of `LabeledAxis`
  LabeledTensor(const BatchTensor & tensor, const std::vector<const LabeledAxis *> & axes);

  /// Copy constructor
  LabeledTensor(const Derived & other);

  /// A potentially dangerous implicit conversion
  // Should we mark it explicit?
  operator BatchTensor() const;

  /// A potentially dangerous implicit conversion
  // Should we mark it explicit?
  operator torch::Tensor() const;

  /// Setup new empty storage
  [[nodiscard]] static Derived
  empty(TorchShapeRef batch_shape,
        const std::vector<const LabeledAxis *> & axes,
        const torch::TensorOptions & options = default_tensor_options());

  /// Setup new empty storage like another LabeledTensor
  [[nodiscard]] static Derived empty_like(const Derived & other);

  /// Setup new storage with zeros
  [[nodiscard]] static Derived
  zeros(TorchShapeRef batch_shape,
        const std::vector<const LabeledAxis *> & axes,
        const torch::TensorOptions & options = default_tensor_options());

  /// Setup new storage with zeros like another LabeledTensor
  [[nodiscard]] static Derived zeros_like(const Derived & other);

  /// Clone this LabeledTensor
  Derived clone(torch::MemoryFormat memory_format = torch::MemoryFormat::Contiguous) const;

  template <typename T>
  void copy_(const T & other);

  /// Get the underlying tensor
  /// @{
  const BatchTensor & tensor() const { return _tensor; }
  BatchTensor & tensor() { return _tensor; }
  /// @}

  /// Get the tensor options
  torch::TensorOptions options() const { return _tensor.options(); }

  /// Return the number of batch dimensions
  TorchSize batch_dim() const;

  /// Return the number of base dimensions
  TorchSize base_dim() const;

  /// Return the batch size
  TorchShapeRef batch_sizes() const;

  /// Return the base size
  TorchShapeRef base_sizes() const;

  /// Get all the labeled axes
  const std::vector<const LabeledAxis *> & axes() const { return _axes; }

  /// Get a specific labeled axis
  const LabeledAxis & axis(TorchSize i = 0) const { return *_axes[i]; }

  /// How to slice the tensor given the names on each axis
  template <typename... S>
  TorchSlice slice_indices(S &&... names) const;

  /// The shape of the entire LabeledTensor
  TorchShapeRef storage_size() const;

  /// The shape of a sub-block specified by the names on each dimension
  template <typename... S>
  TorchShape storage_size(S &&... names) const;

  /// Return a labeled view into the tensor.
  /// **No reshaping is done.**
  template <typename... S>
  BatchTensor operator()(S &&... names) const;

  /// Slice the tensor on the given dimension by a single sub-axis
  Derived slice(TorchSize i, const std::string & name) const;

  /// Get the sub-block labeled by the given sub-axis names
  template <typename... S>
  Derived block(S &&... names) const;

  /// Get a batch
  Derived batch_index(TorchSlice indices) const;

  /// Set a index sliced on the batch dimensions to a value
  void batch_index_put(TorchSlice indices, const torch::Tensor & other);

  /// Return an index sliced on the batch dimensions
  BatchTensor base_index(TorchSlice indices) const;

  /// Set a index sliced on the batch dimensions to a value
  void base_index_put(TorchSlice indices, const torch::Tensor & other);

  /// Template setup for appropriate variable types
  template <typename T>
  struct variable_type
  {
    typedef T type;
  };

  /// Get and interpret the view as an object
  template <typename T, typename... S>
  typename variable_type<T>::type get(S &&... names) const
  {
    return T((*this)(names...).view(utils::add_shapes(batch_sizes(), T::const_base_sizes)),
             batch_dim());
  }

  /// Get and interpret the view as a list of objects
  template <typename T, typename... S>
  typename variable_type<T>::type get_list(S &&... names) const
  {
    return T(((*this)(names...))
                 .reshape(utils::add_shapes(this->batch_sizes(), -1, T::const_base_sizes)),
             this->batch_dim() + sizeof...(names));
  }

  /// Set and interpret the input as an object
  template <typename T, typename... S>
  void set(const BatchTensorBase<T> & value, S &&... names)
  {
    (*this)(names...).index_put_(
        {torch::indexing::None},
        value.reshape(utils::add_shapes(value.batch_sizes(), storage_size(names...))));
  }

  /// Set and interpret the input as a list of objects
  template <typename T, typename... S>
  void set_list(const BatchTensorBase<T> & value, S &&... names)
  {
    this->set(BatchTensor(value, value.batch_dim() - sizeof...(names)), names...);
  }

  /// Negation
  Derived operator-() const;

  /// Change tensor options
  Derived to(const torch::TensorOptions & options) const;

protected:
  /// The tensor
  BatchTensor _tensor;

  /// The labeled axes of this tensor
  // Urgh, I can't use const references here as the elements of a vector has to be "assignable".
  std::vector<const LabeledAxis *> _axes;

private:
  template <std::size_t... I, typename... S>
  TorchSlice slice_indices_impl(std::index_sequence<I...>, S &&... names) const;

  template <std::size_t... I, typename... S>
  TorchShape storage_size_impl(std::index_sequence<I...>, S &&... names) const;

  template <std::size_t... I, typename... S>
  Derived block_impl(std::index_sequence<I...>, S &&... names) const;
};

template <class Derived, TorchSize D>
template <typename T>
void
LabeledTensor<Derived, D>::copy_(const T & other)
{
  _tensor.copy_(other);
}

template <class Derived, TorchSize D>
template <typename... S>
TorchSlice
LabeledTensor<Derived, D>::slice_indices(S &&... names) const
{
  static_assert(sizeof...(names) == D, "Wrong labaled dimesion in LabeledTensor::slice_indices");
  return slice_indices_impl(std::make_index_sequence<sizeof...(names)>(),
                            std::forward<S>(names)...);
}

template <class Derived, TorchSize D>
template <std::size_t... I, typename... S>
TorchSlice
LabeledTensor<Derived, D>::slice_indices_impl(std::index_sequence<I...>, S &&... names) const
{
  return {_axes[I]->indices(names)...};
}

template <class Derived, TorchSize D>
template <typename... S>
TorchShape
LabeledTensor<Derived, D>::storage_size(S &&... names) const
{
  static_assert(sizeof...(names) == D, "Wrong labaled dimesion in LabeledTensor::storage_size");
  return storage_size_impl(std::make_index_sequence<D>(), std::forward<S>(names)...);
}

template <class Derived, TorchSize D>
template <std::size_t... I, typename... S>
TorchShape
LabeledTensor<Derived, D>::storage_size_impl(std::index_sequence<I...>, S &&... names) const
{
  return {_axes[I]->storage_size(names)...};
}

template <class Derived, TorchSize D>
template <typename... S>
BatchTensor
LabeledTensor<Derived, D>::operator()(S &&... names) const
{
  static_assert(sizeof...(names) == D, "Wrong labeled dimension in LabeledTensor::operator()");
  return base_index(slice_indices(names...));
}

template <class Derived, TorchSize D>
template <typename... S>
Derived
LabeledTensor<Derived, D>::block(S &&... names) const
{
  return block_impl(std::make_index_sequence<sizeof...(names)>(), std::forward<S>(names)...);
}

template <class Derived, TorchSize D>
template <std::size_t... I, typename... S>
Derived
LabeledTensor<Derived, D>::block_impl(std::index_sequence<I...>, S &&... names) const
{
  TorchSlice idx = {_axes[I]->indices(names)...};
  std::vector<const LabeledAxis *> new_axes = {&_axes[I]->subaxis(names)...};
  return Derived(_tensor.base_index(idx), new_axes);
}
} // namespace neml2
