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
#include "neml2/tensors/Tensor.h"

namespace neml2
{
/**
 * @brief The primary data structure in NEML2 for working with labeled tensor views.
 *
 * Each LabeledTensor consists of one Tensor and one or more LabeledAxis. The
 * `LabeledTensor<D>` is templated on the base dimension \f$D\f$. LabeledTensor handles the
 * creation, modification, and accessing of labeled tensors.
 *
 * @tparam D The number of base dimensions
 */
template <class Derived, Size D>
class LabeledTensor
{
public:
  /// Default constructor
  LabeledTensor() = default;

  /// Construct from a torch::Tensor and array of `LabeledAxis`
  LabeledTensor(const torch::Tensor & tensor, const std::array<const LabeledAxis *, D> & axes);

  /// Construct from a Tensor with array of `LabeledAxis`
  LabeledTensor(const Tensor & tensor, const std::array<const LabeledAxis *, D> & axes);

  /// Copy constructor
  LabeledTensor(const Derived & other);

  /// Assignment operator
  void operator=(const Derived & other);

  /// A potentially dangerous implicit conversion
  // Should we mark it explicit?
  operator Tensor() const;

  /// A potentially dangerous implicit conversion
  // Should we mark it explicit?
  operator torch::Tensor() const;

  /// Setup new empty storage
  [[nodiscard]] static Derived
  empty(TensorShapeRef batch_shape,
        const std::array<const LabeledAxis *, D> & axes,
        const torch::TensorOptions & options = default_tensor_options());

  /// Setup new empty storage like another LabeledTensor
  [[nodiscard]] static Derived empty_like(const Derived & other);

  /// Setup new storage with zeros
  [[nodiscard]] static Derived
  zeros(TensorShapeRef batch_shape,
        const std::array<const LabeledAxis *, D> & axes,
        const torch::TensorOptions & options = default_tensor_options());

  /// Setup new storage with zeros like another LabeledTensor
  [[nodiscard]] static Derived zeros_like(const Derived & other);

  /// Clone this LabeledTensor
  Derived clone(torch::MemoryFormat memory_format = torch::MemoryFormat::Contiguous) const;

  /// Copy the value from another tensor
  template <typename T>
  void copy_(const T & other);

  /// Return a copy without gradient graphs
  Derived detach() const;

  /// Detach from gradient graphs
  void detach_();

  /// Zero out this tensor
  void zero_();

  /// Get the underlying tensor
  ///@{
  const Tensor & tensor() const { return _tensor; }
  Tensor & tensor() { return _tensor; }
  /// @}

  /// Get the tensor options
  torch::TensorOptions options() const { return _tensor.options(); }

  /// Return the number of batch dimensions
  Size batch_dim() const;

  /// Return the number of base dimensions
  Size base_dim() const;

  /// Return the batch size
  TensorShapeRef batch_sizes() const;

  /// Return the base size
  TensorShapeRef base_sizes() const;

  /// Get all the labeled axes
  const std::array<const LabeledAxis *, D> & axes() const { return _axes; }

  /// Get a specific labeled axis
  const LabeledAxis & axis(Size i = 0) const { return *_axes[i]; }

  /// How to slice the tensor given the names on each axis
  template <typename... S>
  indexing::TensorIndices slice_indices(S &&... names) const;

  /// The shape of the entire LabeledTensor
  TensorShapeRef storage_size() const;

  /// The shape of a sub-block specified by the names on each dimension
  template <typename... S>
  TensorShape storage_size(S &&... names) const;

  /// Return a labeled view into the tensor.
  /// **No reshaping is done.**
  template <typename... S>
  Tensor operator()(S &&... names) const;

  /// Slice the tensor on the given dimension by a single variable or sub-axis
  Derived slice(Size i, const std::string & name) const;

  /// Get the sub-block labeled by the given sub-axis names
  template <typename... S>
  Derived block(S &&... names) const;

  /// Get a batch
  Derived batch_index(indexing::TensorIndices indices) const;

  /// Set a index sliced on the batch dimensions to a value
  void batch_index_put(indexing::TensorIndices indices, const torch::Tensor & other);

  /// Return an index sliced on the batch dimensions
  Tensor base_index(indexing::TensorIndices indices) const;

  /// Set a index sliced on the batch dimensions to a value
  void base_index_put(indexing::TensorIndices indices, const torch::Tensor & other);

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
  void set(const TensorBase<T> & value, S &&... names)
  {
    (*this)(names...).index_put_(
        {torch::indexing::None},
        value.reshape(utils::add_shapes(value.batch_sizes(), storage_size(names...))));
  }

  /// Set and interpret the input as a list of objects
  template <typename T, typename... S>
  void set_list(const TensorBase<T> & value, S &&... names)
  {
    this->set(Tensor(value, value.batch_dim() - sizeof...(names)), names...);
  }

  /// Negation
  Derived operator-() const;

  /// Change tensor options
  Derived to(const torch::TensorOptions & options) const;

protected:
  /// The tensor
  Tensor _tensor;

  /// The labeled axes of this tensor
  std::array<const LabeledAxis *, D> _axes;

private:
  template <std::size_t... I, typename... S>
  indexing::TensorIndices slice_indices_impl(std::index_sequence<I...>, S &&... names) const;

  template <std::size_t... I, typename... S>
  TensorShape storage_size_impl(std::index_sequence<I...>, S &&... names) const;

  template <std::size_t... I, typename... S>
  Derived block_impl(std::index_sequence<I...>, S &&... names) const;
};

template <class Derived, Size D>
template <typename T>
void
LabeledTensor<Derived, D>::copy_(const T & other)
{
  _tensor.copy_(other);
}

template <class Derived, Size D>
template <typename... S>
indexing::TensorIndices
LabeledTensor<Derived, D>::slice_indices(S &&... names) const
{
  return slice_indices_impl(std::make_index_sequence<sizeof...(names)>(),
                            std::forward<S>(names)...);
}

template <class Derived, Size D>
template <std::size_t... I, typename... S>
indexing::TensorIndices
LabeledTensor<Derived, D>::slice_indices_impl(std::index_sequence<I...>, S &&... names) const
{
  return {_axes[I]->indices(names)...};
}

template <class Derived, Size D>
template <typename... S>
TensorShape
LabeledTensor<Derived, D>::storage_size(S &&... names) const
{
  return storage_size_impl(std::make_index_sequence<D>(), std::forward<S>(names)...);
}

template <class Derived, Size D>
template <std::size_t... I, typename... S>
TensorShape
LabeledTensor<Derived, D>::storage_size_impl(std::index_sequence<I...>, S &&... names) const
{
  return {_axes[I]->storage_size(names)...};
}

template <class Derived, Size D>
template <typename... S>
Tensor
LabeledTensor<Derived, D>::operator()(S &&... names) const
{
  return base_index(slice_indices(names...));
}

template <class Derived, Size D>
template <typename... S>
Derived
LabeledTensor<Derived, D>::block(S &&... names) const
{
  return block_impl(std::make_index_sequence<sizeof...(names)>(), std::forward<S>(names)...);
}

template <class Derived, Size D>
template <std::size_t... I, typename... S>
Derived
LabeledTensor<Derived, D>::block_impl(std::index_sequence<I...>, S &&... names) const
{
  indexing::TensorIndices idx = {_axes[I]->indices(names)...};
  std::array<const LabeledAxis *, D> new_axes = {&_axes[I]->subaxis(names)...};
  return Derived(_tensor.base_index(idx), new_axes);
}
} // namespace neml2
