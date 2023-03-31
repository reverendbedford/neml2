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
template <TorchSize N, TorchSize D>
class LabeledTensor
{
public:
  /// Empty constructor
  LabeledTensor(const torch::TensorOptions & options = default_tensor_options);

  /// Construct from a `BatchTensor` with vector of `LabeledAxis`
  /// This constructor is useful when the size of the LabeledAxis vector is unknown at compile time.
  LabeledTensor(const BatchTensor<N> & tensor, const std::vector<const LabeledAxis *> & axes);

  /// Setup new storage with zeros
  LabeledTensor(TorchShapeRef batch_size,
                const std::vector<const LabeledAxis *> & axes,
                const torch::TensorOptions & options = default_tensor_options);

  /// Copy constructor
  LabeledTensor(const LabeledTensor & other);

  /// A potentially dangerous implicit conversion
  // Should we mark it explicit?
  operator torch::Tensor() const { return _tensor; }

  /// Return the configuration of the underlying tensor
  torch::TensorOptions options() const { return _tensor.options(); }

  [[nodiscard]] LabeledTensor<N, D> to(const torch::TensorOptions & options) const;

  /// Assignment operator
  LabeledTensor<N, D> & operator=(const LabeledTensor<N, D> & other);

  /// Clone a this LabeledTensor
  LabeledTensor<N, D> clone() const { return LabeledTensor<N, D>(_tensor.clone(), _axes); }

  /// (Partially) deep copy another LabeledTensor, as we only copy the tensor,
  /// and we never ever deep copy the LabeledAxis.
  void copy(const LabeledTensor<N, D> & other);

  /// Get the underlying tensor
  /// @{
  const BatchTensor<N> & tensor() const { return _tensor; }
  BatchTensor<N> & tensor() { return _tensor; }
  /// @}

  /// Get all the labeled axes
  const std::vector<const LabeledAxis *> & axes() const { return _axes; }

  /// Get a specific labeled axis
  const LabeledAxis & axis(TorchSize i) const { return *_axes[i]; }

  /// How to slice the tensor given the names on each axis
  template <typename... S>
  TorchSlice slice_indices(S &&... names) const;

  /// The shape of the entire LabeledTensor
  TorchShape storage_size() const;

  /// The shape of a sub-block specified by the names on each dimension
  template <typename... S>
  TorchShape storage_size(S &&... names) const;

  /// Return a labeled view into the tensor.
  /// **No reshaping is done.**
  template <typename... S>
  BatchTensor<N> operator()(S &&... names) const;

  /// Slice the tensor on the given dimension by a single sub-axis
  LabeledTensor<N, D> slice(TorchSize i, const std::string & name) const;

  /// Get the sub-block labeled by the given sub-axis names
  template <typename... S>
  LabeledTensor<N, D> block(S &&... names) const;

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
    return T((*this)(names...).view(utils::add_shapes(_tensor.batch_sizes(), T::_base_sizes)));
  }

  /// Set and interpret the input as an object
  template <typename... S>
  void set(const BatchTensor<N> & value, S &&... names)
  {
    (*this)(names...).index_put_(
        {torch::indexing::None},
        value.reshape(utils::add_shapes(value.batch_sizes(), storage_size(names...))));
  }

  /// Unary minus = additive inverse
  LabeledTensor<N, D> operator-() const;

protected:
  /// The underlying raw tensor (without the labels)
  BatchTensor<N> _tensor;

  /// The labeled axes of this tensor
  // Urgh, I can't use const references here as the elements of a vector has to be "assignable".
  std::vector<const LabeledAxis *> _axes;

private:
  template <std::size_t... I, typename... S>
  TorchSlice slice_indices_impl(std::index_sequence<I...>, S &&... names) const;

  template <std::size_t... I, typename... S>
  TorchShape storage_size_impl(std::index_sequence<I...>, S &&... names) const;

  template <std::size_t... I, typename... S>
  LabeledTensor<N, D> block_impl(std::index_sequence<I...>, S &&... names) const;
};

template <TorchSize N, TorchSize D>
LabeledTensor<N, D>::LabeledTensor(const torch::TensorOptions & options)
  : _tensor(options)
{
}

template <TorchSize N, TorchSize D>
LabeledTensor<N, D>::LabeledTensor(const BatchTensor<N> & tensor,
                                   const std::vector<const LabeledAxis *> & axes)
  : _tensor(tensor),
    _axes(axes)
{
  neml_assert_dbg(axes.size() == D, "Wrong labeled dimension");

  // Check that the size of the tensor was compatible
  neml_assert_dbg(tensor.sizes() == utils::add_shapes(tensor.batch_sizes(), storage_size()),
                  "LabeledTensor does not have the right size");
}

template <TorchSize N, TorchSize D>
LabeledTensor<N, D>::LabeledTensor(TorchShapeRef batch_size,
                                   const std::vector<const LabeledAxis *> & axes,
                                   const torch::TensorOptions & options)
  : _axes(axes)
{
  _tensor = torch::zeros(utils::add_shapes(batch_size, storage_size()), options);
}

template <TorchSize N, TorchSize D>
LabeledTensor<N, D>::LabeledTensor(const LabeledTensor<N, D> & other)
  : _tensor(other._tensor)
{
  _axes = other._axes;
}

template <TorchSize N, TorchSize D>
LabeledTensor<N, D>
LabeledTensor<N, D>::to(const torch::TensorOptions & options) const
{
  auto res = clone();
  res._tensor = res._tensor.to(options);
  return res;
}

template <TorchSize N, TorchSize D>
LabeledTensor<N, D> &
LabeledTensor<N, D>::operator=(const LabeledTensor<N, D> & other)
{
  _tensor = other._tensor;
  _axes = other._axes;
  return *this;
}

template <TorchSize N, TorchSize D>
void
LabeledTensor<N, D>::copy(const LabeledTensor<N, D> & other)
{
  _tensor.copy_(other._tensor);
}

template <TorchSize N, TorchSize D>
template <typename... S>
TorchSlice
LabeledTensor<N, D>::slice_indices(S &&... names) const
{
  static_assert(sizeof...(names) == D, "Wrong labaled dimesion in LabeledTensor::slice_indices");
  return slice_indices_impl(std::make_index_sequence<sizeof...(names)>(),
                            std::forward<S>(names)...);
}

template <TorchSize N, TorchSize D>
template <std::size_t... I, typename... S>
TorchSlice
LabeledTensor<N, D>::slice_indices_impl(std::index_sequence<I...>, S &&... names) const
{
  return {_axes[I]->indices(names)...};
}

template <TorchSize N, TorchSize D>
TorchShape
LabeledTensor<N, D>::storage_size() const
{
  TorchShape s;
  s.reserve(_axes.size());
  std::transform(_axes.begin(),
                 _axes.end(),
                 std::back_inserter(s),
                 [](const LabeledAxis * axis) { return axis->storage_size(); });
  return s;
}

template <TorchSize N, TorchSize D>
template <typename... S>
TorchShape
LabeledTensor<N, D>::storage_size(S &&... names) const
{
  static_assert(sizeof...(names) == D, "Wrong labaled dimesion in LabeledTensor::storage_size");
  return storage_size_impl(std::make_index_sequence<sizeof...(names)>(), std::forward<S>(names)...);
}

template <TorchSize N, TorchSize D>
template <std::size_t... I, typename... S>
TorchShape
LabeledTensor<N, D>::storage_size_impl(std::index_sequence<I...>, S &&... names) const
{
  return {_axes[I]->storage_size(names)...};
}

template <TorchSize N, TorchSize D>
template <typename... S>
BatchTensor<N>
LabeledTensor<N, D>::operator()(S &&... names) const
{
  static_assert(sizeof...(names) == D, "Wrong labeled dimension in LabeledTensor::operator()");

  return _tensor.base_index(slice_indices(names...));
}

template <TorchSize N, TorchSize D>
LabeledTensor<N, D>
LabeledTensor<N, D>::slice(TorchSize i, const std::string & name) const
{
  TorchSlice idx(_tensor.base_dim(), torch::indexing::Slice());
  idx[i] = _axes[i]->indices(name);

  auto new_axes = _axes;
  new_axes[i] = &_axes[i]->subaxis(name);

  return LabeledTensor<N, D>(_tensor.base_index(idx), new_axes);
}

template <TorchSize N, TorchSize D>
template <typename... S>
LabeledTensor<N, D>
LabeledTensor<N, D>::block(S &&... names) const
{
  return block_impl(std::make_index_sequence<sizeof...(names)>(), std::forward<S>(names)...);
}

template <TorchSize N, TorchSize D>
template <std::size_t... I, typename... S>
LabeledTensor<N, D>
LabeledTensor<N, D>::block_impl(std::index_sequence<I...>, S &&... names) const
{
  TorchSlice idx = {_axes[I]->indices(names)...};
  std::vector<const LabeledAxis *> new_axes = {&_axes[I]->subaxis(names)...};
  return LabeledTensor<N, D>(_tensor.base_index(idx), new_axes);
}

template <TorchSize N, TorchSize D>
LabeledTensor<N, D>
LabeledTensor<N, D>::operator-() const
{
  return LabeledTensor<N, D>(-_tensor.clone(), _axes);
}
} // namespace neml2
