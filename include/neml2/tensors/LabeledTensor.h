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

#include "neml2/tensors/StorageTensor.h"

namespace neml2
{
/**
 * @brief The primary data structure in NEML2 for working with labeled tensor views.
 *
 * Each LabeledTensor consists of one BatchTensor and one or more LabeledAxis. The
 * `LabeledTensor<D>` is templated on the base dimension @tparam D. LabeledTensor handles the
 * creation, modification, and accessing of labeled tensors.
 *
 * @tparam D The number of base dimensions
 */
template <TorchSize D>
class LabeledTensor : public StorageTensor<D>
{
public:
  /// View into LabeledTensor
  template <typename T>
  class View : public StorageTensor<D>::template View<T>
  {
  public:
    View() = default;

    View(StorageTensor<D> *,
         const std::array<LabeledAxisAccessor, D> &,
         SmartVector<typename StorageTensor<D>::ViewBase> &);

    /**
     * @copydoc neml2::StorageTensor::ViewBase::operator=
     *
     * Since this is a view managed by torch, we just need to reshape and copy the content into
     * _value
     */
    virtual void operator=(const BatchTensor &) override;

    virtual const BatchTensor & raw_value() const override { return _raw_value; }

    virtual const T & value() const override { return _value; }

    virtual void reinit() override;

    virtual void requires_grad_(bool req = true) override
    {
      if (req)
        throw NEMLException(
            "LabeledTensor does not support AD. This could happen if you attept to use AD in "
            "inplace assembly mode. Switching to concatenation assembly mode should fix this "
            "issue.");
    }

    virtual typename StorageTensor<D>::template View<T> & clone() override;

  private:
    /// View into the LabeledTensor without reshaping
    BatchTensor _raw_value;
    /// View into _view with the correct logical base shape of @tparam T
    T _value;
  };

  /// Default constructor
  LabeledTensor() = default;

  /// Copy constructor
  LabeledTensor(const LabeledTensor<D> & other);

  /// Construct an empty tensor given batch shape and array of axes
  LabeledTensor(TorchShapeRef batch_shape,
                const std::array<const LabeledAxis *, D> & axes,
                const torch::TensorOptions & options = default_tensor_options());

  /// Construct from a Tensor with batch dim and array of axes
  LabeledTensor(const torch::Tensor & tensor,
                TorchSize batch_dim,
                const std::array<const LabeledAxis *, D> & axes);

  /// Construct from a BatchTensor with array of axes
  LabeledTensor(const BatchTensor & tensor, const std::array<const LabeledAxis *, D> & axes);

  /// Chain rule
  static LabeledTensor<2> chain(const LabeledTensor<2> &, const LabeledTensor<2> &);

  /// Second order chain rule
  static LabeledTensor<3> chain(const LabeledTensor<3> &,
                                const LabeledTensor<3> &,
                                const LabeledTensor<2> &,
                                const LabeledTensor<2> &);

  /// Copy assignment operator
  void operator=(const LabeledTensor<D> & other);

  /// Get a view by slicing each axis
  template <typename T>
  [[nodiscard]] View<T> & view(const std::array<LabeledAxisAccessor, D> &);

  virtual const BatchTensor & view_raw(const std::array<LabeledAxisAccessor, D> &) override;

  LabeledTensor<D> clone() const;

  virtual void to_(const torch::TensorOptions &) override;

  virtual void copy_(const BatchTensor &) override;

  virtual void zero_() override;

  virtual BatchTensor assemble() const override;

  /// Convert to BatchTensor (does not take ownership)
  virtual BatchTensor tensor() const;

  virtual BatchTensor get(const std::array<LabeledAxisAccessor, D> & names) const override;

  virtual void set_(const std::array<LabeledAxisAccessor, D> &, const BatchTensor &) override;

  virtual void collect_(const StorageTensor<D> & other,
                        const LabeledAxisAccessor & i1 = LabeledAxisAccessor(),
                        const LabeledAxisAccessor & i2 = LabeledAxisAccessor()) override;

  /// Calculate slicing indices given the names on each axis
  TorchSlice slice_indices(const std::array<LabeledAxisAccessor, D> & names) const;

  /// Calculate storage shape given the names on each axis
  TorchShape storage_sizes(const std::array<LabeledAxisAccessor, D> & names) const;

protected:
  /// The tensor
  BatchTensor _tensor;
};

using LabeledVector = LabeledTensor<1>;
using LabeledMatrix = LabeledTensor<2>;
using LabeledTensor3D = LabeledTensor<3>;
} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

namespace neml2
{
template <TorchSize D>
template <typename T>
LabeledTensor<D>::View<T>::View(StorageTensor<D> * storage,
                                const std::array<LabeledAxisAccessor, D> & indices,
                                SmartVector<typename StorageTensor<D>::ViewBase> & peers)
  : StorageTensor<D>::template View<T>(storage, indices, peers)
{
  reinit();
}

template <TorchSize D>
template <typename T>
void
LabeledTensor<D>::View<T>::reinit()
{
  auto & storage = this->template storage<LabeledTensor<D>>();

  _raw_value = storage.tensor().base_index(storage.slice_indices(this->indices()));

  if constexpr (std::is_same_v<T, BatchTensor>)
    _value = _raw_value;
  else
  {
    auto shape = utils::add_shapes(_raw_value.batch_sizes(), T::const_base_sizes);
    _value = T(_raw_value.view(shape));
  }
}

template <TorchSize D>
template <typename T>
void
LabeledTensor<D>::View<T>::operator=(const BatchTensor & val)
{
  _value.index_put_({torch::indexing::Slice()},
                    val.batch_expand(_value.batch_sizes()).base_reshape(_value.base_sizes()));
}

template <TorchSize D>
template <typename T>
typename LabeledTensor<D>::template View<T> &
LabeledTensor<D>::view(const std::array<LabeledAxisAccessor, D> & i)
{
  auto new_view = std::make_unique<View<T>>(this, i, this->_views[i]);
  auto new_view_base = this->_views[i].set_pointer(std::move(new_view));
  auto new_view_ptr = dynamic_cast<View<T> *>(new_view_base);
  neml_assert(new_view_ptr, "Failed to cast view to concrete type");
  return *new_view_ptr;
}

template <TorchSize D>
template <typename T>
typename StorageTensor<D>::template View<T> &
LabeledTensor<D>::View<T>::clone()
{
  auto & storage = this->template storage<LabeledTensor<D>>();
  return storage.template view<T>(this->indices());
}
} // namespace neml2
