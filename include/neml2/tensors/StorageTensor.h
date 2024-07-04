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
#include "neml2/base/SmartVector.h"
#include "neml2/tensors/LabeledAxis.h"
#include "neml2/tensors/BatchTensor.h"

namespace neml2
{
/// Abstract base class for tensor storage
template <TorchSize D>
class StorageTensor
{
public:
  /// Abstract base class for view
  class ViewBase
  {
  public:
    ViewBase(SmartVector<ViewBase> & peers)
      : _peers(peers)
    {
    }

    virtual ~ViewBase() = default;

    /// Get the flattened raw value without reshaping
    virtual BatchTensor & raw_value() = 0;

    /// Some operations may "break" the view --

    /**
     * @brief Effectively recreate the view
     *
     * This method is needed because some operations may "break" the view.
     * It is the derived class's responsibility to store enough information so that a view can be
     * recreated upon request.
     */
    virtual void reinit() = 0;

  protected:
    /// Peer views that view into the same location
    SmartVector<ViewBase> & _peers;
  };

  /// Intermediate base class for view (with additional information on tensor type)
  template <typename T>
  class View : public ViewBase
  {
  public:
    using ViewBase::ViewBase;

    /// Assignment operator
    virtual void operator=(const BatchTensor &) = 0;

    /// Get the value with the correct base shape
    virtual T & value() = 0;

    /// Make this a leaf variable in backward AD
    virtual void requires_grad_(bool req = true) = 0;

    /// Make a clone of this view
    virtual View<T> & clone() = 0;
  };

  StorageTensor() = default;

  StorageTensor(const std::array<const LabeledAxis *, D> & axes)
    : _axes(axes)
  {
  }

  virtual ~StorageTensor() = default;

  /// Implicit conversion to BatchTensor
  operator BatchTensor() const { return assemble(); }

  /// Send to another device or dtype
  virtual void to_(const torch::TensorOptions &) = 0;

  /// Copy values from another BatchTensor
  virtual void copy_(const BatchTensor &) = 0;

  /// Zero out the storage tensor
  virtual void zero_() = 0;

  /// Assemble the tensor to a BatchTensor (takes ownership)
  virtual BatchTensor assemble() const = 0;

  /// Get all the labeled axes
  const std::array<const LabeledAxis *, D> & axes() const { return _axes; }

  /// Get a specific labeled axis
  const LabeledAxis & axis(TorchSize i = 0) const { return *_axes[i]; }

  /// Get the raw tensor value given indices (takes ownership)
  virtual BatchTensor get(const std::array<LabeledAxisAccessor, D> &) const = 0;

  /// Set the raw tensor value given indices
  virtual void set_(const std::array<LabeledAxisAccessor, D> &, const BatchTensor &) = 0;

  /// Get the tensor and reinterpret as the given tensor type (takes ownership)
  template <typename T>
  T reinterpret(const std::array<LabeledAxisAccessor, D> &) const;

  /// Reinitialize all views
  void reinit_views();

  /// Get all views
  const std::map<std::array<LabeledAxisAccessor, D>, SmartVector<ViewBase>> & views() const;

  /// Get all views into the given indices
  const SmartVector<ViewBase> & views(const std::array<LabeledAxisAccessor, D> & i) const;

protected:
  /// The labeled axes of this tensor
  std::array<const LabeledAxis *, D> _axes;

  /// Storage of all views
  std::map<std::array<LabeledAxisAccessor, D>, SmartVector<ViewBase>> _views;
};
} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementations
///////////////////////////////////////////////////////////////////////////////

namespace neml2
{
template <TorchSize D>
template <typename T>
T
StorageTensor<D>::reinterpret(const std::array<LabeledAxisAccessor, D> & i) const
{
  return get(i).base_reshape(T::const_base_sizes);
}

template <TorchSize D>
void
StorageTensor<D>::reinit_views()
{
  for (auto && [i, views] : _views)
    for (auto & view : views)
      view.reinit();
}

template <TorchSize D>
const std::map<std::array<LabeledAxisAccessor, D>,
               SmartVector<typename StorageTensor<D>::ViewBase>> &
StorageTensor<D>::views() const
{
  return _views;
}

template <TorchSize D>
const SmartVector<typename StorageTensor<D>::ViewBase> &
StorageTensor<D>::views(const std::array<LabeledAxisAccessor, D> & i) const
{
  neml_assert_dbg(_views.count(i), "No view associated with the given indices");
  return _views.at(i);
}
} // namespace neml2
