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
    ViewBase() = default;

    /// Get the flattened raw value without reshaping
    BatchTensor & raw_value() = 0;

  protected:
    /// Peer views that view into the same location
    SmartVector<ViewBase> & _peers;
  };

  /// Intermediate base class for view (with additional information on tensor type)
  template <typename T>
  class View : public ViewBase
  {
  public:
    View() = default;

    /// Get the value with the correct base shape
    T & value() = 0;
  };

  StorageTensor() = default;

  /// Copy assignment operator
  virtual void operator=(const StorageTensor<D> &) = 0;

  /// Get a view by slicing each axis
  template <typename T>
  View<T> & view(const std::array<VariableName, D> &);

  /// Copy values from another BatchTensor
  virtual void copy_(const BatchTensor &) = 0;

  /// Zero out the storage tensor
  virtual void zero_() = 0;

  /// Get a slice of the tensor.
  virtual BatchTensor get(const std::array<VariableName, D> & names) const;

  /// Tensor value modifier
  virtual void set_(const std::array<VariableName, D> &, const BatchTensor &) = 0;

  /// Assemble the tensor to a BatchTensor (takes ownership)
  virtual BatchTensor assemble() const = 0;

  /// Get all the labeled axes
  const std::array<const LabeledAxis *, D> & axes() const { return _axes; }

  /// Get a specific labeled axis
  const LabeledAxis & axis(TorchSize i = 0) const { return *_axes[i]; }

protected:
  /// It is the derived class's responsibility to make the view
  virtual std::unique_ptr<ViewBase> make_view() const = 0;

  /// The labeled axes of this tensor
  std::array<const LabeledAxis *, D> _axes;

  /// Storage of all views
  std::map<std::array<VariableName, D>, SmartVector<ViewBase>> _views;
};
} // namespace neml2
