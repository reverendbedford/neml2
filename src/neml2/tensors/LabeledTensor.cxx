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

#include "neml2/tensors/LabeledTensor.h"

#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledTensor3D.h"

namespace neml2
{
template <class Derived, TorchSize D>
LabeledTensor<Derived, D>::LabeledTensor(const torch::Tensor & tensor,
                                         TorchSize batch_dim,
                                         const std::array<const LabeledAxis *, D> & axes)
  : _tensor(tensor, batch_dim),
    _axes(axes)
{
  // Check that the size of the tensor was compatible
  neml_assert_dbg(base_sizes() == storage_size(), "LabeledTensor does not have the right size");
}

template <class Derived, TorchSize D>
LabeledTensor<Derived, D>::LabeledTensor(const BatchTensor & tensor,
                                         const std::array<const LabeledAxis *, D> & axes)
  : _tensor(tensor),
    _axes(axes)
{
  // Check that the size of the tensor was compatible
  neml_assert_dbg(base_sizes() == storage_size(), "LabeledTensor does not have the right size");
}

template <class Derived, TorchSize D>
LabeledTensor<Derived, D>::LabeledTensor(const Derived & other)
  : _tensor(other),
    _axes(other.axes())
{
}

template <class Derived, TorchSize D>
Derived
LabeledTensor<Derived, D>::empty(TorchShapeRef batch_size,
                                 const std::array<const LabeledAxis *, D> & axes,
                                 const torch::TensorOptions & options)
{
  TorchShape s;
  s.reserve(axes.size());
  std::transform(axes.begin(),
                 axes.end(),
                 std::back_inserter(s),
                 [](const LabeledAxis * axis) { return axis->storage_size(); });
  return Derived(BatchTensor::empty(batch_size, s, options), axes);
}

template <class Derived, TorchSize D>
Derived
LabeledTensor<Derived, D>::zeros(TorchShapeRef batch_size,
                                 const std::array<const LabeledAxis *, D> & axes,
                                 const torch::TensorOptions & options)
{
  TorchShape s;
  s.reserve(axes.size());
  std::transform(axes.begin(),
                 axes.end(),
                 std::back_inserter(s),
                 [](const LabeledAxis * axis) { return axis->storage_size(); });
  return Derived(BatchTensor::zeros(batch_size, s, options), axes);
}

template <class Derived, TorchSize D>
void
LabeledTensor<Derived, D>::operator=(const StorageTensor<D> & other)
{
  _tensor = other.tensor();
  _axes = other.axes();
}

template <class Derived, TorchSize D>
void
LabeledTensor<Derived, D>::copy_(const BatchTensor & val)
{
  _tensor.copy_(val);
}

template <class Derived, TorchSize D>
void
LabeledTensor<Derived, D>::zero_()
{
  _tensor.zero_();
}

template <class Derived, TorchSize D>
BatchTensor
LabeledTensor<Derived, D>::get(const std::array<VariableName, D> & names) const
{
  return _tensor.base_index(slice_indices(names));
}

template <class Derived, TorchSize D>
void
LabeledTensor<Derived, D>::set_(const std::array<VariableName, D> & names, const BatchTensor & val)
{
  _tensor.base_index_put(slice_indices(names), value.base_reshape(storage_size(names)));
}

template <class Derived, TorchSize D>
BatchTensor
LabeledTensor<Derived, D>::assemble() const
{
  return _tensor.clone();
}

template <class Derived, TorchSize D>
BatchTensor
LabeledTensor<Derived, D>::tensor() const
{
  return _tensor;
}

template <class Derived, TorchSize D>
TorchSlice
LabeledTensor<Derived, D>::slice_indices(const std::array<VariableName, D> & names) const
{
  TorchSlice indices(D);
  for (TorchSize i = 0; i < D; i++)
    indices[i] = axis(i)->indices(names[i]);
  return indices;
}

template <class Derived, TorchSize D>
TorchSlice
LabeledTensor<Derived, D>::storage_sizes(const std::array<VariableName, D> & names) const
{
  TorchShape sizes(D);
  for (TorchSize i = 0; i < D; i++)
    sizes[i] = axis(i)->storage_size(names[i]);
  return sizes;
}

template class LabeledTensor<LabeledVector, 1>;
template class LabeledTensor<LabeledMatrix, 2>;
template class LabeledTensor<LabeledTensor3D, 3>;
}
