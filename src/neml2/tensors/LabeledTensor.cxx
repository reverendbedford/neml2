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

namespace neml2
{
template <TorchSize D>
LabeledTensor<D>::LabeledTensor(const LabeledTensor<D> & other)
  : StorageTensor<D>(other.axes()),
    _tensor(other.tensor())
{
}

template <TorchSize D>
LabeledTensor<D>::LabeledTensor(TorchShapeRef batch_size,
                                const std::array<const LabeledAxis *, D> & axes,
                                const torch::TensorOptions & options)
  : StorageTensor<D>(axes)
{
  TorchShape s;
  s.reserve(axes.size());
  std::transform(axes.begin(),
                 axes.end(),
                 std::back_inserter(s),
                 [](const LabeledAxis * axis) { return axis->storage_size(); });
  _tensor = BatchTensor::zeros(batch_size, s, options);
}

template <TorchSize D>
LabeledTensor<D>::LabeledTensor(const torch::Tensor & tensor,
                                TorchSize batch_dim,
                                const std::array<const LabeledAxis *, D> & axes)
  : StorageTensor<D>(axes),
    _tensor(tensor, batch_dim)
{
  // Check that the size of the tensor was compatible
  for (TorchSize i = 0; i < D; i++)
    neml_assert_dbg(_tensor.base_size(i) == this->axis(i).storage_size(),
                    "LabeledTensor does not have the right size");
}

template <TorchSize D>
LabeledTensor<D>::LabeledTensor(const BatchTensor & tensor,
                                const std::array<const LabeledAxis *, D> & axes)
  : StorageTensor<D>(axes),
    _tensor(tensor)
{
  // Check that the size of the tensor was compatible
  for (TorchSize i = 0; i < D; i++)
    neml_assert_dbg(_tensor.base_size(i) == this->axis(i).storage_size(),
                    "LabeledTensor does not have the right size");
}

template <TorchSize D>
void
LabeledTensor<D>::operator=(const LabeledTensor<D> & other)
{
  _tensor = other.tensor();
  this->_axes = other.axes();
}

template <TorchSize D>
LabeledTensor<D>
LabeledTensor<D>::clone() const
{
  return LabeledTensor<D>(_tensor.clone(), this->_axes);
}

template <TorchSize D>
void
LabeledTensor<D>::to_(const torch::TensorOptions & options)
{
  _tensor = _tensor.to(options);
  this->reinit_views();
}

template <TorchSize D>
void
LabeledTensor<D>::copy_(const BatchTensor & val)
{
  _tensor.copy_(val);
}

template <TorchSize D>
void
LabeledTensor<D>::zero_()
{
  _tensor.zero_();
}

template <TorchSize D>
BatchTensor
LabeledTensor<D>::assemble() const
{
  return _tensor.clone();
}

template <TorchSize D>
BatchTensor
LabeledTensor<D>::tensor() const
{
  return _tensor;
}

template <TorchSize D>
BatchTensor
LabeledTensor<D>::get(const std::array<LabeledAxisAccessor, D> & names) const
{
  return _tensor.base_index(slice_indices(names)).clone();
}

template <TorchSize D>
void
LabeledTensor<D>::set_(const std::array<LabeledAxisAccessor, D> & names, const BatchTensor & val)
{
  _tensor.base_index_put(slice_indices(names), val.base_reshape(this->storage_sizes(names)));
}

template <TorchSize D>
void
LabeledTensor<D>::collect_(const LabeledTensor<D> & other,
                           const LabeledAxisAccessor & i1,
                           const LabeledAxisAccessor & i2)
{
  if constexpr (D > 1)
    neml_assert_dbg(this->axis(1) == other.axis(1),
                    "Can only collect from tensor with conformal y axes");
  if constexpr (D > 2)
    neml_assert_dbg(this->axis(2) == other.axis(2),
                    "Can only collect from tensor with conformal z axes");
  const auto indices =
      this->axis(0).subaxis(i1).common_indices(other.axis(0).subaxis(i2), /*recursive=*/true);
  for (const auto & [idxi, idxi_other] : indices)
    _tensor.base_index_put({idxi}, other.tensor().base_index({idxi_other}));
}

template <TorchSize D>
TorchSlice
LabeledTensor<D>::slice_indices(const std::array<LabeledAxisAccessor, D> & names) const
{
  TorchSlice indices;
  for (TorchSize i = 0; i < D; i++)
    indices.push_back(this->axis(i).indices(names[i]));
  return indices;
}

template <TorchSize D>
TorchShape
LabeledTensor<D>::storage_sizes(const std::array<LabeledAxisAccessor, D> & names) const
{
  TorchShape sizes(D);
  for (TorchSize i = 0; i < D; i++)
    sizes[i] = this->axis(i).storage_size(names[i]);
  return sizes;
}

template class LabeledTensor<1>;
template class LabeledTensor<2>;
template class LabeledTensor<3>;
}
