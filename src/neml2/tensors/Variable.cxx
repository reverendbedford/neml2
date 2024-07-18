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

#include "neml2/tensors/Variable.h"

namespace neml2
{
void
VariableBase::cache(TensorShapeRef batch_shape)
{
  _batch_sizes = batch_shape.vec();
}

void
VariableBase::setup_views(const LabeledVector * value,
                          const LabeledMatrix * deriv,
                          const LabeledTensor3D * secderiv)
{
  if (value)
  {
    _value_storage = value;
    _raw_value = _value_storage->base_index(name());
  }

  if (deriv)
  {
    _derivative_storage = deriv;
    for (auto arg : _derivative_storage->axis(1).variable_names())
      _dvalue_d[arg] = _derivative_storage->base_index({name(), arg});
  }

  if (secderiv)
  {
    _second_derivative_storage = secderiv;
    for (auto arg1 : _second_derivative_storage->axis(1).variable_names())
      for (auto arg2 : _second_derivative_storage->axis(2).variable_names())
        _d2value_d[arg1][arg2] = _second_derivative_storage->base_index({name(), arg1, arg2});
  }
}

const LabeledVector &
VariableBase::value_storage() const
{
  neml_assert_dbg(_value_storage, "Variable value storage not initialized.");
  return *_value_storage;
}

const LabeledMatrix &
VariableBase::derivative_storage() const
{
  neml_assert_dbg(_derivative_storage, "Variable derivative storage not initialized.");
  return *_derivative_storage;
}

const LabeledTensor3D &
VariableBase::second_derivative_storage() const
{
  neml_assert_dbg(_second_derivative_storage, "Variable 2nd derivative storage not initialized.");
  return *_second_derivative_storage;
}

Derivative
VariableBase::d(const VariableBase & x)
{
  neml_assert_dbg(_dvalue_d.count(x.name()),
                  "Error retrieving first derivative: ",
                  name(),
                  " does not depend on ",
                  x.name());

  return Derivative(_dvalue_d[x.name()]);
}

Derivative
VariableBase::d(const VariableBase & x1, const VariableBase & x2)
{
  neml_assert_dbg(_d2value_d.count(x1.name()),
                  "Error retrieving second derivative: ",
                  name(),
                  " does not depend on ",
                  x1.name());
  neml_assert_dbg(_d2value_d[x1.name()].count(x2.name()),
                  "Error retrieving second derivative: d(",
                  name(),
                  ")/d(",
                  x1.name(),
                  ") does not depend on ",
                  x2.name());

  return Derivative(_d2value_d[x1.name()][x2.name()]);
}

void
Derivative::operator=(const Tensor & val)
{
  _value.index_put_({torch::indexing::Slice()},
                    val.batch_expand_as(_value).base_reshape(_value.base_sizes()));
}
}
