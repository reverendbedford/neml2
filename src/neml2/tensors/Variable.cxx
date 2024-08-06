// Copyright 2024, UChicago Argonne, LLC
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
#include "neml2/models/Model.h"

namespace neml2
{
VariableBase::VariableBase(const VariableName & name_in, const Model * owner)
  : _name(name_in),
    _owner(owner),
    _src(nullptr),
    _is_state(_name.start_with("state")),
    _is_old_state(_name.start_with("old_state")),
    _is_force(_name.start_with("forces")),
    _is_old_force(_name.start_with("old_forces")),
    _is_residual(_name.start_with("residual")),
    _is_parameter(_name.start_with("parameters")),
    _is_other(!_is_state && !_is_old_state && !_is_force && !_is_old_force && !_is_residual &&
              !_is_parameter),
    _is_solve_dependent(_is_state || _is_residual || _is_parameter)
{
}

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
    _raw_value = value->base_index(name());

  if (deriv)
    for (auto arg : deriv->axis(1).variable_names())
      _dvalue_d[arg] = deriv->base_index({name(), arg});

  if (secderiv)
    for (auto arg1 : secderiv->axis(1).variable_names())
      for (auto arg2 : secderiv->axis(2).variable_names())
        _d2value_d[arg1][arg2] = secderiv->base_index({name(), arg1, arg2});
}

void
VariableBase::setup_views(const VariableBase * other)
{
  neml_assert(other, "Variable cannot follow a nullptr");

  if (other->src())
    setup_views(other->src());
  else
  {
    _src = other;
    _raw_value = Tensor(other->raw_value().view(sizes()), batch_dim());
  }
}

bool
VariableBase::is_dependent() const
{
  return !currently_solving_nonlinear_system() || is_solve_dependent();
}

Derivative
VariableBase::d(const VariableBase & x)
{
  neml_assert_dbg(_dvalue_d.count(x.name()),
                  "Error retrieving first derivative: ",
                  name(),
                  " does not depend on ",
                  x.name());

  neml_assert_dbg(
      x.is_dependent(),
      "During implicit solve, it is not necessary to calculate derivative with respect to "
      "non-state variables. This error is triggered by an attempt to set the derivative of ",
      name(),
      " with respect to ",
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

  neml_assert_dbg(
      x1.is_dependent() || x2.is_dependent(),
      "During implicit solve, it is not necessary to calculate derivative with respect to "
      "non-state variables. This error is triggered by an attempt to set the derivative of ",
      name(),
      " with respect to ",
      x1.name(),
      " and ",
      x2.name());

  return Derivative(_d2value_d[x1.name()][x2.name()]);
}

Derivative &
Derivative::operator=(const Tensor & val)
{
  _value.index_put_({torch::indexing::Slice()},
                    val.batch_expand_as(_value).base_reshape(_value.base_sizes()));
  return *this;
}
}
