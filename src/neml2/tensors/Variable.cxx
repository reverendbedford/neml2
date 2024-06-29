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
VariableBase::cache(TorchShapeRef batch_shape)
{
  _batch_sizes = batch_shape.vec();
}

void
VariableBase::setup_views(const VariableBase * other)
{
  neml_assert(_assembly_mode == other->_assembly_mode, "Assembly mode mismatch");
  neml_assert(other, "other != nullptr");

  if (_assembly_mode == Model::AssemblyMode::INPLACE)
    setup_views(
        other->_value_storage, other->_derivative_storage, other->_second_derivative_storage);
  else if (_assembly_mode == Model::AssemblyMode::CONCATENATION)
    setup_views(
        __args_idx, other->__value, other->_derivative_storage, other->_second_derivative_storage);
  else
    throw NEMLException("Unknown assembly mode");
}

void
VariableBase::setup_views(const LabeledVector * value,
                          const LabeledMatrix * deriv,
                          const LabeledTensor3D * secderiv)
{
  neml_assert(_assembly_mode == Model::AssemblyMode::INPLACE,
              "This method only works in inplace assembly mode");

  _value_storage = value;
  _derivative_storage = deriv;
  _second_derivative_storage = secderiv;
}

void
VariableBase::setup_views(BatchTensor * value)
{
  neml_assert(_assembly_mode == Model::AssemblyMode::CONCATENATION,
              "This method only works in concatenation assembly mode");

  __raw_value = value;
}

void
VariableBase::setup_views(const std::unordered_map<VariableName, size_t> * idx,
                          BatchTensor * value,
                          std::vector<BatchTensor *> * deriv,
                          std::vector<std::vector<BatchTensor *>> * secderiv)
{
  neml_assert(_assembly_mode == Model::AssemblyMode::CONCATENATION,
              "This method only works in concatenation assembly mode");

  __args_idx = idx;
  __raw_value = value;
  __dvalue_d = deriv;
  __d2value_d = secderiv;
}

void
VariableBase::reinit_views(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert(_assembly_mode == Model::AssemblyMode::INPLACE,
              "This method only works in inplace assembly mode");

  if (out)
    neml_assert(_value_storage, "Variable value storage not initialized.");
  if (dout_din)
    neml_assert(_derivative_storage, "Variable derivative storage not initialized.");
  if (d2out_din2)
    neml_assert(_second_derivative_storage, "Variable second derivative storage not initialized.");

  if (out)
    _raw_value = (*_value_storage)(name());

  if (dout_din)
    for (const auto & arg : args())
      _dvalue_d[arg] = (*_derivative_storage)(name(), arg);

  if (d2out_din2)
    for (const auto & arg1 : args())
      for (const auto & arg2 : args())
        _d2value_d[arg1][arg2] = (*_second_derivative_storage)(name(), arg1, arg2);
}

const LabeledVector &
VariableBase::value_storage() const
{
  neml_assert_dbg(_assembly_mode == Model::AssemblyMode::INPLACE,
                  "This method only works in inplace assembly mode");

  neml_assert_dbg(_value_storage, "Variable value storage not initialized.");
  return *_value_storage;
}

const LabeledMatrix &
VariableBase::derivative_storage() const
{
  neml_assert_dbg(_assembly_mode == Model::AssemblyMode::INPLACE,
                  "This method only works in inplace assembly mode");

  neml_assert_dbg(_derivative_storage, "Variable derivative storage not initialized.");
  return *_derivative_storage;
}

const LabeledTensor3D &
VariableBase::second_derivative_storage() const
{
  neml_assert_dbg(_assembly_mode == Model::AssemblyMode::INPLACE,
                  "This method only works in inplace assembly mode");

  neml_assert_dbg(_second_derivative_storage, "Variable 2nd derivative storage not initialized.");
  return *_second_derivative_storage;
}

Derivative
VariableBase::d(const VariableBase & x)
{
  if (_assembly_mode == Model::AssemblyMode::INPLACE)
  {
    neml_assert_dbg(_dvalue_d.count(x.name()),
                    "Error retrieving first derivative: ",
                    name(),
                    " does not depend on ",
                    x.name());
    return Derivative(_dvalue_d[x.name()]);
  }
  else if (_assembly_mode == Model::AssemblyMode::CONCATENATION)
  {
    neml_assert_dbg(__args_idx->count(x.name()) && __dvalue_d->size() > (*__args_idx)[x.name()],
                    "Error retrieving first derivative: ",
                    name(),
                    " does not depend on ",
                    x.name());
    return Derivative(*(*__dvalue_d)[(*__args_idx)[x.name()]]);
  }
  else
    throw NEMLException("Unknown assembly mode");
}

Derivative
VariableBase::d(const VariableBase & x1, const VariableBase & x2)
{
  if (_assembly_mode == Model::AssemblyMode::INPLACE)
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
  else if (_assembly_mode == Model::AssemblyMode::CONCATENATION)
  {
    neml_assert_dbg(__args_idx->count(x1.name()) && __d2value_d->size() > (*__args_idx)[x1.name()],
                    "Error retrieving second derivative: ",
                    name(),
                    " does not depend on ",
                    x1.name());
    neml_assert_dbg(__args_idx->count(x2.name()) &&
                        (*__d2value_d)[(*__args_idx)[x1.name()]].size() > (*__args_idx)[x2.name()],
                    "Error retrieving second derivative: d(",
                    name(),
                    ")/d(",
                    x1.name(),
                    ") does not depend on ",
                    x2.name());
    return Derivative(*(*__d2value_d)[(*__args_idx)[x1.name()]][(*__args_idx)[x2.name()]]);
  }
  else
    throw NEMLException("Unknown assembly mode");
}

void
Derivative::operator=(const BatchTensor & val)
{
  if (_assembly_mode == Model::AssemblyMode::INPLACE)
    _value.index_put_({torch::indexing::Slice()},
                      val.batch_expand(_batch_sizes).base_reshape(_base_sizes));
  else if (_assembly_mode == Model::AssemblyMode::CONCATENATION)
    _value = val.batch_expand(_batch_sizes).base_reshape(_base_sizes);
  else
    throw NEMLException("Unknown assembly mode");
}
}
