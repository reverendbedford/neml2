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

#include "neml2/models/VariableStore.h"
#include "neml2/models/Model.h"

namespace neml2
{
VariableStore::VariableStore(const OptionSet & options, Model * object)
  : _object(object),
    _options(options),
    _input_axis(declare_axis("input")),
    _output_axis(declare_axis("output"))
{
}

LabeledAxis &
VariableStore::declare_axis(const std::string & name)
{
  neml_assert(!_axes.has_key(name),
              "Trying to declare an axis named ",
              name,
              ", but an axis with the same name already exists.");

  auto axis = std::make_unique<LabeledAxis>();
  return *_axes.set_pointer(name, std::move(axis));
}

void
VariableStore::setup_layout()
{
  input_axis().setup_layout();
  output_axis().setup_layout();
}

VariableBase *
VariableStore::input_variable(const VariableName & name)
{
  return _input_views.query_value(name);
}

VariableBase *
VariableStore::output_variable(const VariableName & name)
{
  return _output_views.query_value(name);
}

TensorType
VariableStore::input_type(const VariableName & name) const
{
  const auto * var_ptr = _input_views.query_value(name);
  neml_assert(var_ptr, "Input variable ", name, " does not exist.");
  return var_ptr->type();
}

TensorType
VariableStore::output_type(const VariableName & name) const
{
  const auto * var_ptr = _output_views.query_value(name);
  neml_assert(var_ptr, "Output variable ", name, " does not exist.");
  return var_ptr->type();
}

void
VariableStore::cache(TensorShapeRef batch_shape)
{
  for (auto && [name, var] : input_variables())
    var.cache(batch_shape);
  for (auto && [name, var] : output_variables())
    var.cache(batch_shape);
}

void
VariableStore::allocate_variables(TensorShapeRef batch_shape,
                                  const torch::TensorOptions & options,
                                  bool in,
                                  bool out,
                                  bool dout_din,
                                  bool d2out_din2)
{
  // Allocate input storage only if this is a host model
  if (in && _object->host() == _object)
    _in = LabeledVector::zeros(batch_shape, {&input_axis()}, options);

  // Allocate output storage
  if (out)
    _out = LabeledVector::zeros(batch_shape, {&output_axis()}, options);

  if (dout_din)
    _dout_din = LabeledMatrix::zeros(batch_shape, {&output_axis(), &input_axis()}, options);

  if (d2out_din2)
    _d2out_din2 = LabeledTensor3D::zeros(
        batch_shape, {&output_axis(), &input_axis(), &input_axis()}, options);
}

void
VariableStore::setup_input_views(VariableStore * host)
{
  neml_assert_dbg(host || _object->host<VariableStore>() == host,
                  "setup_input_views called on a non-host model without specifying the host as an "
                  "argument");
  for (auto && [name, var] : input_variables())
  {
    if (_object->host<VariableStore>() == host)
      var.setup_views(&host->input_storage());
    else
      var.setup_views(host->input_variable(name));
  }
}

void
VariableStore::setup_output_views(bool out, bool dout_din, bool d2out_din2)
{
  for (auto && [name, var] : output_variables())
    var.setup_views(out ? &_out : nullptr,
                    dout_din ? &_dout_din : nullptr,
                    d2out_din2 ? &_d2out_din2 : nullptr);
}

void
VariableStore::zero(bool dout_din, bool d2out_din2)
{
  if (dout_din)
    _dout_din.zero_();

  if (d2out_din2)
    _d2out_din2.zero_();
}
} // namespace neml2
