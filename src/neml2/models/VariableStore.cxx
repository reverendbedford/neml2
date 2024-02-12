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

#include "neml2/models/VariableStore.h"

namespace neml2
{
VariableStore::VariableStore(const OptionSet & options, NEML2Object * object)
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
VariableStore::input_view(const VariableName & name)
{
  return _input_views.query_value(name);
}

VariableBase *
VariableStore::output_view(const VariableName & name)
{
  return _output_views.query_value(name);
}

void
VariableStore::send_input_to(const torch::TensorOptions & options)
{
  if (_object->host() == _object)
    _in = _in.to(options);
}

void
VariableStore::send_output_to(const torch::TensorOptions & options,
                              bool out,
                              bool dout_din,
                              bool d2out_din2)
{
  if (out)
    _out = _out.to(options);
  if (dout_din)
    _dout_din = _dout_din.to(options);
  if (d2out_din2)
    _d2out_din2 = _d2out_din2.to(options);
}

void
VariableStore::cache(TorchShapeRef batch_shape)
{
  for (auto && [name, var] : input_views())
    var.cache(batch_shape);
  for (auto && [name, var] : output_views())
    var.cache(batch_shape);
}

void
VariableStore::allocate_variables(TorchShapeRef batch_shape,
                                  const torch::TensorOptions & options,
                                  int deriv_order)
{
  // Allocate input storage only if this is a host model
  if (_object->host() == _object)
    _in = LabeledVector::zeros(batch_shape, {&input_axis()}, options);

  neml_assert_dbg(deriv_order >= 0 && deriv_order <= 2,
                  "Expect derivative order from [0, 2], got ",
                  deriv_order);

  // Allocate output storage
  if (deriv_order >= 0)
    _out = LabeledVector::zeros(batch_shape, {&output_axis()}, options);

  if (deriv_order >= 1)
    _dout_din = LabeledMatrix::zeros(batch_shape, {&output_axis(), &input_axis()}, options);

  if (deriv_order >= 2)
    _d2out_din2 = LabeledTensor3D::zeros(
        batch_shape, {&output_axis(), &input_axis(), &input_axis()}, options);
}

void
VariableStore::setup_input_views()
{
  for (auto && [name, var] : input_views())
    var.setup_views(&_object->host<VariableStore>()->input_storage());
}

void
VariableStore::setup_output_views(bool out, bool dout_din, bool d2out_din2)
{
  for (auto && [name, var] : output_views())
    var.setup_views(out ? &_out : nullptr,
                    dout_din ? &_dout_din : nullptr,
                    d2out_din2 ? &_d2out_din2 : nullptr);
}

void
VariableStore::detach_and_zero(bool out, bool dout_din, bool d2out_din2)
{
  bool out_detached = false;
  bool dout_din_detached = false;
  bool d2out_din2_detached = false;

  // Detach and zero per request
  if (out)
  {
    if (_out.tensor().requires_grad())
    {
      _out.tensor().detach_();
      out_detached = true;
    }
  }

  if (dout_din)
  {
    if (_dout_din.tensor().requires_grad())
    {
      _dout_din.tensor().detach_();
      dout_din_detached = true;
    }
    _dout_din.zero_();
  }

  if (d2out_din2)
  {
    if (_d2out_din2.tensor().requires_grad())
    {
      _d2out_din2.tensor().detach_();
      d2out_din2_detached = true;
    }
    _d2out_din2.zero_();
  }

  // If the storage is detached in-place, we need to reconfigure all the views.
  setup_output_views(out_detached, dout_din_detached, d2out_din2_detached);
}
} // namespace neml2
