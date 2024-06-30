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
#include "neml2/models/Model.h"

namespace neml2
{
OptionSet
VariableStore::expected_options()
{
  OptionSet options;
  options.set<EnumSelection>("_assembly_mode") = EnumSelection(
      {"INPLACE", "CONCATENATION"},
      {static_cast<int>(AssemblyMode::INPLACE), static_cast<int>(AssemblyMode::CONCATENATION)},
      "CONCATENATION");
  options.set("_assembly_mode").suppressed() = true;
  return options;
}

VariableStore::VariableStore(const OptionSet & options, NEML2Object * object)
  : _object(object),
    _options(options),
    _assembly_mode(options.get<EnumSelection>("_assembly_mode").as<AssemblyMode>()),
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
                                  bool in,
                                  bool out,
                                  bool dout_din,
                                  bool d2out_din2)
{
  if (_assembly_mode == AssemblyMode::INPLACE)
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
  else if (_assembly_mode == AssemblyMode::CONCATENATION)
  {
    // Pre-sort the input and output variables so that the final assembly using concatenation
    // would just be a simple torch::cat.
    __in_idx = _input_axis.assembly_indices(_input_views.keys());
    __out_idx = _output_axis.assembly_indices(_output_views.keys());

    // Resize the storage vectors accordingly
    const auto ninput = __in_idx.size();
    const auto noutput = __out_idx.size();

    if (in && _object->host() == _object)
    {
      __in.resize(ninput);
      for (const auto & [var, i] : __in_idx)
        __in[i] = BatchTensor::zeros(batch_shape, {_input_axis.storage_size(var)}, options);
    }

    if (out)
    {
      __out.resize(noutput);
      for (const auto & [var, i] : __out_idx)
        __out[i] = BatchTensor::zeros(batch_shape, {_output_axis.storage_size(var)}, options);
    }

    if (dout_din)
    {
      __dout_din.resize(noutput);
      for (const auto & [yvar, i] : __out_idx)
      {
        __dout_din[i].resize(ninput);
        for (const auto & [xvar, j] : __in_idx)
          __dout_din[i][j] =
              BatchTensor::zeros(batch_shape,
                                 {_output_axis.storage_size(yvar), _input_axis.storage_size(xvar)},
                                 options);
      }
    }

    if (d2out_din2)
    {
      __d2out_din2.resize(noutput);
      for (const auto & [yvar, i] : __out_idx)
      {
        __d2out_din2[i].resize(ninput);
        for (const auto & [xvar1, j] : __in_idx)
        {
          __d2out_din2[i][j].resize(ninput);
          for (const auto & [xvar2, k] : __in_idx)
            __d2out_din2[i][j][k] = BatchTensor::zeros(batch_shape,
                                                       {_output_axis.storage_size(yvar),
                                                        _input_axis.storage_size(xvar1),
                                                        _input_axis.storage_size(xvar2)},
                                                       options);
        }
      }
    }
  }
  else
    throw NEMLException("Unknown assembly mode");
}

BatchTensor &
VariableStore::input_storage(const VariableName & x)
{
  return __in[__in_idx.at(x)];
}

const BatchTensor &
VariableStore::input_storage(const VariableName & x) const
{
  return __in[__in_idx.at(x)];
}

BatchTensor &
VariableStore::output_storage(const VariableName & y)
{
  return __out[__out_idx.at(y)];
}

const BatchTensor &
VariableStore::output_storage(const VariableName & y) const
{
  return __out[__out_idx.at(y)];
}

BatchTensor &
VariableStore::derivative_storage(const VariableName & y, const VariableName & x)
{
  return __dout_din[__out_idx.at(y)][__in_idx.at(x)];
}

const BatchTensor &
VariableStore::derivative_storage(const VariableName & y, const VariableName & x) const
{
  return __dout_din[__out_idx.at(y)][__in_idx.at(x)];
}

BatchTensor &
VariableStore::second_derivative_storage(const VariableName & y,
                                         const VariableName & x1,
                                         const VariableName & x2)
{
  return __d2out_din2[__out_idx.at(y)][__in_idx.at(x1)][__in_idx.at(x2)];
}

const BatchTensor &
VariableStore::second_derivative_storage(const VariableName & y,
                                         const VariableName & x1,
                                         const VariableName & x2) const
{
  return __d2out_din2[__out_idx.at(y)][__in_idx.at(x1)][__in_idx.at(x2)];
}

void
VariableStore::setup_input_views()
{
  if (_assembly_mode == AssemblyMode::INPLACE)
  {
    for (auto && [name, var] : input_views())
      var.setup_views(&_object->host<VariableStore>()->input_storage());
  }
  else if (_assembly_mode == AssemblyMode::CONCATENATION)
  {
    for (auto && [name, var] : input_views())
      var.setup_views(&_object->host<VariableStore>()->input_storage(name));
  }
  else
    throw NEMLException("Unknown assembly mode");
}

void
VariableStore::setup_output_views(bool out, bool dout_din, bool d2out_din2)
{
  if (_assembly_mode == AssemblyMode::INPLACE)
  {
    neml_assert(!out || !_out.axes().empty(), "Output storage not allocated");
    neml_assert(!dout_din || !_dout_din.axes().empty(), "Derivative storage not allocated");
    neml_assert(!d2out_din2 || !_d2out_din2.axes().empty(),
                "Second derivative storage not allocated");
    for (auto && [name, var] : output_views())
      var.setup_views(out ? &_out : nullptr,
                      dout_din ? &_dout_din : nullptr,
                      d2out_din2 ? &_d2out_din2 : nullptr);
  }
  else if (_assembly_mode == AssemblyMode::CONCATENATION)
  {
    neml_assert(!out || !__out.empty(), "Output storage not allocated");
    neml_assert(!dout_din || !__dout_din.empty(), "Derivative storage not allocated");
    neml_assert(!d2out_din2 || !__d2out_din2.empty(), "Second derivative storage not allocated");
    for (auto && [name, var] : output_views())
    {
      const auto var_idx = __out_idx[name];
      var.setup_views(__in_idx,
                      out ? &__out[var_idx] : nullptr,
                      dout_din ? &__dout_din[var_idx] : nullptr,
                      d2out_din2 ? &__d2out_din2[var_idx] : nullptr);
    }
  }
  else
    throw NEMLException("Unknown assembly mode");
}

void
VariableStore::detach_and_zero(bool out, bool dout_din, bool d2out_din2)
{
  // No-op for concatenation mode
  if (_assembly_mode == AssemblyMode::CONCATENATION)
    return;

  // Detach and zero per request
  if (out)
  {
    if (_out.tensor().requires_grad())
      _out.tensor().detach_();
    _out.zero_();
  }

  if (dout_din)
  {
    if (_dout_din.tensor().requires_grad())
      _dout_din.tensor().detach_();
    _dout_din.zero_();
  }

  if (d2out_din2)
  {
    if (_d2out_din2.tensor().requires_grad())
      _d2out_din2.tensor().detach_();
    _d2out_din2.zero_();
  }

  // Since the storage is detached in-place, we need to reconfigure all the views.
  setup_output_views(out, dout_din, d2out_din2);
}
} // namespace neml2
