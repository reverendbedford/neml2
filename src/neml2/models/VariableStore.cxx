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
#include "neml2/misc/math.h"
#include "neml2/tensors/LabeledTensor.h"

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
      _in = std::make_unique<LabeledVector>(
          batch_shape, std::array<const LabeledAxis *, 1>{&input_axis()}, options);

    // Allocate output storage
    if (out)
      _out = std::make_unique<LabeledVector>(
          batch_shape, std::array<const LabeledAxis *, 1>{&output_axis()}, options);

    if (dout_din)
      _dout_din = std::make_unique<LabeledMatrix>(
          batch_shape, std::array<const LabeledAxis *, 2>{&output_axis(), &input_axis()}, options);

    if (d2out_din2)
      _d2out_din2 = std::make_unique<LabeledTensor3D>(
          batch_shape,
          std::array<const LabeledAxis *, 3>{&output_axis(), &input_axis(), &input_axis()},
          options);
  }
  else if (_assembly_mode == AssemblyMode::CONCATENATION)
  {
  }
  else
    throw NEMLException("Unknown assembly mode");
}

// LabeledVector
// VariableStore::assemble_output() const
// {
//   neml_assert_dbg(_assembly_mode == AssemblyMode::CONCATENATION,
//                   "Use output_storage() for inplace assembly mode");
//   return LabeledVector(math::cat(__out, -1), {&_output_axis});
// }

// LabeledMatrix
// VariableStore::assemble_derivative() const
// {
//   neml_assert_dbg(_assembly_mode == AssemblyMode::CONCATENATION,
//                   "Use derivative_storage()() for inplace assembly mode");
//   std::vector<BatchTensor> rows(__dout_din.size());
//   std::transform(__dout_din.begin(),
//                  __dout_din.end(),
//                  rows.begin(),
//                  [](const auto & row) { return math::cat(row, -1); });
//   return LabeledMatrix(math::cat(rows, -2), {&_output_axis, &_input_axis});
// }

// LabeledTensor3D
// VariableStore::assemble_second_derivative() const
// {
//   neml_assert_dbg(_assembly_mode == AssemblyMode::CONCATENATION,
//                   "Use second_derivative_storage() for inplace assembly mode");
//   std::vector<BatchTensor> rows(__d2out_din2.size());
//   std::transform(__d2out_din2.begin(),
//                  __d2out_din2.end(),
//                  rows.begin(),
//                  [](const auto & row)
//                  {
//                    std::vector<BatchTensor> cols(row.size());
//                    std::transform(row.begin(),
//                                   row.end(),
//                                   cols.begin(),
//                                   [](const auto & col) { return math::cat(col, -1); });
//                    return math::cat(cols, -2);
//                  });
//   return LabeledTensor3D(math::cat(rows, -3), {&_output_axis, &_input_axis, &_input_axis});
// }

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
  }
  else
    throw NEMLException("Unknown assembly mode");
}

void
VariableStore::setup_output_views(bool out, bool dout_din, bool d2out_din2)
{
  if (_assembly_mode == AssemblyMode::INPLACE)
  {
    neml_assert(!out || !_out, "Output storage not allocated");
    neml_assert(!dout_din || !_dout_din, "Derivative storage not allocated");
    neml_assert(!d2out_din2 || !_d2out_din2, "Second derivative storage not allocated");

    for (auto && [name, var] : output_views())
      var.setup_views(out ? _out.get() : nullptr,
                      dout_din ? _dout_din.get() : nullptr,
                      d2out_din2 ? _d2out_din2.get() : nullptr);
  }
  else if (_assembly_mode == AssemblyMode::CONCATENATION)
  {
  }
  else
    throw NEMLException("Unknown assembly mode");
}

void
VariableStore::detach_and_zero(bool /*out*/, bool dout_din, bool d2out_din2)
{
  if (_assembly_mode == AssemblyMode::INPLACE)
  {
    if (dout_din)
      _dout_din->zero_();

    if (d2out_din2)
      _d2out_din2->zero_();
  }
  else if (_assembly_mode == AssemblyMode::CONCATENATION)
  {
  }
  else
    throw NEMLException("Unknown assembly mode");
}
} // namespace neml2
