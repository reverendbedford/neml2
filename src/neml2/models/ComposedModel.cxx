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

#include "neml2/models/ComposedModel.h"

namespace neml2
{
register_NEML2_object(ComposedModel);

OptionSet
ComposedModel::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<std::vector<std::string>>("models");
  options.set<std::vector<VariableName>>("additional_outputs");
  return options;
}

ComposedModel::ComposedModel(const OptionSet & options)
  : Model(options),
    _additional_outputs(options.get<std::vector<VariableName>>("additional_outputs"))
{
  // Add registered models as nodes in the dependency resolver
  for (const auto & model_name : options.get<std::vector<std::string>>("models"))
    register_model<Model>(model_name, 0, /*merge_input=*/false);

  // Add registered models as nodes in the dependency resolver
  for (auto submodel : registered_models())
    _dependency.add_node(submodel);
  for (const auto & var : _additional_outputs)
    _dependency.add_additional_outbound_item(var);

  // Resolve the dependency
  _dependency.unique_item_provider() = true;
  _dependency.unique_item_consumer() = false;
  _dependency.resolve();

  // Register input variables
  for (const auto & item : _dependency.inbound_items())
  {
    auto var = item.value;
    auto sz = item.parent->input_axis().storage_size(var);

    if (input_axis().has_variable(var))
      neml_assert(input_axis().storage_size(var) == sz,
                  "Multiple sub-models in a ComposedModel define the same input variable ",
                  var,
                  ", but with different shape/storage size.");
    else
      declare_input_variable(sz, var);
  }

  // Register output variables
  for (const auto & item : _dependency.outbound_items())
  {
    auto var = item.value;
    auto sz = item.parent->output_axis().storage_size(var);
    neml_assert(!output_axis().has_variable(var),
                "Multiple sub-models in a ComposedModel define the same output variable ",
                var);

    declare_output_variable(sz, var);
  }
}

void
ComposedModel::to(const torch::TensorOptions & options)
{
  Model::to(options);
  _din_din = _din_din.to(options);
}

void
ComposedModel::allocate_variables(TorchShapeRef batch_shape, const torch::TensorOptions & options)
{
  Model::allocate_variables(batch_shape, options);
  _din_din = LabeledMatrix::identity(batch_shape, input_axis(), options);
}

void
ComposedModel::setup_submodel_input_views()
{
  for (auto submodel : registered_models())
  {
    for (const auto & item : _dependency.inbound_items())
      if (item.parent == submodel)
        item.parent->input_view(item.value)->setup_views(input_view(item.value));

    for (const auto & [item, providers] : _dependency.item_providers())
      if (item.parent == submodel)
        item.parent->input_view(item.value)
            ->setup_views(&providers.begin()->parent->output_storage());

    submodel->setup_submodel_input_views();
  }
}

void
ComposedModel::set_value(bool out, bool dout_din, bool d2out_din2)
{
  clear_chain_rule_cache();

  for (auto i : _dependency.resolution())
  {
    i->set_value(out, dout_din, d2out_din2);

    if (dout_din && !d2out_din2)
      apply_chain_rule(i);

    if (d2out_din2)
      apply_second_order_chain_rule(i);
  }

  for (auto model : _dependency.end_nodes())
  {
    if (out)
      output_storage().fill(model->output_storage());

    if (dout_din)
      derivative_storage().fill(_dpout_din[model]);

    if (d2out_din2)
      second_derivative_storage().fill(_d2pout_din2[model]);
  }

  clear_chain_rule_cache();
}

void
ComposedModel::apply_chain_rule(Model * i)
{
  auto dpin_din = LabeledMatrix::empty(batch_sizes(), {&i->input_axis(), &input_axis()}, options());
  dpin_din.fill(_din_din);

  if (_dependency.node_providers().count(i))
    for (auto dep : _dependency.node_providers().at(i))
      dpin_din.fill(_dpout_din[dep]);

  _dpout_din[i] = i->derivative_storage().chain(dpin_din);
}

void
ComposedModel::apply_second_order_chain_rule(Model * i)
{
  auto dpin_din = LabeledMatrix::empty(batch_sizes(), {&i->input_axis(), &input_axis()}, options());
  auto d2pin_din2 = LabeledTensor3D::zeros(
      batch_sizes(), {&i->input_axis(), &input_axis(), &input_axis()}, options());
  dpin_din.fill(_din_din);

  if (_dependency.node_providers().count(i))
    for (auto dep : _dependency.node_providers().at(i))
    {
      dpin_din.fill(_dpout_din[dep]);
      d2pin_din2.fill(_d2pout_din2[dep]);
    }

  _dpout_din[i] = i->derivative_storage().chain(dpin_din);
  _d2pout_din2[i] =
      i->second_derivative_storage().chain(d2pin_din2, i->derivative_storage(), dpin_din);
}

} // namespace neml2
