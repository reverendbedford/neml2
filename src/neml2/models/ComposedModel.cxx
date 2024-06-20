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
  options.doc() =
      "Compose multiple models together to form a single model. The composed model can then be "
      "treated as a new model and composed with others. The [system documentation](@ref "
      "model-composition) provides in-depth explanation on how the models are composed together.";

  NonlinearSystem::enable_automatic_scaling(options);

  options.set<std::vector<std::string>>("models");
  options.set("models").doc() = "Models being composed together";

  options.set<std::vector<VariableName>>("additional_outputs");
  options.set("additional_outputs").doc() =
      "Extra output variables to be extracted from the composed model in addition to the ones "
      "identified through dependency resolution.";

  options.set<std::vector<std::string>>("priority");
  options.set("priority").doc() =
      "Priorities of models in decreasing order. A model with higher priority will be evaluated "
      "first. This is useful for breaking cyclic dependency.";

  return options;
}

ComposedModel::ComposedModel(const OptionSet & options)
  : Model(options),
    _additional_outputs(options.get<std::vector<VariableName>>("additional_outputs"))
{
  // Each sub-model shall have _independent_ output storage. This is because the same model could
  // be registered as a sub-model by different models, and it could be evaluated with _different_
  // input, and hence yields _different_ output values.
  for (const auto & model_name : options.get<std::vector<std::string>>("models"))
  {
    const auto & submodel =
        register_model<Model>(model_name, 0, /*nonlinear=*/false, /*merge_input=*/false);
    register_nonlinear_params(submodel);
  }

  // Each sub-model may have nonlinear parameters. In our design, nonlinear parameters _are_
  // models. Since we do not want to put the burden of adding nonlinear parameters in the input
  // file through the option 'models', we should do more behind the scenes to register them for
  // the user.
  //
  // Registering nonlinear parameters here ensures dependency resolution. And if a nonlinear
  // parameter is registered by multiple models (which is very possible), we won't have to
  // evaluate the nonlinar parameter over and over again!
  for (const auto & model_name : options.get<std::vector<std::string>>("models"))
  {
    const auto & submodel = Factory::get_object<Model>("Models", model_name);
    register_nonlinear_params(submodel);
  }

  // Add registered models as nodes in the dependency resolver
  for (auto submodel : registered_models())
    _dependency.add_node(submodel);
  for (const auto & var : _additional_outputs)
    _dependency.add_additional_outbound_item(var);

  // Define priority in the event of cyclic dependency
  auto priority_order = options.get<std::vector<std::string>>("priority");
  size_t priority = priority_order.empty() ? 0 : priority_order.size() - 1;
  for (const auto & model_name : priority_order)
    _dependency.set_priority(registered_model(model_name), priority--);

  // Resolve the dependency
  _dependency.unique_item_provider() = true;
  _dependency.unique_item_consumer() = false;
  _dependency.resolve();

  // Sort the registered models by dependency resolution
  _registered_models = _dependency.resolution();

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
ComposedModel::register_nonlinear_params(const Model & m)
{
  for (auto && [pname, param] : m.nl_params())
  {
    neml_assert_dbg(param->name().size() == 1, "Internal parameter name error");

    auto submodel = Factory::get_object_ptr<Model>("Models", param->name().vec()[0]);
    _registered_models.push_back(submodel.get());

    // Nonlinear parameters could be nested...
    register_nonlinear_params(*submodel);
  }
}

void
ComposedModel::check_AD_limitation() const
{
  if (_AD_1st_deriv || _AD_2nd_deriv)
    throw NEMLException(
        "ComposedModel does not use automatic differentiation. _use_AD_first_derivative and "
        "_use_AD_second_derivative should be set to false.");
}

void
ComposedModel::allocate_variables(int deriv_order, bool options_changed)
{
  Model::allocate_variables(deriv_order, options_changed);

  if (options_changed)
    _din_din = LabeledMatrix::identity(batch_sizes(), input_axis(), options());
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

  for (auto i : registered_models())
  {
    if (out && !dout_din && !d2out_din2)
      i->value();
    else if (out && dout_din && !d2out_din2)
      i->value_and_dvalue();
    else if (out && dout_din && d2out_din2)
      i->value_and_dvalue_and_d2value();
    else
      throw NEMLException("Unsupported call signature to set_value");

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
