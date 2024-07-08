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
#include "neml2/models/ChainRule.h"

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
    register_model<Model>(model_name, 0, /*nonlinear=*/false, /*merge_input=*/false);

  // Each sub-model may have nonlinear parameters. In our design, nonlinear parameters _are_
  // models. Since we do not want to put the burden of "adding nonlinear parameters in the input
  // file through the option 'models'" on users, we should do more behind the scenes to register
  // them.
  //
  // Registering nonlinear parameters here ensures dependency resolution. And if a nonlinear
  // parameter is registered by multiple models (which is very possible), we won't have to
  // evaluate the nonlinar parameter over and over again!
  for (auto submodel : registered_models())
    for (auto && [pname, pmodel] : submodel->named_nonlinear_parameter_models(/*recursive=*/false))
      _registered_models.push_back(pmodel);

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

  // Create the ChainRule object to manage chain rule application
  if (assembly_mode() == AssemblyMode::INPLACE)
    _chain_rule = std::make_unique<ChainRuleImpl<AssemblyMode::INPLACE>>(this);
  else if (assembly_mode() == AssemblyMode::CONCATENATION)
    _chain_rule = std::make_unique<ChainRuleImpl<AssemblyMode::INPLACE>>(this);
  else
    throw NEMLException("Unknown assembly mode");
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
ComposedModel::allocate_variables()
{
  Model::allocate_variables();
  _chain_rule->allocate_variables();
}

void
ComposedModel::setup_submodel_input_views()
{
  for (auto submodel : registered_models())
  {
    for (const auto & item : _dependency.inbound_items())
      if (item.parent == submodel)
        submodel->input_view(item.value)->setup_views(&input_view(item.value)->storage());

    for (const auto & [item, providers] : _dependency.item_providers())
      if (item.parent == submodel)
        submodel->input_view(item.value)->setup_views(&providers.begin()->parent->output_storage());

    submodel->setup_submodel_input_views();
  }
}

void
ComposedModel::set_value(bool out, bool dout_din, bool d2out_din2)
{
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
  }

  _chain_rule->apply(/*second_order=*/d2out_din2);

  for (auto i : _dependency.end_nodes())
  {
    if (out)
      output_storage().collect_(i->output_storage());

    if (dout_din)
      derivative_storage().collect_(_chain_rule->total_derivative(i));

    if (d2out_din2)
      second_derivative_storage().collect_(_chain_rule->total_second_derivative(i));
  }

  _chain_rule->clear();
}

} // namespace neml2
