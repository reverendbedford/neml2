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
#include "neml2/misc/math.h"
#include <thread>

namespace neml2
{
std::map<std::thread::id, std::exception_ptr> ComposedModel::_async_exceptions = {};

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

  options.set<bool>("automatic_nonlinear_parameter") = true;
  options.set("automatic_nonlinear_parameter").doc() =
      "Whether to automatically add dependent nonlinear parameters";

  return options;
}

ComposedModel::ComposedModel(const OptionSet & options)
  : Model(options),
    _additional_outputs(options.get<std::vector<VariableName>>("additional_outputs")),
    _auto_nl_param(options.get<bool>("automatic_nonlinear_parameter"))
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
  auto submodels = registered_models();
  if (_auto_nl_param)
    for (auto submodel : submodels)
      for (auto && [pname, pmodel] : submodel->named_nonlinear_parameter_models(/*recursive=*/true))
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
    auto type = item.parent->input_variable(var)->type();

    if (input_axis().has_variable(var))
      neml_assert(input_axis().storage_size(var) == sz,
                  "Multiple sub-models in a ComposedModel define the same input variable ",
                  var,
                  ", but with different shape/storage size.");
    else
      declare_input_variable(sz, type, var);
  }

  // Register output variables
  for (const auto & item : _dependency.outbound_items())
  {
    auto var = item.value;
    auto sz = item.parent->output_axis().storage_size(var);
    auto type = item.parent->output_variable(var)->type();
    neml_assert(!output_axis().has_variable(var),
                "Multiple sub-models in a ComposedModel define the same output variable ",
                var);

    declare_output_variable(sz, type, var);
  }

  // Declare nonlinear parameters
  for (auto submodel : submodels)
  {
    for (auto && [pname, param] : submodel->named_nonlinear_parameters(/*recursive=*/true))
      if (input_axis().has_variable(param->name()))
        _nl_params[pname] = param;
    for (auto && [pname, pmodel] : submodel->named_nonlinear_parameter_models(/*recursive=*/true))
      if (_nl_params.count(pname))
        _nl_param_models[pname] = pmodel;
  }
}

void
ComposedModel::setup()
{
  Model::setup();

  // Setup assembly indices
  for (auto i : registered_models())
  {
    for (auto [i1, i2] : i->input_axis().common_indices(input_axis()))
      _assembly_indices[i].insert_or_assign(i1, std::make_pair(this, i2));

    if (_dependency.node_providers().count(i))
      for (auto dep : _dependency.node_providers().at(i))
        for (auto [i1, i2] : i->input_axis().common_indices(dep->output_axis()))
          _assembly_indices[i].insert_or_assign(i1, std::make_pair(dep, i2));
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

std::map<std::string, const VariableBase *>
ComposedModel::named_nonlinear_parameters(bool /*recursive*/) const
{
  return _nl_params;
}

std::map<std::string, Model *>
ComposedModel::named_nonlinear_parameter_models(bool /*recursive*/) const
{
  return _nl_param_models;
}

void
ComposedModel::allocate_variables(bool in, bool out)
{
  Model::allocate_variables(in, out);

  if (out)
  {
    if (requires_grad())
    {
      _dpout_din[this] = LabeledMatrix::identity(batch_sizes(), input_axis(), options());
      for (auto i : registered_models())
        _dpout_din[i] =
            LabeledMatrix::zeros(batch_sizes(), {&i->output_axis(), &input_axis()}, options());
    }

    if (requires_2nd_grad())
    {
      _d2pout_din2[this] = LabeledTensor3D::zeros(
          batch_sizes(), {&input_axis(), &input_axis(), &input_axis()}, options());
      for (auto i : registered_models())
        _d2pout_din2[i] = LabeledTensor3D::zeros(
            batch_sizes(), {&i->output_axis(), &input_axis(), &input_axis()}, options());
    }
  }
}

void
ComposedModel::setup_output_views()
{
  Model::setup_output_views();

  for (auto i : registered_models())
  {
    // Setup views for dpin/din
    if (requires_grad())
    {
      _dpin_din_views[i].clear();
      for (auto & [i1, dep_i2] : _assembly_indices[i])
        _dpin_din_views[i].push_back(_dpout_din[dep_i2.first].tensor().base_index({dep_i2.second}));
    }

    // Setup views for d2pin/din2
    if (requires_2nd_grad())
    {
      _d2pin_din2_views[i].clear();
      for (auto & [i1, dep_i2] : _assembly_indices[i])
        _d2pin_din2_views[i].push_back(
            _d2pout_din2[dep_i2.first].tensor().base_index({dep_i2.second}));
    }
  }
}

void
ComposedModel::setup_submodel_input_views(VariableStore * host)
{
  for (auto submodel : registered_models())
  {
    for (const auto & item : _dependency.inbound_items())
      if (item.parent == submodel)
        submodel->input_variable(item.value)->setup_views(host->input_variable(item.value));

    for (const auto & [item, providers] : _dependency.item_providers())
      if (item.parent == submodel)
      {
        auto depmodel = providers.begin()->parent;
        submodel->input_variable(item.value)->setup_views(depmodel->output_variable(item.value));
      }

    submodel->setup_submodel_input_views(submodel);
  }
}

void
ComposedModel::set_value(bool out, bool dout_din, bool d2out_din2)
{
  _async_exceptions.clear();
  _async_results.clear();
  for (auto i : registered_models())
    _async_results[i] = std::async(
        std::launch::deferred, &ComposedModel::set_value_async, this, i, out, dout_din, d2out_din2);

  for (auto && [i, future] : _async_results)
    future.wait();

  // Rethrow exceptions raised from other threads to the main thread
  rethrow_exceptions();

  for (auto model : _dependency.end_nodes())
  {
    if (out)
      output_storage().fill(model->output_storage());

    if (dout_din)
      derivative_storage().fill(_dpout_din[model]);

    if (d2out_din2)
      second_derivative_storage().fill(_d2pout_din2[model]);
  }
}

void
ComposedModel::set_value_async(Model * i, bool out, bool dout_din, bool d2out_din2)
{
  try
  {
    // Wait for dependent models
    if (_dependency.node_providers().count(i))
      for (auto dep : _dependency.node_providers().at(i))
        _async_results[dep].wait();

    if (out && !dout_din && !d2out_din2)
      i->value();
    else if (dout_din && !d2out_din2)
      i->value_and_dvalue();
    else if (d2out_din2)
      i->value_and_dvalue_and_d2value();
    else
      throw NEMLException("Unsupported call signature to set_value");

    if (dout_din && !d2out_din2)
      apply_chain_rule(i);

    if (d2out_din2)
      apply_second_order_chain_rule(i);
  }
  catch (...)
  {
    _async_exceptions[std::this_thread::get_id()] = std::current_exception();
  }
}

void
ComposedModel::rethrow_exceptions() const
{
  if (!_async_exceptions.empty())
  {
    std::stringstream error;
    for (auto & [tid, eptr] : _async_exceptions)
      try
      {
        std::rethrow_exception(eptr);
      }
      catch (const std::exception & e)
      {
        error << "During threaded ComposedModel evaluation for '" << name()
              << "', one thread threw an exception with the following message:\n"
              << e.what() << "\n\n";
      }
    throw NEMLException(error.str());
  }
}

void
ComposedModel::apply_chain_rule(Model * i)
{
  if (_dpin_din_views[i].empty())
    return;

  auto dpin_din =
      LabeledMatrix(math::base_cat(_dpin_din_views[i]), {&i->input_axis(), &input_axis()});
  _dpout_din[i].copy_(i->derivative_storage().chain(dpin_din));
}

void
ComposedModel::apply_second_order_chain_rule(Model * i)
{
  if (_dpin_din_views[i].empty())
    return;

  auto dpin_din =
      LabeledMatrix(math::base_cat(_dpin_din_views[i]), {&i->input_axis(), &input_axis()});
  auto d2pin_din2 = LabeledTensor3D(math::base_cat(_d2pin_din2_views[i]),
                                    {&i->input_axis(), &input_axis(), &input_axis()});
  _dpout_din[i].copy_(i->derivative_storage().chain(dpin_din));
  _d2pout_din2[i].copy_(
      i->second_derivative_storage().chain(d2pin_din2, i->derivative_storage(), dpin_din));
}

} // namespace neml2
