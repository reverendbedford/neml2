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

#include "neml2/models/Model.h"

namespace neml2
{
Model::Stage Model::stage = UPDATING;

OptionSet
Model::expected_options()
{
  OptionSet options = Data::expected_options();
  options += NonlinearSystem::expected_options();
  NonlinearSystem::disable_automatic_scaling(options);

  options.section() = "Models";

  options.set<bool>("_use_AD_first_derivative") = false;
  options.set<bool>("_use_AD_second_derivative") = false;
  options.set<int>("_extra_derivative_order") = 0;
  options.set<bool>("_nonlinear_system") = false;
  options.set<bool>("_inference_mode") = false;

  options.set("_use_AD_first_derivative").suppressed() = true;
  options.set("_use_AD_second_derivative").suppressed() = true;
  options.set("_extra_derivative_order").suppressed() = true;
  options.set("_nonlinear_system").suppressed() = true;
  options.set("_inference_mode").suppressed() = true;

  return options;
}

Model::Model(const OptionSet & options)
  : Data(options),
    ParameterStore(options, this),
    VariableStore(options, this),
    NonlinearSystem(options),
    _AD_1st_deriv(options.get<bool>("_use_AD_first_derivative")),
    _AD_2nd_deriv(options.get<bool>("_use_AD_second_derivative")),
    _options(default_tensor_options()),
    _deriv_order(-1),
    _extra_deriv_order(options.get<int>("_extra_derivative_order")),
    _nonlinear_system(options.get<bool>("_nonlinear_system")),
    _inference_mode(options.get<bool>("_inference_mode"))
{
  check_AD_limitation();
}

std::vector<Diagnosis>
Model::preflight() const
{
  neml_assert(host() == this, "This method should only be called on the host model.");

  std::vector<Diagnosis> errors;

  // Check for statefulness
  if (input_axis().has_subaxis("old_state"))
  {
    if (!output_axis().has_subaxis("state"))
      errors.push_back(
          make_diagnosis(name(),
                         ": input axis has sub-axis 'old_state', but output axis does not "
                         "have sub-axis 'state'."));
    else
    {
      auto s_vars = output_axis().subaxis("state").variable_accessors(/*recursive=*/true);
      for (auto var : input_axis().subaxis("old_state").variable_accessors(/*recursive=*/true))
        if (!s_vars.count(var))
          errors.push_back(make_diagnosis(name(),
                                          ": input axis has old state named ",
                                          var,
                                          ", but it doesn't exist on the output axis."));
    }
  }

  return errors;
}

void
Model::setup()
{
  // Setup input and output axes
  setup_layout();
}

void
Model::reinit(const BatchTensor & tensor, int deriv_order)
{
  reinit(tensor.batch_sizes(), deriv_order, tensor.device(), tensor.scalar_type());
}

void
Model::reinit(TorchShapeRef batch_shape,
              int deriv_order,
              const torch::Device & device,
              const torch::Dtype & dtype)
{
  neml_assert(host() == this, "This method should only be called on the host model.");

  // Cache configuration
  cache(batch_shape, deriv_order, device, dtype);

  // Sync buffers and parameters
  send_buffers_to(options());
  send_parameters_to(options());

  // Allocate variable storage and set up variable views
  reinit(/*in=*/true, /*out=*/true);
}

void
Model::reinit(bool in, bool out)
{
  // Allocate variable storage
  allocate_variables(in, out);

  // Setup variable views
  setup_input_views(this);
  setup_output_views();
  setup_nonlinear_system();
}

void
Model::prepare()
{
  if (_inference_mode)
    zero();
  else
    reinit(false, true);
}

void
Model::allocate_variables(bool in, bool out)
{
#ifndef NDEBUG
  _evaluated_once = false;
#endif

  VariableStore::allocate_variables(batch_sizes(),
                                    options(),
                                    /*in=*/in,
                                    /*out=*/out,
                                    /*dout_din=*/out && requires_grad(),
                                    /*d2out_din2=*/out && requires_2nd_grad());

  if (in && is_nonlinear_system())
  {
    _ndof = output_axis().storage_size("residual");
    _solution = BatchTensor::empty(batch_sizes(), _ndof, options());
  }

  for (auto submodel : registered_models())
    submodel->allocate_variables(in, out);
}

void
Model::setup_input_views(VariableStore * host)
{
  VariableStore::setup_input_views(host);
  setup_submodel_input_views(this);
}

void
Model::setup_submodel_input_views(VariableStore * host)
{
  for (auto submodel : registered_models())
  {
    for (auto && [name, var] : submodel->input_views())
      var.setup_views(&host->input_view(name)->value_storage());
    submodel->setup_submodel_input_views(submodel);
  }
}

void
Model::setup_output_views()
{
  VariableStore::setup_output_views(true, requires_grad(), requires_2nd_grad());
  setup_submodel_output_views();
}

void
Model::setup_submodel_output_views()
{
  for (auto submodel : registered_models())
    submodel->setup_output_views();
}

void
Model::setup_nonlinear_system()
{
  if (is_nonlinear_system())
  {
    _residual = output_storage()("residual");
    if (requires_grad())
      _Jacobian = derivative_storage()("residual", "state");
  }

  for (auto submodel : registered_models())
    submodel->setup_nonlinear_system();
}

void
Model::zero()
{
  VariableStore::zero(requires_grad(), requires_2nd_grad());

  for (auto submodel : registered_models())
    submodel->zero();
}

void
Model::set_solution(const BatchTensor & x)
{
  NonlinearSystem::set_solution(x);

  // Also update the model input variables
  LabeledVector sol(x, {&output_axis().subaxis("residual")});
  host<VariableStore>()->input_storage().slice("state").fill(sol);
}

void
Model::cache(TorchShapeRef batch_shape,
             int deriv_order,
             const torch::Device & device,
             const torch::Dtype & dtype)
{
  _batch_sizes = batch_shape.empty() ? TorchShape{1} : batch_shape.vec();
  VariableStore::cache(_batch_sizes);

  _deriv_order = std::max(deriv_order + _extra_deriv_order, _deriv_order);

  _options = default_tensor_options().device(device).dtype(dtype);

  for (auto submodel : registered_models())
    submodel->cache(batch_shape, _deriv_order, device, dtype);
}

void
Model::check_AD_limitation() const
{
  if (_AD_1st_deriv && !_AD_2nd_deriv)
    throw NEMLException("AD derivative is requested, but AD second derivative is not requested.");
  if (_AD_1st_deriv || _AD_2nd_deriv)
    neml_assert(!_inference_mode, "Inference mode does not support AD");
}

void
Model::input_requires_grad_(bool req)
{
  for (auto && [name, var] : input_views())
    var.requires_grad_(req);
}

void
Model::use_AD_derivatives(bool first, bool second)
{
  _AD_1st_deriv = first;
  _AD_2nd_deriv = second;
  check_AD_limitation();
}

void
Model::set_input(const LabeledVector & in)
{
  neml_assert_batch_broadcastable(in, input_storage());
  neml_assert_dbg(in.axis(0) == input_axis(),
                  "Incompatible input axis. The model has input axis: \n",
                  input_axis(),
                  "The input vector has axis: \n",
                  in.axis(0));

  input_storage().copy_(in.tensor().batch_expand(batch_sizes()).clone());
}

LabeledVector
Model::get_output()
{
  return output_storage().clone();
}

LabeledMatrix
Model::get_doutput_dinput()
{
  return derivative_storage().clone();
}

LabeledTensor3D
Model::get_d2output_dinput2()
{
  return second_derivative_storage().clone();
}

LabeledVector
Model::value(const LabeledVector & in)
{
  set_input(in);
  prepare();

  value();
  return get_output();
}

std::tuple<LabeledVector, LabeledMatrix>
Model::value_and_dvalue(const LabeledVector & in)
{
  set_input(in);
  prepare();

  value_and_dvalue();
  return {get_output(), get_doutput_dinput()};
}

std::tuple<LabeledVector, LabeledMatrix, LabeledTensor3D>
Model::value_and_dvalue_and_d2value(const LabeledVector & in)
{
  set_input(in);
  prepare();

  value_and_dvalue_and_d2value();
  return {get_output(), get_doutput_dinput(), get_d2output_dinput2()};
}

void
Model::value()
{
  check_inplace_dbg();
  {
    c10::InferenceMode guard(_inference_mode);
    set_value(true, false, false);
  }
}

void
Model::value_and_dvalue()
{
  neml_assert_dbg(requires_grad(),
                  "value_and_dvalue() is called but derivative storage hasn't been allocated.");

  check_inplace_dbg();

  if (!_AD_1st_deriv)
  {
    c10::InferenceMode guard(_inference_mode);
    set_value(true, true, false);
  }
  else
  {
    input_requires_grad_();
    set_value(true, false, false);
    extract_derivatives(/*retain_graph=*/true, /*create_graph=*/false, /*allow_unused=*/true);
  }
}

void
Model::value_and_dvalue_and_d2value()
{
  neml_assert_dbg(requires_2nd_grad(),
                  "value_and_dvalue_and_d2value() is called but second derivative storage hasn't "
                  "been allocated.");

  check_inplace_dbg();

  if (!_AD_2nd_deriv)
  {
    c10::InferenceMode guard(_inference_mode);
    set_value(true, true, true);
  }
  else
  {
    input_requires_grad_();

    if (!_AD_1st_deriv)
      set_value(true, true, false);
    else
    {
      set_value(true, false, false);
      extract_derivatives(/*retain_graph=*/true, /*create_graph=*/true, /*allow_unused=*/true);
    }

    extract_second_derivatives(
        /*retain_graph=*/true, /*create_graph=*/false, /*allow_unused=*/true);
  }
}

void
Model::check_inplace_dbg()
{
#ifndef NDEBUG
  neml_assert_dbg(_inference_mode || !_evaluated_once,
                  "During the non-inference mode forward pass, model '",
                  name(),
                  "' is being evaluated a second time");
  _evaluated_once = true;
#endif
}

Model *
Model::registered_model(const std::string & name) const
{
  for (auto submodel : _registered_models)
    if (submodel->name() == name)
      return submodel;

  throw NEMLException("There is no registered model named '" + name + "' in '" + this->name() +
                      "'");
}

const std::set<VariableName>
Model::consumed_items() const
{
  return input_axis().variable_accessors(true);
}

const std::set<VariableName>
Model::provided_items() const
{
  return output_axis().variable_accessors(true);
}

void
Model::assemble(bool residual, bool Jacobian)
{
  prepare();

  if (residual && !Jacobian)
    value();
  else if (Jacobian)
    value_and_dvalue();
}

void
Model::extract_derivatives(bool retain_graph, bool create_graph, bool allow_unused)
{
  // Loop over rows to retrieve the derivatives
  if (output_storage().tensor().requires_grad())
    for (TorchSize i = 0; i < output_storage().base_sizes()[0]; i++)
    {
      auto grad_outputs = BatchTensor::zeros_like(output_storage());
      grad_outputs.index_put_({torch::indexing::Ellipsis, i}, 1.0);
      for (auto && [name, var] : input_views())
      {
        auto dyi_dvar = torch::autograd::grad({output_storage()},
                                              {var.tensor()},
                                              {grad_outputs},
                                              retain_graph,
                                              create_graph,
                                              allow_unused)[0];

        if (dyi_dvar.defined())
        {
          derivative_storage().base_index_put(
              {i, input_axis().indices(name)},
              dyi_dvar.reshape(utils::add_shapes(batch_sizes(), var.base_storage())));
        }
      }
    }
}

void
Model::extract_second_derivatives(bool retain_graph, bool create_graph, bool allow_unused)
{
  // Loop over rows to retrieve the second derivatives
  if (derivative_storage().tensor().requires_grad())
    for (TorchSize i = 0; i < derivative_storage().base_sizes()[0]; i++)
      for (TorchSize j = 0; j < derivative_storage().base_sizes()[1]; j++)
      {
        auto grad_outputs = torch::zeros_like(derivative_storage());
        grad_outputs.index_put_({torch::indexing::Ellipsis, i, j}, 1.0);
        for (auto && [name, var] : input_views())
        {
          auto dydxij_dvar = torch::autograd::grad({derivative_storage()},
                                                   {var.tensor()},
                                                   {grad_outputs},
                                                   retain_graph,
                                                   create_graph,
                                                   allow_unused)[0];
          if (dydxij_dvar.defined())
            second_derivative_storage().base_index_put(
                {i, j, input_axis().indices(name)},
                dydxij_dvar.reshape(utils::add_shapes(batch_sizes(), var.base_storage())));
        }
      }
}
} // namespace neml2
