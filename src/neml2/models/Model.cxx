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

#include "neml2/models/Model.h"
#include <torch/autograd.h>

namespace neml2
{
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
  options.set<bool>("_enable_AD") = true;

  options.set("_use_AD_first_derivative").suppressed() = true;
  options.set("_use_AD_second_derivative").suppressed() = true;
  options.set("_extra_derivative_order").suppressed() = true;
  options.set("_nonlinear_system").suppressed() = true;
  options.set("_enable_AD").suppressed() = true;

  return options;
}

Model::Model(const OptionSet & options)
  : Data(options),
    ParameterStore(options, this),
    VariableStore(options, this),
    NonlinearSystem(options),
    DiagnosticsInterface(this),
    _options(default_tensor_options()),
    _nonlinear_system(options.get<bool>("_nonlinear_system")),
    _deriv_order(-1),
    _extra_deriv_order(options.get<int>("_extra_derivative_order")),
    _enable_AD(options.get<bool>("_enable_AD")),
    _AD_1st_deriv(options.get<bool>("_use_AD_first_derivative")),
    _AD_2nd_deriv(options.get<bool>("_use_AD_second_derivative"))
#ifndef NDEBUG
    ,
    _evaluated_once(false)
#endif
{
}

void
Model::diagnose(std::vector<Diagnosis> & diagnoses) const
{
  for (auto * submodel : registered_models())
    submodel->diagnose(diagnoses);

  // Make sure variables are defined on the reserved subaxes
  for (auto && [name, var] : input_variables())
    diagnostic_check_input_variable(diagnoses, var);
  for (auto && [name, var] : output_variables())
    diagnostic_check_output_variable(diagnoses, var);

  if (is_nonlinear_system())
    diagnose_nl_sys(diagnoses);

  // Check for statefulness
  if (this == host())
    if (input_axis().has_subaxis("old_state"))
      for (auto var : input_axis().subaxis("old_state").variable_names())
        diagnostic_assert(diagnoses,
                          output_axis().has_variable(var.prepend("state")),
                          "Input axis has old state variable ",
                          var,
                          ", but the corresponding output state variable doesn't exist.");
}

void
Model::diagnose_nl_sys(std::vector<Diagnosis> & diagnoses) const
{
  for (auto * submodel : registered_models())
    submodel->diagnose_nl_sys(diagnoses);

  // Check if any input variable is solve-dependent
  bool input_solve_dep = false;
  for (auto && [name, var] : input_variables())
    if (var.is_solve_dependent())
      input_solve_dep = true;

  // If any input variable is solve-dependent, ALL output variables must be solve-dependent!
  if (input_solve_dep)
    for (auto && [name, var] : output_variables())
      diagnostic_assert(
          diagnoses,
          var.is_solve_dependent(),
          "This model is part of a nonlinear system. At least one of the input variables is "
          "solve-dependent, so all output variables MUST be solve-dependent, i.e., they must be on "
          "one of the following sub-axes: state, residual, parameters. However, got output "
          "variable ",
          name);
}

void
Model::setup()
{
  // Setup input and output axes
  setup_layout();
}

std::tuple<const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>
Model::get_system_matrices() const
{
  neml_assert_dbg(is_nonlinear_system(), "This is not a nonlinear system");
  return {_dr_ds, _dr_dsn, _dr_df, _dr_dfn, _dr_dp};
}

void
Model::reinit(const Tensor & tensor, int deriv_order)
{
  reinit(tensor.batch_sizes(), deriv_order, tensor.device(), tensor.scalar_type());
}

void
Model::reinit(TensorShapeRef batch_shape,
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
  setup_output_views();
  setup_nonlinear_system();
  setup_input_views(this);
}

void
Model::prepare()
{
  if (is_AD_disabled())
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

  check_AD_limitation();

  VariableStore::allocate_variables(batch_sizes(),
                                    options(),
                                    /*in=*/in,
                                    /*out=*/out,
                                    /*dout_din=*/out && requires_grad(),
                                    /*d2out_din2=*/out && requires_2nd_grad());

  if (in && is_nonlinear_system())
  {
    _ndof = output_axis().storage_size("residual");
    _solution = Tensor::empty(batch_sizes(), _ndof, options());
  }

  for (auto * submodel : registered_models())
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
  for (auto * submodel : registered_models())
  {
    for (auto && [name, var] : submodel->input_variables())
      var.setup_views(host->input_variable(name));
    submodel->setup_submodel_input_views(submodel);
  }
}

void
Model::setup_output_views()
{
  VariableStore::setup_output_views(true, requires_grad(), requires_2nd_grad());
  setup_submodel_output_views();

  if (is_nonlinear_system() && requires_grad())
  {
    if (input_axis().has_state())
      _dr_ds = derivative_storage().base_index({"residual", "state"});
    if (input_axis().has_old_state())
      _dr_dsn = derivative_storage().base_index({"residual", "old_state"});
    if (input_axis().has_forces())
      _dr_df = derivative_storage().base_index({"residual", "forces"});
    if (input_axis().has_old_forces())
      _dr_dfn = derivative_storage().base_index({"residual", "old_forces"});
    if (input_axis().has_parameters())
      _dr_dp = derivative_storage().base_index({"residual", "parameters"});
  }
}

void
Model::setup_submodel_output_views()
{
  for (auto * submodel : registered_models())
    submodel->setup_output_views();
}

void
Model::setup_nonlinear_system()
{
  if (is_nonlinear_system())
  {
    _residual = output_storage().base_index({"residual"});
    if (requires_grad())
      _Jacobian = derivative_storage().base_index({"residual", "state"});
  }

  for (auto * submodel : registered_models())
    submodel->setup_nonlinear_system();
}

void
Model::zero()
{
  VariableStore::zero(requires_grad(), requires_2nd_grad());

  for (auto * submodel : registered_models())
    submodel->zero();
}

void
Model::set_solution(const Tensor & x)
{
  NonlinearSystem::set_solution(x);

  // Also update the model input variables
  LabeledVector sol(x, {&output_axis().subaxis("residual")});
  host<VariableStore>()->input_storage().slice("state").fill(sol);
}

void
Model::cache(TensorShapeRef batch_shape,
             int deriv_order,
             const torch::Device & device,
             const torch::Dtype & dtype)
{
  _batch_sizes = batch_shape.empty() ? TensorShape{1} : TensorShape(batch_shape);
  VariableStore::cache(_batch_sizes);

  _deriv_order = std::max(deriv_order + _extra_deriv_order, _deriv_order);

  _options = default_tensor_options().device(device).dtype(dtype);

  for (auto * submodel : registered_models())
    submodel->cache(batch_shape, _deriv_order, device, dtype);
}

void
Model::check_AD_limitation() const
{
  if (_AD_1st_deriv && !_AD_2nd_deriv)
    throw NEMLException("AD derivative is requested, but AD second derivative is not requested.");
  if (_AD_1st_deriv || _AD_2nd_deriv)
    neml_assert(is_AD_enabled(), "AD is requested but not enabled");
}

void
Model::input_requires_grad_(bool req)
{
  for (auto && [name, var] : input_variables())
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
Model::check_input(const LabeledVector & in) const
{
  neml_assert(utils::sizes_broadcastable(in.batch_sizes(), batch_sizes()),
              "The provided input has batch shape ",
              in.batch_sizes(),
              " which cannot be broadcast to the model's batch shape ",
              batch_sizes(),
              ". Make sure the model has been initialized using `reinit` and that the provided "
              "input has the correct shape.");
  neml_assert(in.base_storage() == input_storage().base_storage(),
              "The provided input has base storage size ",
              in.base_storage(),
              ", but the model's input storage expects a base storage size of ",
              input_storage().base_storage(),
              ". Make sure the model has been initialized using `reinit` and that the provided "
              "input has the correct shape.");
}

void
Model::set_input(const LabeledVector & in)
{
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
  check_input(in);
  set_input(in);
  prepare();

  value();
  return get_output();
}

std::tuple<LabeledVector, LabeledMatrix>
Model::value_and_dvalue(const LabeledVector & in)
{
  check_input(in);
  set_input(in);
  prepare();

  value_and_dvalue();
  return {get_output(), get_doutput_dinput()};
}

LabeledMatrix
Model::dvalue(const LabeledVector & in)
{
  check_input(in);
  set_input(in);
  prepare();

  dvalue();
  return get_doutput_dinput();
}

std::tuple<LabeledVector, LabeledMatrix, LabeledTensor3D>
Model::value_and_dvalue_and_d2value(const LabeledVector & in)
{
  check_input(in);
  set_input(in);
  prepare();

  value_and_dvalue_and_d2value();
  return {get_output(), get_doutput_dinput(), get_d2output_dinput2()};
}

std::tuple<LabeledMatrix, LabeledTensor3D>
Model::dvalue_and_d2value(const LabeledVector & in)
{
  check_input(in);
  set_input(in);
  prepare();

  dvalue_and_d2value();
  return {get_doutput_dinput(), get_d2output_dinput2()};
}

LabeledTensor3D
Model::d2value(const LabeledVector & in)
{
  check_input(in);
  set_input(in);
  prepare();

  d2value();
  return get_d2output_dinput2();
}

void
Model::value()
{
  check_inplace_dbg();
  {
    c10::InferenceMode guard(is_AD_disabled());
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
    c10::InferenceMode guard(is_AD_disabled());
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
Model::dvalue()
{
  neml_assert_dbg(requires_grad(),
                  "dvalue() is called but derivative storage hasn't been allocated.");

  check_inplace_dbg();

  if (!_AD_1st_deriv)
  {
    c10::InferenceMode guard(is_AD_disabled());
    set_value(false, true, false);
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
    c10::InferenceMode guard(is_AD_disabled());
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
Model::dvalue_and_d2value()
{
  neml_assert_dbg(requires_2nd_grad(),
                  "dvalue_and_d2value() is called but second derivative storage hasn't "
                  "been allocated.");

  check_inplace_dbg();

  if (!_AD_2nd_deriv)
  {
    c10::InferenceMode guard(is_AD_disabled());
    set_value(false, true, true);
  }
  else
  {
    input_requires_grad_();

    if (!_AD_1st_deriv)
      set_value(false, true, false);
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
Model::d2value()
{
  neml_assert_dbg(requires_2nd_grad(),
                  "d2value() is called but second derivative storage hasn't been allocated.");

  check_inplace_dbg();

  if (!_AD_2nd_deriv)
  {
    c10::InferenceMode guard(is_AD_disabled());
    set_value(false, false, true);
  }
  else
  {
    input_requires_grad_();

    if (!_AD_1st_deriv)
      set_value(false, true, false);
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
  neml_assert_dbg(!_enable_AD || !_evaluated_once,
                  "Model '",
                  name(),
                  "' is being evaluated a second time, which could lead to in-place modification "
                  "of function graph. If you do not need to use automatic differentiation, set "
                  "enable_AD = false to avoid this error.");
  _evaluated_once = true;
#endif
}

Model *
Model::registered_model(const std::string & name) const
{
  for (auto * submodel : _registered_models)
    if (submodel->name() == name)
      return submodel;

  throw NEMLException("There is no registered model named '" + name + "' in '" + this->name() +
                      "'");
}

std::set<VariableName>
Model::consumed_items() const
{
  return input_axis().variable_names();
}

std::set<VariableName>
Model::provided_items() const
{
  return output_axis().variable_names();
}

void
Model::assemble(bool residual, bool Jacobian)
{
  prepare();

  if (residual && !Jacobian)
    value();
  else if (residual && Jacobian)
    value_and_dvalue();
  else if (!residual && Jacobian)
    dvalue();
}

void
Model::extract_derivatives(bool retain_graph, bool create_graph, bool allow_unused)
{
  // Loop over rows to retrieve the derivatives
  if (output_storage().tensor().requires_grad())
    for (Size i = 0; i < output_storage().base_sizes()[0]; i++)
    {
      auto grad_outputs = Tensor::zeros_like(output_storage());
      grad_outputs.index_put_({torch::indexing::Ellipsis, i}, 1.0);
      for (auto && [name, var] : input_variables())
      {
        auto dyi_dvar = torch::autograd::grad({output_storage()},
                                              {var.tensor()},
                                              {grad_outputs},
                                              retain_graph,
                                              create_graph,
                                              allow_unused)[0];

        if (dyi_dvar.defined())
        {
          derivative_storage().tensor().base_index_put_(
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
    for (Size i = 0; i < derivative_storage().base_sizes()[0]; i++)
      for (Size j = 0; j < derivative_storage().base_sizes()[1]; j++)
      {
        auto grad_outputs = torch::zeros_like(derivative_storage());
        grad_outputs.index_put_({torch::indexing::Ellipsis, i, j}, 1.0);
        for (auto && [name, var] : input_variables())
        {
          auto dydxij_dvar = torch::autograd::grad({derivative_storage()},
                                                   {var.tensor()},
                                                   {grad_outputs},
                                                   retain_graph,
                                                   create_graph,
                                                   allow_unused)[0];
          if (dydxij_dvar.defined())
            second_derivative_storage().tensor().base_index_put_(
                {i, j, input_axis().indices(name)},
                dydxij_dvar.reshape(utils::add_shapes(batch_sizes(), var.base_storage())));
        }
      }
}
} // namespace neml2
