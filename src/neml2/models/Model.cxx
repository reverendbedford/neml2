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
  options.set<bool>("_nonlinear_system") = false;

  options.set("_use_AD_first_derivative").suppressed() = true;
  options.set("_use_AD_second_derivative").suppressed() = true;
  options.set("_nonlinear_system").suppressed() = true;

  return options;
}

Model::Model(const OptionSet & options)
  : Data(options),
    ParameterStore(options, this),
    VariableStore(options, this),
    NonlinearSystem(options),
    DiagnosticsInterface(this),
    _nonlinear_system(options.get<bool>("_nonlinear_system")),
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
  for (auto && [name, var] : variables())
    diagnostic_check_variable(diagnoses, var);

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
  for (auto && [name, var] : variables())
    if (var.ftype() == FType::INPUT && var.is_solve_dependent())
      input_solve_dep = true;

  // If any input variable is solve-dependent, ALL output variables must be solve-dependent!
  if (input_solve_dep)
    for (auto && [name, var] : variables())
      if (var.ftype() == FType::OUTPUT)
        diagnostic_assert(
            diagnoses,
            var.is_solve_dependent(),
            "This model is part of a nonlinear system. At least one of the input variables is "
            "solve-dependent, so all output variables MUST be solve-dependent, i.e., they must be "
            "on one of the following sub-axes: state, residual, parameters. However, got output "
            "variable ",
            name);
}

void
Model::setup()
{
  setup_layout();

  if (host() == this)
  {
    set_args(this);
    link_output_variables();
    link_input_variables();
  }
}

void
Model::set_args(Model * model)
{
  _args = model->variables(FType::INPUT);
  for (auto * submodel : _registered_models)
    submodel->set_args(submodel);
}

void
Model::link_input_variables()
{
  for (auto * submodel : _registered_models)
  {
    link_input_variables(submodel);
    submodel->link_input_variables();
  }
}

void
Model::link_input_variables(Model * submodel)
{
  for (auto && [name, var] : submodel->variables())
    if (var.ftype() == FType::INPUT)
      var.ref(variable(name));
}

void
Model::link_output_variables()
{
  for (auto * submodel : _registered_models)
  {
    link_output_variables(submodel);
    submodel->link_output_variables();
  }
}

void
Model::link_output_variables(Model * /*submodel*/)
{
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
Model::use_AD_derivatives(bool first, bool second)
{
  _AD_1st_deriv = first;
  _AD_2nd_deriv = second;
  check_AD_limitation();
}

void
Model::check_input(const LabeledVector & in) const
{
  neml_assert(in.axis() == input_axis(),
              "The provided input has axis: \n",
              in.axis(),
              ", but the model's input axis is: \n",
              input_axis());
}

void
Model::set_input(const LabeledVector & in)
{
  _options = in.options();
  _batch_sizes = in.batch_sizes();
  for (const auto & [var, val] : in.split())
    variable(var).set(val);
}

LabeledVector
Model::value(const LabeledVector & in)
{
  check_input(in);
  set_input(in);

  prepare();
  value();
  const auto y = get_output();
  finalize();

  return y;
}

std::tuple<LabeledVector, LabeledMatrix>
Model::value_and_dvalue(const LabeledVector & in)
{
  check_input(in);
  set_input(in);

  prepare();
  value_and_dvalue();
  const auto y = get_output();
  const auto dy_dx = get_doutput_dinput();
  finalize();

  return {y, dy_dx};
}

LabeledMatrix
Model::dvalue(const LabeledVector & in)
{
  check_input(in);
  set_input(in);

  prepare();
  dvalue();
  const auto dy_dx = get_doutput_dinput();
  finalize();

  return dy_dx;
}

std::tuple<LabeledVector, LabeledMatrix, LabeledTensor3D>
Model::value_and_dvalue_and_d2value(const LabeledVector & in)
{
  check_input(in);
  set_input(in);

  prepare();
  value_and_dvalue_and_d2value();
  const auto y = get_output();
  const auto dy_dx = get_doutput_dinput();
  const auto d2y_dx2 = get_d2output_dinput2();
  finalize();

  return {y, dy_dx, d2y_dx2};
}

std::tuple<LabeledMatrix, LabeledTensor3D>
Model::dvalue_and_d2value(const LabeledVector & in)
{
  check_input(in);
  set_input(in);

  prepare();
  dvalue_and_d2value();
  const auto dy_dx = get_doutput_dinput();
  const auto d2y_dx2 = get_d2output_dinput2();
  finalize();

  return {dy_dx, d2y_dx2};
}

LabeledTensor3D
Model::d2value(const LabeledVector & in)
{
  check_input(in);
  set_input(in);

  prepare();
  d2value();
  const auto d2y_dx2 = get_d2output_dinput2();
  finalize();

  return d2y_dx2;
}

void
Model::value()
{
  ensure_single_evaluation_dbg();
  set_value(true, false, false);
}

void
Model::value_and_dvalue()
{
  ensure_single_evaluation_dbg();

  if (!_AD_1st_deriv)
    set_value(true, true, false);
  else
  {
    set_value(true, false, false);
    extract_derivatives(/*retain_graph=*/true, /*create_graph=*/false, /*allow_unused=*/true);
  }
}

void
Model::dvalue()
{
  ensure_single_evaluation_dbg();

  if (!_AD_1st_deriv)
    set_value(false, true, false);
  else
  {
    set_value(true, false, false);
    extract_derivatives(/*retain_graph=*/true, /*create_graph=*/false, /*allow_unused=*/true);
  }
}

void
Model::value_and_dvalue_and_d2value()
{
  ensure_single_evaluation_dbg();

  if (!_AD_2nd_deriv)
    set_value(true, true, true);

  else
  {
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
  ensure_single_evaluation_dbg();

  if (!_AD_2nd_deriv)
    set_value(false, true, true);
  else
  {
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
  ensure_single_evaluation_dbg();

  if (!_AD_2nd_deriv)
    set_value(false, false, true);
  else
  {
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
Model::ensure_single_evaluation_dbg()
{
#ifndef NDEBUG
  neml_assert_dbg(!_evaluated_once,
                  "Model '",
                  name(),
                  "' is being evaluated a second time, which indicates a potential issue in our "
                  "dependency resolution. Please consider creating an issue on GitHub.");
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
  if (residual && !Jacobian)
    value();
  else if (residual && Jacobian)
    value_and_dvalue();
  else if (!residual && Jacobian)
    dvalue();
}

LabeledVector
Model::get_output() const
{
  auto vars = output_axis().sort_by_assembly_order(output_axis().variable_names());
  auto vals_flat = std::vector<Tensor>(vars.size());
  for (std::size_t i = 0; i < vars.size(); ++i)
    vals_flat[i] = variable(vars[i]).get().base_flatten();
  return LabeledVector::assemble(vals_flat, output_axis());
}

LabeledMatrix
Model::get_doutput_dinput() const
{
  auto yvars = output_axis().sort_by_assembly_order(output_axis().variable_names());
  auto xvars = input_axis().sort_by_assembly_order(input_axis().variable_names());
  auto vals_flat = std::vector<std::vector<Tensor>>(yvars.size());
  for (std::size_t i = 0; i < yvars.size(); ++i)
  {
    const auto & derivs = variable(yvars[i]).derivatives();
    vals_flat[i].resize(xvars.size());
    for (std::size_t j = 0; j < xvars.size(); ++j)
      vals_flat[i][j] = derivs.at(xvars[j]);
  }
  return LabeledMatrix::assemble(vals_flat, output_axis(), input_axis());
}

LabeledTensor3D
Model::get_d2output_dinput2() const
{
  auto yvars = output_axis().sort_by_assembly_order(output_axis().variable_names());
  auto xvars = input_axis().sort_by_assembly_order(input_axis().variable_names());
  auto vals_flat = std::vector<std::vector<std::vector<Tensor>>>(yvars.size());
  for (std::size_t i = 0; i < yvars.size(); ++i)
  {
    const auto & secderivs = variable(yvars[i]).second_derivatives();
    vals_flat[i].resize(xvars.size());
    for (std::size_t j = 0; j < xvars.size(); ++j)
    {
      vals_flat[i][j].resize(xvars.size());
      for (std::size_t k = 0; k < xvars.size(); ++k)
        vals_flat[i][j][k] = secderivs.at(xvars[j]).at(xvars[k]);
    }
  }
  return LabeledTensor3D::assemble(vals_flat, output_axis(), input_axis(), input_axis());
}

void
Model::prepare()
{
  VariableStore::initialize_derivatives(_args, options());
  _evaluated_once = false;

  for (auto * submodel : _registered_models)
    submodel->prepare();
}

void
Model::finalize()
{
  VariableStore::clear();

  for (auto * submodel : _registered_models)
    submodel->finalize();
}
} // namespace neml2
