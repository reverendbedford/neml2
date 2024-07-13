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

#include "ModelUnitTest.h"
#include "utils.h"
#include "neml2/misc/math.h"

#include <torch/cuda.h>

namespace neml2
{
register_NEML2_object(ModelUnitTest);

OptionSet
ModelUnitTest::expected_options()
{
  OptionSet options = Driver::expected_options();
  options.set<std::string>("model");
  options.set<TensorShape>("batch_shape") = {1};
  options.set<bool>("check_values") = true;
  options.set<bool>("check_first_derivatives") = true;
  options.set<bool>("check_second_derivatives") = false;
  options.set<bool>("check_AD_first_derivatives") = true;
  options.set<bool>("check_AD_second_derivatives") = true;
  options.set<bool>("check_AD_derivatives") = true;
  options.set<bool>("check_parameter_derivatives") = false;
  options.set<bool>("check_cuda") = true;
  options.set<bool>("check_inference") = true;
  options.set<std::vector<VariableName>>("input_batch_tensor_names");
  options.set<std::vector<CrossRef<BatchTensor>>>("input_batch_tensor_values");
  options.set<std::vector<VariableName>>("output_batch_tensor_names");
  options.set<std::vector<CrossRef<BatchTensor>>>("output_batch_tensor_values");
  options.set<std::vector<VariableName>>("input_scalar_names");
  options.set<std::vector<CrossRef<Scalar>>>("input_scalar_values");
  options.set<std::vector<VariableName>>("input_symr2_names");
  options.set<std::vector<CrossRef<SR2>>>("input_symr2_values");
  options.set<std::vector<VariableName>>("input_skewr2_names");
  options.set<std::vector<CrossRef<WR2>>>("input_skewr2_values");
  options.set<std::vector<VariableName>>("output_scalar_names");
  options.set<std::vector<CrossRef<Scalar>>>("output_scalar_values");
  options.set<std::vector<VariableName>>("output_symr2_names");
  options.set<std::vector<CrossRef<SR2>>>("output_symr2_values");
  options.set<std::vector<VariableName>>("output_skewr2_names");
  options.set<std::vector<CrossRef<WR2>>>("output_skewr2_values");
  options.set<std::vector<VariableName>>("input_rot_names");
  options.set<std::vector<CrossRef<Rot>>>("input_rot_values");
  options.set<std::vector<VariableName>>("output_rot_names");
  options.set<std::vector<CrossRef<Rot>>>("output_rot_values");
  options.set<Real>("output_rel_tol") = 1e-5;
  options.set<Real>("output_abs_tol") = 1e-8;
  options.set<Real>("derivatives_rel_tol") = 1e-5;
  options.set<Real>("derivatives_abs_tol") = 1e-8;
  options.set<Real>("second_derivatives_rel_tol") = 1e-5;
  options.set<Real>("second_derivatives_abs_tol") = 1e-8;
  options.set<Real>("parameter_derivatives_rel_tol") = 1e-5;
  options.set<Real>("parameter_derivatives_abs_tol") = 1e-8;
  return options;
}

ModelUnitTest::ModelUnitTest(const OptionSet & options)
  : Driver(options),
    _model(get_model(options.get<std::string>("model"), false)),
    _model_inference(get_model(options.get<std::string>("model"), true)),
    _batch_shape(options.get<TensorShape>("batch_shape")),
    _check_values(options.get<bool>("check_values")),
    _check_1st_deriv(options.get<bool>("check_first_derivatives")),
    _check_2nd_deriv(options.get<bool>("check_second_derivatives")),
    _check_AD_1st_deriv(options.get<bool>("check_AD_first_derivatives")),
    _check_AD_2nd_deriv(options.get<bool>("check_AD_second_derivatives")),
    _check_AD_derivs(options.get<bool>("check_AD_derivatives")),
    _check_param_derivs(options.get<bool>("check_parameter_derivatives")),
    _check_cuda(options.get<bool>("check_cuda")),
    _check_inference(options.get<bool>("check_inference")),
    _deriv_order(-1),
    _out_rtol(options.get<Real>("output_rel_tol")),
    _out_atol(options.get<Real>("output_abs_tol")),
    _deriv_rtol(options.get<Real>("derivatives_rel_tol")),
    _deriv_atol(options.get<Real>("derivatives_abs_tol")),
    _secderiv_rtol(options.get<Real>("second_derivatives_rel_tol")),
    _secderiv_atol(options.get<Real>("second_derivatives_abs_tol")),
    _param_rtol(options.get<Real>("parameter_derivatives_rel_tol")),
    _param_atol(options.get<Real>("parameter_derivatives_abs_tol"))
{
  _in = LabeledVector::zeros(_batch_shape, {&_model.input_axis()});
  fill_vector<BatchTensor>(_in, "input_batch_tensor_names", "input_batch_tensor_values");
  fill_vector<Scalar>(_in, "input_scalar_names", "input_scalar_values");
  fill_vector<SR2>(_in, "input_symr2_names", "input_symr2_values");
  fill_vector<WR2>(_in, "input_skewr2_names", "input_skewr2_values");
  fill_vector<Rot>(_in, "input_rot_names", "input_rot_values");

  _out = LabeledVector::zeros(_batch_shape, {&_model.output_axis()});
  fill_vector<BatchTensor>(_out, "output_batch_tensor_names", "output_batch_tensor_values");
  fill_vector<Scalar>(_out, "output_scalar_names", "output_scalar_values");
  fill_vector<SR2>(_out, "output_symr2_names", "output_symr2_values");
  fill_vector<WR2>(_out, "output_skewr2_names", "output_skewr2_values");
  fill_vector<Rot>(_out, "output_rot_names", "output_rot_values");

  if (_check_2nd_deriv || _check_AD_2nd_deriv || _check_AD_derivs)
    _deriv_order = 2;
  else if (_check_1st_deriv)
    _deriv_order = 1;
  else
    _deriv_order = 0;
}

bool
ModelUnitTest::run()
{
  if (!run(_model))
    return false;

  if (_check_inference)
    if (!run(_model_inference))
      return false;

  return true;
}

bool
ModelUnitTest::run(Model & model)
{
  model.reinit(_in, _deriv_order);
  check_all(model);

  if (_check_cuda && torch::cuda::is_available())
  {
    _in = _in.to(torch::kCUDA);
    model.reinit(_in, _deriv_order);
    check_all(model);
  }

  return true;
}

void
ModelUnitTest::check_all(Model & model)
{
  if (_check_values)
    check_values(model);

  if (_check_1st_deriv)
    check_derivatives(model, false, false);

  if (_check_2nd_deriv)
    check_second_derivatives(model, false, false);

  // AD is not supported in inference mode
  if (!model.inference_mode())
  {
    if (_check_AD_1st_deriv)
      check_derivatives(model, true, true);

    if (_check_AD_2nd_deriv)
      check_second_derivatives(model, false, true);

    if (_check_AD_derivs)
      check_second_derivatives(model, true, true);

    if (_check_param_derivs)
      check_parameter_derivatives(model);
  }
}

void
ModelUnitTest::check_values(Model & model)
{
  auto out = model.value(_in);
  neml_assert(utils::allclose(out, _out.to(out.options()), _out_rtol, _out_atol),
              "The model gives values that are different from expected. The expected labels are:\n",
              _out.axis(0),
              "\nThe model gives the following labels:\n",
              out.axis(0),
              "\nThe expected values are:\n",
              _out.tensor(),
              "\nThe model gives the following values:\n",
              out.tensor());
}

void
ModelUnitTest::check_derivatives(Model & model, bool first, bool second)
{
  model.use_AD_derivatives(first, second);
  auto exact = std::get<1>(model.value_and_dvalue(_in));
  auto numerical = finite_differencing_derivative(
      [this, &model](const BatchTensor & x) { return model.value(LabeledVector(x, _in.axes())); },
      _in);
  neml_assert(torch::allclose(exact, numerical, _deriv_rtol, _deriv_atol),
              "The model gives derivatives that are different from those given by finite "
              "differencing. The model gives:\n",
              exact.tensor(),
              "\nFinite differencing gives:\n",
              numerical);
}

void
ModelUnitTest::check_second_derivatives(Model & model, bool first, bool second)
{
  model.use_AD_derivatives(first, second);
  auto exact = std::get<2>(model.value_and_dvalue_and_d2value(_in));
  auto numerical = finite_differencing_derivative(
      [this, &model](const BatchTensor & x)
      { return std::get<1>(model.value_and_dvalue(LabeledVector(x, _in.axes()))); },
      _in);
  neml_assert(torch::allclose(exact, numerical, _secderiv_rtol, _secderiv_atol),
              "The model gives second derivatives that are different from those given by finite "
              "differencing. The model gives:\n",
              exact.tensor(),
              "\nFinite differencing gives:\n",
              numerical);
}

void
ModelUnitTest::check_parameter_derivatives(Model & model)
{
  for (auto && [name, param] : model.named_parameters())
  {
    auto pval = BatchTensor(param).batch_expand_copy(_batch_shape);
    pval.requires_grad_(true);
    param.set(pval);
    auto out = model.value(_in);
    auto exact = math::jacrev(out, BatchTensor(param));
    auto numerical = finite_differencing_derivative(
        [&, &param = param](const BatchTensor & x)
        {
          param.set(x);
          return model.value(_in);
        },
        BatchTensor(param));
    neml_assert(torch::allclose(exact, numerical, _param_rtol, _param_atol),
                "The model gives derivatives for parameter '",
                name,
                "' that are different from those given by finite "
                "differencing. The model gives:\n",
                exact,
                "\nFinite differencing gives:\n",
                numerical);
  }
}
}
