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

namespace neml2
{
register_NEML2_object(ModelUnitTest);

ParameterSet
ModelUnitTest::expected_params()
{
  ParameterSet params = Driver::expected_params();
  params.set<std::string>("model");
  params.set<TorchSize>("nbatch") = 1;
  params.set<bool>("check_first_derivatives") = true;
  params.set<bool>("check_second_derivatives") = false;
  params.set<bool>("check_AD_first_derivatives") = true;
  params.set<bool>("check_AD_second_derivatives") = true;
  params.set<bool>("check_AD_derivatives") = true;
  params.set<bool>("check_cuda") = true;
  params.set<std::vector<LabeledAxisAccessor>>("input_scalar_names");
  params.set<std::vector<CrossRef<Scalar>>>("input_scalar_values");
  params.set<std::vector<LabeledAxisAccessor>>("input_symr2_names");
  params.set<std::vector<CrossRef<SymR2>>>("input_symr2_values");
  params.set<std::vector<LabeledAxisAccessor>>("output_scalar_names");
  params.set<std::vector<CrossRef<Scalar>>>("output_scalar_values");
  params.set<std::vector<LabeledAxisAccessor>>("output_symr2_names");
  params.set<std::vector<CrossRef<SymR2>>>("output_symr2_values");
  params.set<Real>("output_rel_tol") = 1e-5;
  params.set<Real>("output_abs_tol") = 1e-8;
  params.set<Real>("derivatives_rel_tol") = 1e-5;
  params.set<Real>("derivatives_abs_tol") = 1e-8;
  params.set<Real>("second_derivatives_rel_tol") = 1e-5;
  params.set<Real>("second_derivatives_abs_tol") = 1e-8;
  return params;
}

ModelUnitTest::ModelUnitTest(const ParameterSet & params)
  : Driver(params),
    _model(Factory::get_object<Model>("Models", params.get<std::string>("model"))),
    _nbatch(params.get<TorchSize>("nbatch")),
    _check_1st_deriv(params.get<bool>("check_first_derivatives")),
    _check_2nd_deriv(params.get<bool>("check_second_derivatives")),
    _check_AD_1st_deriv(params.get<bool>("check_AD_first_derivatives")),
    _check_AD_2nd_deriv(params.get<bool>("check_AD_second_derivatives")),
    _check_AD_derivs(params.get<bool>("check_AD_derivatives")),
    _check_cuda(params.get<bool>("check_cuda")),
    _out_rtol(params.get<Real>("output_rel_tol")),
    _out_atol(params.get<Real>("output_abs_tol")),
    _deriv_rtol(params.get<Real>("derivatives_rel_tol")),
    _deriv_atol(params.get<Real>("derivatives_abs_tol")),
    _secderiv_rtol(params.get<Real>("second_derivatives_rel_tol")),
    _secderiv_atol(params.get<Real>("second_derivatives_abs_tol"))
{
  _in = LabeledVector::zeros(_nbatch, {&_model.input()});
  fill_vector<Scalar>(_in, params, "input_scalar_names", "input_scalar_values");
  fill_vector<SymR2>(_in, params, "input_symr2_names", "input_symr2_values");

  _out = LabeledVector::zeros(_nbatch, {&_model.output()});
  fill_vector<Scalar>(_out, params, "output_scalar_names", "output_scalar_values");
  fill_vector<SymR2>(_out, params, "output_symr2_names", "output_symr2_values");
}

bool
ModelUnitTest::run()
{
  check_all();

  if (_check_cuda && torch::cuda::is_available())
  {
    _model.to(torch::kCUDA);
    check_all();
  }

  return true;
}

void
ModelUnitTest::check_all()
{
  check_values();

  if (_check_1st_deriv)
    check_derivatives(false, false);

  if (_check_2nd_deriv)
    check_second_derivatives(false, false);

  if (_check_AD_1st_deriv)
    check_derivatives(true, true);

  if (_check_AD_2nd_deriv)
    check_second_derivatives(false, true);

  if (_check_AD_derivs)
    check_second_derivatives(true, true);
}

void
ModelUnitTest::check_values()
{
  auto out = _model.value(_in);
  neml_assert(utils::allclose(out, _out, _out_rtol, _out_atol),
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
ModelUnitTest::check_derivatives(bool first, bool second)
{
  _model.use_AD_derivatives(first, second);
  auto exact = _model.dvalue(_in);
  auto numerical = LabeledMatrix::zeros(_nbatch, {&_model.output(), &_model.input()});
  finite_differencing_derivative(
      [this](const LabeledVector & x) { return _model.value(x); }, _in, numerical);
  neml_assert(torch::allclose(exact.tensor(), numerical.tensor(), _deriv_rtol, _deriv_atol),
              "The model gives derivatives that are different from those given by finite "
              "differencing. The model gives:\n",
              exact.tensor(),
              "\nFinite differencing gives:\n",
              numerical.tensor());
}

void
ModelUnitTest::check_second_derivatives(bool first, bool second)
{
  _model.use_AD_derivatives(first, second);
  auto exact = _model.d2value(_in);
  auto numerical =
      LabeledTensor3D::zeros(_nbatch, {&_model.output(), &_model.input(), &_model.input()});
  finite_differencing_derivative(
      [this](const LabeledVector & x) { return _model.dvalue(x); }, _in, numerical);
  neml_assert(torch::allclose(exact.tensor(), numerical.tensor(), _secderiv_rtol, _secderiv_atol),
              "The model gives second derivatives that are different from those given by finite "
              "differencing. The model gives:\n",
              exact.tensor(),
              "\nFinite differencing gives:\n",
              numerical.tensor());
}
}
