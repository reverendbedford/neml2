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

#include "neml2/models/solid_mechanics/KocksMeckingFlowViscosity.h"

namespace neml2
{
register_NEML2_object(KocksMeckingFlowViscosity);

OptionSet
KocksMeckingFlowViscosity::expected_options()
{
  OptionSet options = NonlinearParameter<Scalar>::expected_options();
  options.set<CrossRef<Scalar>>("A");
  options.set<CrossRef<Scalar>>("B");
  options.set<CrossRef<Scalar>>("shear_modulus");

  options.set<Real>("eps0");
  options.set<Real>("k");
  options.set<Real>("b");

  options.set<VariableName>("temperature") = VariableName("forces", "T");
  return options;
}

KocksMeckingFlowViscosity::KocksMeckingFlowViscosity(const OptionSet & options)
  : NonlinearParameter<Scalar>(options),
    _A(declare_parameter<Scalar>("A", "A")),
    _B(declare_parameter<Scalar>("B", "B")),
    _mu(declare_parameter<Scalar>("shear_modulus", "shear_modulus")),
    _eps0(options.get<Real>("eps0")),
    _k(options.get<Real>("k")),
    _b3(options.get<Real>("b") * options.get<Real>("b") * options.get<Real>("b")),
    _T(declare_input_variable<Scalar>("temperature"))
{
}

void
KocksMeckingFlowViscosity::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto post = math::pow(_eps0, _k * _T * _A / (_mu * _b3));

  if (out)
    _p = math::exp(_B) * _mu * post;

  if (dout_din)
  {
    _p.d(_T) = _A * math::exp(_B) * _k * std::log(_eps0) * post / _b3;

    if (const auto A = nl_param("A"))
      _p.d(*A) = math::exp(_B) * _k * _T * std::log(_eps0) * post / _b3;

    if (const auto B = nl_param("B"))
      _p.d(*B) = math::exp(_B) * _mu * post;

    if (const auto mu = nl_param("mu"))
      _p.d(*mu) = math::exp(_B) * post * (_b3 * _mu - _A * _k * _T * std::log(_eps0)) / (_b3 * _mu);
  }

  if (d2out_din2)
  {
    // T
    _p.d(_T, _T) = math::pow(_A * _k * std::log(_eps0) / _b3, 2.0) * math::exp(_B) * post / _mu;

    if (const auto A = nl_param("A"))
      _p.d(_T, *A) = math::exp(_B) * _k * std::log(_eps0) * post * (_A * _k * _T + _b3 * _mu) /
                     (_b3 * _b3 * _mu);

    if (const auto B = nl_param("B"))
      _p.d(_T, *B) = _A * math::exp(_B) * _k * std::log(_eps0) * post / _b3;

    if (const auto mu = nl_param("mu"))
      _p.d(_T, *mu) =
          -math::pow(_A * _k * std::log(_eps0) / (_b3 * _mu), 2.0) * math::exp(_B) * _T * post;

    // A
    if (const auto A = nl_param("A"))
    {
      _p.d(*A, _T) = math::exp(_B) * _k * std::log(_eps0) * post * (_A * _k * _T + _b3 * _mu) /
                     (_b3 * _b3 * _mu);

      _p.d(*A, *A) = math::exp(_B) * math::pow(_k * _T * std::log(_eps0) / _b3, 2.0) * post / _mu;

      if (const auto B = nl_param("B"))
        _p.d(*A, *B) = math::exp(_B) * _k * _T * std::log(_eps0) * post / _b3;

      if (const auto mu = nl_param("mu"))
        _p.d(*A, *mu) =
            -_A * math::exp(_B) * math::pow(_k * _T * std::log(_eps0) / (_b3 * _mu), 2.0) * post;
    }

    // B
    if (const auto B = nl_param("B"))
    {
      _p.d(*B, _T) = _A * math::exp(_B) * _k * std::log(_eps0) * post / _b3;

      if (const auto A = nl_param("A"))
        _p.d(*B, *A) = math::exp(_B) * _k * _T * std::log(_eps0) * post / _b3;

      _p.d(*B, *B) = math::exp(_B) * _mu * post;

      if (const auto mu = nl_param("mu"))
        _p.d(*B, *mu) =
            math::exp(_B) * post * (_b3 * _mu - _A * _k * _T * std::log(_eps0)) / (_b3 * _mu);
    }

    // mu
    if (const auto mu = nl_param("mu"))
    {
      _p.d(*mu, _T) =
          -math::exp(_B) * math::pow(_A * _k * std::log(_eps0) / (_b3 * _mu), 2.0) * _T * post;

      if (const auto A = nl_param("A)"))
        _p.d(*mu, *A) =
            -_A * math::exp(_B) * math::pow(_k * _T * std::log(_eps0) / (_b3 * _mu), 2.0) * post;

      if (const auto B = nl_param("B"))
        _p.d(*mu, *B) =
            math::exp(_B) * post * (_b3 * _mu - _A * _k * _T * std::log(_eps0)) / (_b3 * _mu);

      _p.d(*mu, *mu) = -math::pow(_A * _k * _T * std::log(_eps0) / (_b3 * _mu), 2.0) *
                       math::exp(_B) * post / _mu;
    }
  }
}
} // namespace neml2
