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

#pragma once

#include "neml2/drivers/Driver.h"
#include "neml2/models/Model.h"

namespace neml2
{
class ModelUnitTest : public Driver
{
public:
  static OptionSet expected_options();

  ModelUnitTest(const OptionSet & options);

  bool run() override;

  Model & model() { return _model; }
  const Model & model() const { return _model; }

  LabeledVector & in() { return _in; }
  const LabeledVector & in() const { return _in; }

  LabeledVector & out() { return _out; }
  const LabeledVector & out() const { return _out; }

private:
  template <typename T>
  void
  fill_vector(LabeledVector & vec, const std::string & option_vars, const std::string & option_vals)
  {
    const auto vars = input_options().get<std::vector<LabeledAxisAccessor>>(option_vars);
    const auto vals = input_options().get<std::vector<CrossRef<T>>>(option_vals);
    neml_assert(vars.size() == vals.size(),
                "Trying to assign ",
                vals.size(),
                " values to ",
                vars.size(),
                " variables.");
    for (size_t i = 0; i < vars.size(); i++)
      vec.set(T(vals[i]), vars[i]);
  }

  void check_all();
  void check_values();
  void check_derivatives(bool first, bool second);
  void check_second_derivatives(bool first, bool second);

  Model & _model;
  const TorchSize _nbatch;
  const bool _check_1st_deriv;
  const bool _check_2nd_deriv;
  const bool _check_AD_1st_deriv;
  const bool _check_AD_2nd_deriv;
  const bool _check_AD_derivs;
  const bool _check_cuda;

  LabeledVector _in;
  LabeledVector _out;

  Real _out_rtol;
  Real _out_atol;
  Real _deriv_rtol;
  Real _deriv_atol;
  Real _secderiv_rtol;
  Real _secderiv_atol;
};
} // namespace neml2
