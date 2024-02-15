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

#include "neml2/models/solid_mechanics/crystal_plasticity/SingleSlipHardeningRule.h"

#include "neml2/tensors/tensors.h"
#include "neml2/tensors/list_tensors.h"

namespace neml2
{
OptionSet
SingleSlipHardeningRule::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<VariableName>("slip_hardening_rate") =
      VariableName("state", "internal", "slip_hardening_rate");
  options.set<VariableName>("slip_hardening") = VariableName("state", "internal", "slip_hardening");
  options.set<VariableName>("sum_slip_rates") = VariableName("state", "internal", "sum_slip_rates");
  return options;
}

SingleSlipHardeningRule::SingleSlipHardeningRule(const OptionSet & options)
  : Model(options),
    _tau_dot(declare_output_variable<Scalar>("slip_hardening_rate")),
    _tau(declare_input_variable<Scalar>("slip_hardening")),
    _gamma_dot_sum(declare_input_variable<Scalar>("sum_slip_rates"))
{
}

} // namespace neml2
