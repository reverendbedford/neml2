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

#include "neml2/models/solid_mechanics/YieldFunction.h"

namespace neml2
{
register_NEML2_object(YieldFunction);

ParameterSet
YieldFunction::expected_params()
{
  ParameterSet params = Model::expected_params();
  params.set<Real>("yield_stress");
  params.set<LabeledAxisAccessor>("stress_measure") = {{"state", "internal", "sm"}};
  params.set<LabeledAxisAccessor>("isotropic_hardening");
  params.set<LabeledAxisAccessor>("yield_function") = {{"state", "internal", "fp"}};
  return params;
}

YieldFunction::YieldFunction(const ParameterSet & params)
  : Model(params),
    stress_measure(
        declare_input_variable<Scalar>(params.get<LabeledAxisAccessor>("stress_measure"))),
    isotropic_hardening(params.get<LabeledAxisAccessor>("isotropic_hardening")),
    yield_function(
        declare_output_variable<Scalar>(params.get<LabeledAxisAccessor>("yield_function"))),
    _s0(register_parameter(
        "sy", Scalar(params.get<Real>("yield_stress"), default_tensor_options), false))
{
  if (!isotropic_hardening.empty())
    declare_input_variable<Scalar>(isotropic_hardening);
  setup();
}

void
YieldFunction::set_value(const LabeledVector & in,
                         LabeledVector * out,
                         LabeledMatrix * dout_din,
                         LabeledTensor3D * d2out_din2) const
{
  if (out)
  {
    auto sm = in.get<Scalar>(stress_measure);
    auto f = sm - _s0;
    if (!isotropic_hardening.empty())
      f -= in.get<Scalar>(isotropic_hardening);
    out->set(std::sqrt(2.0 / 3.0) * f, yield_function);
  }

  if (dout_din)
  {
    auto I = Scalar::identity_map(in.options());
    dout_din->set(std::sqrt(2.0 / 3.0) * I, yield_function, stress_measure);
    if (!isotropic_hardening.empty())
      dout_din->set(-std::sqrt(2.0 / 3.0) * I, yield_function, isotropic_hardening);
  }

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
