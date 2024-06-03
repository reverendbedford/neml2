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

#include "neml2/models/solid_mechanics/MixedControlSetup.h"

namespace neml2
{
register_NEML2_object(MixedControlSetup);

OptionSet
MixedControlSetup::expected_options()
{
  OptionSet options = Model::expected_options();

  options.set<VariableName>("control") = VariableName("forces", "control");
  options.set<Real>("threshold") = 0.5;

  options.set<VariableName>("mixed_state") = VariableName("state", "mixed_state");
  options.set<VariableName>("fixed_values") = VariableName("forces", "fixed_values");

  options.set<VariableName>("cauchy_stress") = VariableName("state", "S");
  options.set<VariableName>("strain") = VariableName("forces", "E");
  return options;
}

MixedControlSetup::MixedControlSetup(const OptionSet & options)
  : Model(options),
    _threshold(options.get<Real>("threshold")),
    _control(declare_input_variable<SR2>("control")),
    _fixed_values(declare_input_variable<SR2>("fixed_values")),
    _mixed_state(declare_input_variable<SR2>("mixed_state")),
    _stress(declare_output_variable<SR2>("cauchy_stress")),
    _strain(declare_output_variable<SR2>("strain"))
{
}

void
MixedControlSetup::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto [dstrain, dstress] = _make_operators(_control.tensor());

  if (out)
  {
    // Even benign in place operations get errors
    _stress = dstress * _fixed_values + dstrain * _mixed_state;
    _strain = dstrain * _fixed_values + dstress * _mixed_state;
  }

  if (dout_din)
  {
    _stress.d(_fixed_values) = dstress;
    _stress.d(_mixed_state) = dstrain;

    _strain.d(_fixed_values) = dstrain;
    _strain.d(_mixed_state) = dstress;
  }

  // All zero
  (void)(d2out_din2);
}

std::pair<SSR4, SSR4>
MixedControlSetup::_make_operators(const SR2 & control) const
{
  auto strain_select = control <= _threshold;
  auto stress_select = control > _threshold;

  // This also converts these to floats
  auto ones_stress = BatchTensor(strain_select.to(_stress.tensor().options()), control.batch_dim());
  auto ones_strain = BatchTensor::ones_like(control) - ones_stress;

  auto dstrain = SSR4(BatchTensor(torch::diag_embed(ones_stress), batch_dim()));
  auto dstress = SSR4(BatchTensor(torch::diag_embed(ones_strain), batch_dim()));

  return std::make_pair(dstrain, dstress);
}

} // namespace neml2
