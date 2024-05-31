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

#include "neml2/models/solid_mechanics/mixed_control/MixedControlSetup.h"

namespace neml2
{
register_NEML2_object(MixedControlSetup);

OptionSet
MixedControlSetup::expected_options()
{
  OptionSet options = Model::expected_options();

  options.set<VariableName>("control") = VariableName("forces", "control");
  options.set<Real>("threshold") = 0.5;

  options.set<VariableName>("state_name") = VariableName("mixed_state");

  options.set<VariableName>("fixed_values_name") = VariableName("fixed_values");

  options.set<VariableName>("cauchy_stress") = VariableName("state", "S");
  options.set<VariableName>("old_cauchy_stress") = VariableName("old_state", "S");
  options.set<VariableName>("strain") = VariableName("forces", "E");
  options.set<VariableName>("old_strain") = VariableName("old_forces", "E");
  return options;
}

MixedControlSetup::MixedControlSetup(const OptionSet & options)
  : Model(options),
    _state_name(options.get<VariableName>("state_name")),
    _fixed_values_name(options.get<VariableName>("fixed_values_name")),
    _threshold(options.get<Real>("threshold")),
    _control(declare_input_variable<SR2>("control")),
    _fixed_values(declare_input_variable<SR2>(_fixed_values_name.on("forces"))),
    _fixed_values_old(declare_input_variable<SR2>(_fixed_values_name.on("old_forces"))),
    _mixed_state(declare_input_variable<SR2>(_state_name.on("state"))),
    _mixed_state_old(declare_input_variable<SR2>(_state_name.on("old_state"))),
    _stress(declare_output_variable<SR2>("cauchy_stress")),
    _stress_old(declare_output_variable<SR2>("old_cauchy_stress")),
    _strain(declare_output_variable<SR2>("strain")),
    _strain_old(declare_output_variable<SR2>("old_strain"))
{
}

void
MixedControlSetup::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto strain_select = _control.tensor() <= _threshold;
  auto stress_select = _control.tensor() > _threshold;

  // This also converts these to floats
  auto ones_stress =
      BatchTensor(strain_select.to(_stress.tensor().options()), _control.batch_dim());
  auto ones_strain = BatchTensor::ones_like(_control.tensor()) - ones_stress;

  auto dstrain = SSR4(BatchTensor(torch::diag_embed(ones_stress), batch_dim()));
  auto dstress = SSR4(BatchTensor(torch::diag_embed(ones_strain), batch_dim()));

  if (out)
  {
    // Even benign in place operations get errors
    _stress = dstress * _fixed_values + dstrain * _mixed_state;
    _strain = dstrain * _fixed_values + dstress * _mixed_state;

    _stress_old = dstress * _fixed_values_old + dstrain * _mixed_state_old;
    _strain_old = dstrain * _fixed_values_old + dstress * _mixed_state_old;
  }

  if (dout_din)
  {
    _stress.d(_fixed_values) = dstress;
    _stress.d(_mixed_state) = dstrain;

    _strain.d(_fixed_values) = dstrain;
    _strain.d(_mixed_state) = dstress;

    _stress_old.d(_fixed_values_old) = dstress;
    _stress_old.d(_mixed_state_old) = dstrain;

    _strain_old.d(_fixed_values_old) = dstrain;
    _strain_old.d(_mixed_state_old) = dstress;
  }

  // All zero
  (void)(d2out_din2);
}

} // namespace neml2
