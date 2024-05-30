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
  options.set<std::vector<std::string>>("control");

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
    _control_types(options.get<std::vector<std::string>>("control")),
    _fixed_values(declare_input_variable<SR2>(_fixed_values_name.on("forces"))),
    _fixed_values_old(declare_input_variable<SR2>(_fixed_values_name.on("old_forces"))),
    _mixed_state(declare_input_variable<SR2>(_state_name.on("state"))),
    _mixed_state_old(declare_input_variable<SR2>(_state_name.on("old_state"))),
    _stress(declare_output_variable<SR2>("cauchy_stress")),
    _stress_old(declare_output_variable<SR2>("old_cauchy_stress")),
    _strain(declare_output_variable<SR2>("strain")),
    _strain_old(declare_output_variable<SR2>("old_strain"))
{
  neml_assert(_control_types.size() == 6,
              "control must be a vector of length six, one for each component of stress/strain!");
  for (size_t i = 0; i < 6; i++)
    neml_assert(_control_types[i] == "stress" || _control_types[i] == "strain",
                "the entries of control must either be 'stress' or 'strain'");

  // Setup the cached derivatives
  _cached_stress_derivative = SSR4::zeros();
  _cached_strain_derivative = SSR4::zeros();
  for (int i = 0; i < 6; i++)
  {
    if (_control_types[i] == "stress")
      _cached_strain_derivative.base_index_put({i, i}, Scalar::ones());
    else
      _cached_stress_derivative.base_index_put({i, i}, Scalar::ones());
  }
}

void
MixedControlSetup::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
  {
    for (int i = 0; i < 6; i++)
    {
      if (_control_types[i] == "stress")
      {
        ((SR2)_stress).base_index_put({i}, ((SR2)_fixed_values).base_index({i}));
        ((SR2)_strain).base_index_put({i}, ((SR2)_mixed_state).base_index({i}));

        ((SR2)_stress_old).base_index_put({i}, ((SR2)_fixed_values_old).base_index({i}));
        ((SR2)_strain_old).base_index_put({i}, ((SR2)_mixed_state_old).base_index({i}));
      }
      else
      {
        ((SR2)_strain).base_index_put({i}, ((SR2)_fixed_values).base_index({i}));
        ((SR2)_stress).base_index_put({i}, ((SR2)_mixed_state).base_index({i}));

        ((SR2)_strain_old).base_index_put({i}, ((SR2)_fixed_values_old).base_index({i}));
        ((SR2)_stress_old).base_index_put({i}, ((SR2)_mixed_state_old).base_index({i}));
      }
    }
  }

  if (dout_din)
  {
    _stress.d(_mixed_state) = _cached_stress_derivative.to(_fixed_values.tensor().options());
    _strain.d(_mixed_state) = _cached_strain_derivative.to(_fixed_values.tensor().options());
    _stress_old.d(_mixed_state_old) =
        _cached_stress_derivative.to(_fixed_values.tensor().options());
    _strain_old.d(_mixed_state_old) =
        _cached_strain_derivative.to(_fixed_values.tensor().options());

    _stress.d(_fixed_values) = _cached_strain_derivative.to(_fixed_values.tensor().options());
    _strain.d(_fixed_values) = _cached_stress_derivative.to(_fixed_values.tensor().options());
    _stress_old.d(_fixed_values_old) =
        _cached_strain_derivative.to(_fixed_values.tensor().options());
    _strain_old.d(_fixed_values_old) =
        _cached_stress_derivative.to(_fixed_values.tensor().options());
  }

  // All zero
  (void)(d2out_din2);
}

} // namespace neml2
