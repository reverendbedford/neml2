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
  options.set<std::vector<std::string>>("control") = {
      "strain", "stress", "stress", "stress", "stress", "stress"};

  options.set<VariableName>("state_name") = VariableName("mixed_state");

  options.set<CrossRef<SR2>>("fixed_values");

  options.set<VariableName>("cauchy_stress") = VariableName("state", "S");
  options.set<VariableName>("strain") = VariableName("forces", "E");
  return options;
}

MixedControlSetup::MixedControlSetup(const OptionSet & options)
  : Model(options),
    _state_name(options.get<VariableName>("state_name")),
    _control_types(options.get<std::vector<std::string>>("control")),
    _fixed_values(declare_parameter<SR2>("fixed_values", "fixed_values")),
    _mixed_state(declare_input_variable<SR2>(_state_name.on("forces"))),
    _mixed_state_old(declare_input_variable<SR2>(_state_name.on("old_forces"))),
    _stress(declare_output_variable<SR2>("cauchy_stress")),
    _strain(declare_output_variable<SR2>("strain"))
{
  neml_assert(_control_types.size() == 6,
              "control must be a vector of length six, one for each component of stress/strain!");
  for (size_t i = 0; i < 6; i++)
    neml_assert(_control_types[i] == "stress" || _control_types[i] == "strain",
                "the entries of control must either be 'stress' or 'strain'");
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
      }
      else
      {
        ((SR2)_strain).base_index_put({i}, ((SR2)_fixed_values).base_index({i}));
        ((SR2)_stress).base_index_put({i}, ((SR2)_mixed_state).base_index({i}));
      }
    }
  }

  if (dout_din)
  {
    for (int i = 0; i < 6; i++)
    {
      if (_control_types[i] == "stress")
      {
        // derivative_storage().base_index_put({output_axis().indices(_strain.name())[i],
        //                                      input_axis().indices(_mixed_state.name())[i]},
        //                                     Scalar::ones(_strain.tensor().options()));
        //((BatchTensor)(_stress.d(_fixed_values))).base_index_put({i, i}, 1.0);
      }
      else
      {
        //((BatchTensorBase)_strain.d(_fixed_values)).base_index_put({i, i}, 1.0);
        //((BatchTensorBase)_stress.d(_mixed_state)).base_index_put({i, i}, 1.0);
      }
    }
  }

  // All zero
  (void)(d2out_din2);
}

} // namespace neml2
