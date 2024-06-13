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

#include "neml2/models/solid_mechanics/OverStress.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(OverStress);

OptionSet
OverStress::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Calculate the over stress \\f$ \\boldsymbol{O} = \\boldsymbol{M} - "
                  "\\boldsymbol{X} \\f$, where \\f$ \\boldsymbol{M} \\f$ is the Mandel stress and "
                  "\\f$ \\boldsymbol{X} \\f$ is the back stress.";

  options.set<VariableName>("mandel_stress") = VariableName("state", "internal", "M");
  options.set("mandel_stress").doc() = "Mandel stress";

  options.set<VariableName>("back_stress") = VariableName("state", "internal", "X");
  options.set("back_stress").doc() = "Back stress";

  options.set<VariableName>("over_stress") = VariableName("state", "internal", "O");
  options.set("over_stress").doc() = "Over stress";

  return options;
}

OverStress::OverStress(const OptionSet & options)
  : Model(options),
    _M(declare_input_variable<SR2>("mandel_stress")),
    _X(declare_input_variable<SR2>("back_stress")),
    _O(declare_output_variable<SR2>("over_stress"))
{
}

void
OverStress::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _O = _M - _X;

  if (dout_din)
  {
    auto I = SR2::identity_map(options());
    _O.d(_M) = I;
    _O.d(_X) = -I;
  }

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
