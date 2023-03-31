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
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
register_NEML2_object(OverStress);

ParameterSet
OverStress::expected_params()
{
  ParameterSet params = Model::expected_params();
  params.set<LabeledAxisAccessor>("mandel_stress") = {{"state", "internal", "M"}};
  params.set<LabeledAxisAccessor>("back_stress") = {{"state", "internal", "X"}};
  params.set<LabeledAxisAccessor>("over_stress") = {{"state", "internal", "O"}};
  return params;
}

OverStress::OverStress(const ParameterSet & params)
  : Model(params),
    mandel_stress(declare_input_variable<SymR2>(params.get<LabeledAxisAccessor>("mandel_stress"))),
    back_stress(declare_input_variable<SymR2>(params.get<LabeledAxisAccessor>("back_stress"))),
    over_stress(declare_output_variable<SymR2>(params.get<LabeledAxisAccessor>("over_stress")))
{
  setup();
}

void
OverStress::set_value(LabeledVector in,
                      LabeledVector * out,
                      LabeledMatrix * dout_din,
                      LabeledTensor3D * d2out_din2) const
{
  if (out)
  {
    auto M = in.get<SymR2>(mandel_stress);
    auto X = in.get<SymR2>(back_stress);
    out->set(M - X, over_stress);
  }

  if (dout_din)
  {
    auto I = SymR2::identity_map(in.options());
    dout_din->set(I, over_stress, mandel_stress);
    dout_din->set(-I, over_stress, back_stress);
  }

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
