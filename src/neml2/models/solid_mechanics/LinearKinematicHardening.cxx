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

#include "neml2/models/solid_mechanics/LinearKinematicHardening.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
register_NEML2_object(LinearKinematicHardening);

ParameterSet
LinearKinematicHardening::expected_params()
{
  ParameterSet params = KinematicHardening::expected_params();
  params.set<Real>("H");
  return params;
}

LinearKinematicHardening::LinearKinematicHardening(const ParameterSet & params)
  : KinematicHardening(params),
    _H(register_parameter("kinematic_hardening_modulus", Scalar(params.get<Real>("H"))))
{
}

void
LinearKinematicHardening::set_value(LabeledVector in,
                                    LabeledVector * out,
                                    LabeledMatrix * dout_din,
                                    LabeledTensor3D * d2out_din2) const
{
  if (out)
    out->set(_H * in(kinematic_plastic_strain), back_stress);

  if (dout_din)
  {
    auto I = SymR2::identity_map(in.options());
    dout_din->set(_H * I, back_stress, kinematic_plastic_strain);
  }

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
