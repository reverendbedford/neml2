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

#include "neml2/models/solid_mechanics/J2StressMeasure.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
J2StressMeasure::J2StressMeasure(const std::string & name)
  : StressMeasure(name)
{
  setup();
}

void
J2StressMeasure::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  // First retrieve the over stress
  auto stress = in.get<SymR2>(overstress);

  // Set the output
  auto s = stress.dev();
  auto J2 = s.norm();
  out.set(J2, stress_measure);

  if (dout_din)
  {
    // Compute the yield function derivative
    auto df_dstress = s / (J2 + EPS);

    // Set the output
    dout_din->set(df_dstress, stress_measure, overstress);
  }
}

void
J2StressMeasure::set_dvalue(LabeledVector in,
                            LabeledMatrix dout_din,
                            LabeledTensor<1, 3> * d2out_din2) const
{
  // First retrieve the Mandel stress
  auto stress = in.get<SymR2>(overstress);

  // Calculate current value...
  auto s = stress.dev();
  auto J2 = s.norm();

  // Compute the yield function derivative
  auto df_dm = s / (J2 + EPS);

  // Set the output
  dout_din.set(df_dm, stress_measure, overstress);

  if (d2out_din2)
  {
    // Compute the yield function second derivative
    auto I = SymSymR4::init(SymSymR4::FillMethod::identity_sym);
    auto J = SymSymR4::init(SymSymR4::FillMethod::identity_dev);
    auto d2f_dstress2 = (I - df_dm.outer(df_dm)) * J / (J2 + EPS);

    // Set the output
    d2out_din2->set(d2f_dstress2, stress_measure, overstress, overstress);
  }
}

}
