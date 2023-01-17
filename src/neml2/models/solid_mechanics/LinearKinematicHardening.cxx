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
LinearKinematicHardening::LinearKinematicHardening(const std::string & name, Scalar H)
  : KinematicHardening(name),
    _H(register_parameter("kinematic_hardening_modulus", H))
{
}

void
LinearKinematicHardening::set_value(LabeledVector in,
                                    LabeledVector out,
                                    LabeledMatrix * dout_din) const
{
  // Map from equivalent plastic strain --> isotropic hardening
  auto g = _H * in.get<SymR2>(plastic_strain);

  // Set the output
  out.set(g, kinematic_hardening);

  if (dout_din)
  {
    // Derivative of the map equivalent plastic strain --> isotropic hardening
    auto dg_dep = _H * SymSymR4::init(SymSymR4::identity_sym).batch_expand(in.batch_size());

    // Set the output
    dout_din->set(dg_dep, kinematic_hardening, plastic_strain);
  }
}
} // namespace neml2
