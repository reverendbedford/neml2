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

#include "neml2/models/solid_mechanics/VoceIsotropicHardening.h"

namespace neml2
{
register_NEML2_object(VoceIsotropicHardening);

ParameterSet
VoceIsotropicHardening::expected_params()
{
  ParameterSet params = IsotropicHardening::expected_params();
  params.set<Real>("saturated_hardening");
  params.set<Real>("saturation_rate");
  return params;
}

VoceIsotropicHardening::VoceIsotropicHardening(const ParameterSet & params)
  : IsotropicHardening(params),
    _R(register_parameter("saturated_hardening", Scalar(params.get<Real>("saturated_hardening")))),
    _d(register_parameter("saturation_rate", Scalar(params.get<Real>("saturation_rate"))))
{
}

void
VoceIsotropicHardening::set_value(LabeledVector in,
                                  LabeledVector * out,
                                  LabeledMatrix * dout_din,
                                  LabeledTensor3D * d2out_din2) const
{
  auto ep = in.get<Scalar>(equivalent_plastic_strain);

  if (out)
    out->set(_R * (1.0 - exp(-_d * ep)), isotropic_hardening);

  if (dout_din)
    dout_din->set(_R * _d * exp(-_d * ep), isotropic_hardening, equivalent_plastic_strain);

  if (d2out_din2)
    d2out_din2->set(-_R * _d * _d * exp(-_d * ep),
                    isotropic_hardening,
                    equivalent_plastic_strain,
                    equivalent_plastic_strain);
}
} // namespace neml2
