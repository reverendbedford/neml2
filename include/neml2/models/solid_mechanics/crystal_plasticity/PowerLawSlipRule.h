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

#pragma once

#include "neml2/models/solid_mechanics/crystal_plasticity/SlipRule.h"

namespace neml2
{
/// Power law slip rate of the form \f$\dot{\gamma}_i = \dot{\gamma}_0 \left| \frac{\tau_i}{\tau_{h,i}} \right| ^ {n-1} \frac{\tau_i}{\tau_{h,i}} \f$
class PowerLawSlipRule : public SlipRule
{
public:
  static OptionSet expected_options();

  PowerLawSlipRule(const OptionSet & options);

protected:
  /// Set the slip rates and derivatives
  virtual void set_value(const LabeledVector & in,
                         LabeledVector * out,
                         LabeledMatrix * dout_din = nullptr,
                         LabeledTensor3D * d2out_din2 = nullptr) const override;

  /// Reference slip rate
  const Scalar & _gamma0;
  /// Rate sensitivity
  const Scalar & _n;
};
} // namespace neml2
