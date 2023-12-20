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

#include "neml2/models/solid_mechanics/crystal_plasticity/SlipStrengthMap.h"

namespace neml2
{
/// A map between the internal variables and the slip strength, for the case where all systems share the same strength
class SingleSlipStrengthMap : public SlipStrengthMap
{
public:
  static OptionSet expected_options();

  /// Setup from options
  SingleSlipStrengthMap(const OptionSet & options);

protected:
  /// Set the slip system strengths and associated derivatives
  virtual void set_value(const LabeledVector & in,
                         LabeledVector * out,
                         LabeledMatrix * dout_din = nullptr,
                         LabeledTensor3D * d2out_din2 = nullptr) const override;

  /// Input: value of the slip hardening variable
  const LabeledAxisAccessor slip_hardening;

  /// Constant strength to offset the slip hardening
  const Scalar & _tau_const;
};
} // namespace neml2
