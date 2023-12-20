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

#include "neml2/models/Model.h"

namespace neml2
{
/// Implicit exponential time integration for rotations
// This function takes as input a skew tensor giving the spin and does the rotational update
class WR2ImplicitExponentialTimeIntegration : public Model
{
public:
  static OptionSet expected_options();

  WR2ImplicitExponentialTimeIntegration(const OptionSet & options);

private:
  /// name of the input variable
  const LabeledAxisAccessor _var_name;
  /// name of the rate variable
  const LabeledAxisAccessor _var_rate_name;

public:
  /// Output: nonlinear resijdual
  const LabeledAxisAccessor res;
  /// Input: current variable spin rate
  const LabeledAxisAccessor var_rate;
  /// Input: current guess at next value
  const LabeledAxisAccessor var;
  /// Input: previous value
  const LabeledAxisAccessor var_n;
  /// Input: next time
  const LabeledAxisAccessor time;
  /// Input: previous time
  const LabeledAxisAccessor time_n;

protected:
  /// Perform the update by defining the nonlinear residual and it's derivatives
  virtual void set_value(const LabeledVector & in,
                         LabeledVector * out,
                         LabeledMatrix * dout_din = nullptr,
                         LabeledTensor3D * d2out_din2 = nullptr) const override;
};
} // namespace neml2
