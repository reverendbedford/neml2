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

#include "neml2/predictors/Predictor.h"

namespace neml2
{
class LinearExtrapolationPredictor : public Predictor
{
public:
  static ParameterSet expected_params();

  LinearExtrapolationPredictor(const ParameterSet & params);

  /**
   * Linearly extrapolate the old and older state in time to get the initial guess for the current
   * state.
   */
  virtual void set_initial_guess(const LabeledVector & in,
                                 const LabeledVector & guess) const override;

  /**
   * Cache the current state and time step size
   */
  virtual void post_solve(const LabeledVector & in, const LabeledVector & out) override;

  /**
   * Shift the state back in time
   */
  virtual void advance_step() override;

protected:
  const LabeledAxisAccessor _time_name;
  const LabeledAxisAccessor _old_time_name;

  LabeledVector _state;
  LabeledVector _state_n;
  LabeledVector _state_nm1;
  Scalar _dt;
  Scalar _dt_n;
};
} // namespace neml2
