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

#include "neml2/predictors/LinearExtrapolationPredictor.h"

namespace neml2
{
register_NEML2_object(LinearExtrapolationPredictor);

ParameterSet
LinearExtrapolationPredictor::expected_params()
{
  ParameterSet params = Predictor::expected_params();
  params.set<LabeledAxisAccessor>("time") = LabeledAxisAccessor{{"t"}};
  return params;
}

LinearExtrapolationPredictor::LinearExtrapolationPredictor(const ParameterSet & params)
  : Predictor(params),
    _time_name(params.get<LabeledAxisAccessor>("time").on("forces")),
    _old_time_name(params.get<LabeledAxisAccessor>("time").on("old_forces"))
{
}

void
LinearExtrapolationPredictor::set_initial_guess(const LabeledVector & in,
                                                const LabeledVector & guess) const
{
  // At the first step there's nothing we can do
  if (_state_n.axes().empty())
    return;

  auto dt = in.get<Scalar>(_time_name) - in.get<Scalar>(_old_time_name);

  // At the second step we can extrapolate from the first step and "zero"
  if (_state_nm1.axes().empty())
  {
    LabeledVector state(_state_n.tensor() + _state_n.tensor() / _dt_n * dt, _state_n.axes());
    guess.slice("state").fill(state);
  }
  // Later on we can use the old state and the older state to do extrapolation
  else
  {
    LabeledVector state(_state_n.tensor() + (_state_n.tensor() - _state_nm1.tensor()) / _dt_n * dt,
                        _state_n.axes());
    guess.slice("state").fill(state);
  }
}

void
LinearExtrapolationPredictor::post_solve(const LabeledVector & in, const LabeledVector & out)
{
  _state = out.slice("state").clone();
  _dt = in.get<Scalar>(_time_name) - in.get<Scalar>(_old_time_name);
}

void
LinearExtrapolationPredictor::advance_step()
{
  if (!_state_n.axes().empty())
    _state_nm1 = _state_n.clone();

  if (!_state.axes().empty())
    _state_n = _state.clone();

  _state = LabeledVector();

  _dt_n = _dt.clone();
}
} // namespace neml2
