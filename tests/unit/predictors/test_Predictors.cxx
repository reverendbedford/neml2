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

#include <catch2/catch.hpp>

#include "utils.h"
#include "neml2/models/ImplicitUpdate.h"

using namespace neml2;

TEST_CASE("Predictors")
{
  SECTION("PreviousStatePredictor") { load_model("unit/predictors/test_PreviousStatePredictor.i"); }
  SECTION("LinearExtrapolationPredictor")
  {
    load_model("unit/predictors/test_LinearExtrapolationPredictor.i");
  }

  auto & model = Factory::get_object<ImplicitUpdate>("Models", "model");

  TorchSize nbatch = 1;
  LabeledVector in(nbatch, {&model.input()});
  in.set(Scalar(15.0), LabeledAxisAccessor{{"forces", "temperature"}});
  in.set(Scalar(1.3), LabeledAxisAccessor{{"forces", "t"}});
  in.set(Scalar(1.1), LabeledAxisAccessor{{"old_forces", "t"}});

  // Take one step
  auto out = model.value(in);
  LabeledVector(in.slice("old_state")).fill(out.slice("state"));
  LabeledVector(in.slice("old_forces")).fill(in.slice("forces"));

  // Take next step
  model.advance_step();
  in.set(Scalar(20.0), LabeledAxisAccessor{{"forces", "temperature"}});
  in.set(Scalar(1.5), LabeledAxisAccessor{{"forces", "t"}});
  out = model.value(in);
  LabeledVector(in.slice("old_state")).fill(out.slice("state"));
  LabeledVector(in.slice("old_forces")).fill(in.slice("forces"));

  // Take another step
  model.advance_step();
  in.set(Scalar(25.0), LabeledAxisAccessor{{"forces", "temperature"}});
  in.set(Scalar(1.6), LabeledAxisAccessor{{"forces", "t"}});
  out = model.value(in);
}
