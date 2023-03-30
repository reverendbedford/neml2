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

#include "TestUtils.h"
#include "SampleRateModel.h"
#include "neml2/models/BackwardEulerTimeIntegration.h"

using namespace neml2;

TEST_CASE("BackwardEulerTimeIntegration", "[BackwardEulerTimeIntegration]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        ScalarBackwardEulerTimeIntegration::expected_params() +
                            ParameterSet(KS{"name", "integrate_foo"},
                                         KS{"type", "ScalarBackwardEulerTimeIntegration"},
                                         KVS{"rate_variable", {"foo_rate"}},
                                         KVS{"variable", {"foo"}}));

  auto & model = Factory::get_object<ScalarBackwardEulerTimeIntegration>("Models", "integrate_foo");

  SECTION("model definition")
  {
    REQUIRE(model.input().has_subaxis("state"));
    REQUIRE(model.input().subaxis("state").has_variable<Scalar>("foo_rate"));
    REQUIRE(model.input().subaxis("state").has_variable<Scalar>("foo"));

    REQUIRE(model.input().has_subaxis("old_state"));
    REQUIRE(model.input().subaxis("old_state").has_variable<Scalar>("foo"));

    REQUIRE(model.input().has_subaxis("forces"));
    REQUIRE(model.input().subaxis("forces").has_variable<Scalar>("time"));

    REQUIRE(model.input().has_subaxis("old_forces"));
    REQUIRE(model.input().subaxis("old_forces").has_variable<Scalar>("time"));

    REQUIRE(model.output().has_subaxis("residual"));
    REQUIRE(model.output().subaxis("residual").has_variable<Scalar>("foo"));
  }

  TorchSize nbatch = 10;
  LabeledVector in(nbatch, model.input());
  in.slice("state").set(Scalar(-0.3, nbatch), "foo_rate");
  in.slice("state").set(Scalar(1.1, nbatch), "foo");
  in.slice("old_state").set(Scalar(0, nbatch), "foo");
  in.slice("forces").set(Scalar(1.3, nbatch), "time");
  in.slice("old_forces").set(Scalar(1.1, nbatch), "time");

  SECTION("model derivatives")
  {
    auto exact = model.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, model.output(), model.input());
    finite_differencing_derivative(
        [model](const LabeledVector & x) { return model.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor(), /*rtol=*/0, /*atol=*/5e-4));
  }
}
