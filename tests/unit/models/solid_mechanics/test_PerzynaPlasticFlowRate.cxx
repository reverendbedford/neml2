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
#include "neml2/models/solid_mechanics/PerzynaPlasticFlowRate.h"

using namespace neml2;

TEST_CASE("PerzynaPlasticFlowRate", "[PerzynaPlasticFlowRate]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        PerzynaPlasticFlowRate::expected_params() +
                            ParameterSet(KS{"name", "eprate"},
                                         KS{"type", "PerzynaPlasticFlowRate"},
                                         KR{"eta", 150},
                                         KR{"n", 6}));

  auto & eprate = Factory::get_object<PerzynaPlasticFlowRate>("Models", "eprate");

  SECTION("model definition")
  {
    REQUIRE(eprate.input().has_subaxis("state"));
    REQUIRE(eprate.input().subaxis("state").has_variable<Scalar>("yield_function"));
    REQUIRE(eprate.output().has_subaxis("state"));
    REQUIRE(eprate.output().subaxis("state").has_variable<Scalar>("hardening_rate"));
  }

  SECTION("model derivatives")
  {
    TorchSize nbatch = 10;
    LabeledVector in(nbatch, eprate.input());
    in.slice("state").set(Scalar(1.3, nbatch), "yield_function");

    auto exact = eprate.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, eprate.output(), eprate.input());
    finite_differencing_derivative(
        [eprate](const LabeledVector & x) { return eprate.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
