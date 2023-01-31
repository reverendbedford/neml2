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
#include "neml2/models/solid_mechanics/J2StressMeasure.h"

using namespace neml2;

TEST_CASE("J2StressMeasure", "[J2StressMeasure]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        J2StressMeasure::expected_params() +
                            ParameterSet(KS{"name", "j2"}, KS{"type", "J2StressMeasure"}));

  auto & sm = Factory::get_object<J2StressMeasure>("Models", "j2");

  SECTION("model definition")
  {
    REQUIRE(sm.input().has_subaxis("state"));
    REQUIRE(sm.input().subaxis("state").has_variable<SymR2>("overstress"));
    REQUIRE(sm.output().has_subaxis("state"));
    REQUIRE(sm.output().subaxis("state").has_variable<Scalar>("stress_measure"));
  }

  SECTION("model derivatives")
  {
    TorchSize nbatch = 10;
    LabeledVector in(nbatch, sm.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(M, "overstress");

    auto exact = sm.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, sm.output(), sm.input());
    finite_differencing_derivative(
        [sm](const LabeledVector & x) { return sm.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));

    auto exactd2 = sm.d2value(in);
    auto numericald2 = LabeledTensor<1, 3>(nbatch, sm.output(), sm.input(), sm.input());
    finite_differencing_derivative(
        [sm](const LabeledVector & x) { return sm.dvalue(x); }, in, numericald2);

    REQUIRE(torch::allclose(exactd2.tensor(), numericald2.tensor()));
  }
}
