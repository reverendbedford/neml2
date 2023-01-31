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
#include "neml2/models/solid_mechanics/IsotropicMandelStress.h"

using namespace neml2;

TEST_CASE("IsotropicMandelStress", "[IsotropicMandelStress]")
{
  TorchSize nbatch = 10;
  auto mandel_stress = IsotropicMandelStress("mandel_stress");

  SECTION("model definition")
  {
    REQUIRE(mandel_stress.input().has_subaxis("state"));
    REQUIRE(mandel_stress.input().subaxis("state").has_variable<SymR2>("cauchy_stress"));
    REQUIRE(mandel_stress.output().has_subaxis("state"));
    REQUIRE(mandel_stress.output().subaxis("state").has_variable<SymR2>("mandel_stress"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, mandel_stress.input());
    auto S = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(S, "cauchy_stress");

    auto exact = mandel_stress.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, mandel_stress.output(), mandel_stress.input());
    finite_differencing_derivative(
        [mandel_stress](const LabeledVector & x) { return mandel_stress.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
