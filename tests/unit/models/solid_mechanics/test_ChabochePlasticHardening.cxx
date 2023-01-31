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
#include "neml2/models/solid_mechanics/YieldFunction.h"
#include "neml2/models/solid_mechanics/ChabochePlasticHardening.h"

using namespace neml2;

TEST_CASE("Chaboche model definition", "[Chaboche]")
{
  TorchSize nbatch = 10;
  Scalar C = 1000.0;
  Scalar g = 10.0;
  Scalar A = 1.0e-6;
  Scalar a = 2.1;
  auto chaboche = ChabochePlasticHardening("backstress_1", C, g, A, a);

  SECTION("model definition")
  {
    // My input should be sufficient for me to evaluate the yield function, hence
    REQUIRE(chaboche.input().has_subaxis("state"));
    REQUIRE(chaboche.input().subaxis("state").has_variable<Scalar>("hardening_rate"));
    REQUIRE(chaboche.input().subaxis("state").has_variable<SymR2>("plastic_flow_direction"));
    REQUIRE(chaboche.input().subaxis("state").has_subaxis("internal_state"));
    REQUIRE(chaboche.input()
                .subaxis("state")
                .subaxis("internal_state")
                .has_variable<SymR2>("backstress"));

    REQUIRE(chaboche.output().has_subaxis("state"));
    REQUIRE(chaboche.output()
                .subaxis("state")
                .subaxis("internal_state")
                .has_variable<SymR2>("backstress_rate"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, chaboche.input());
    SymR2 n = SymR2::init(1.1, 1.2, -1.4, 0.05, 0.5, -0.1).batch_expand(nbatch);
    n = n.dev();
    n /= n.norm();
    auto Xi = SymR2::init(-10, 15, 5, -7, 15, 20).batch_expand(nbatch);
    in.slice("state").set(Scalar(0.01, nbatch), "hardening_rate");
    in.slice("state").set(n, "plastic_flow_direction");
    in.slice("state").slice("internal_state").set(Xi, "backstress");

    auto exact = chaboche.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, chaboche.output(), chaboche.input());
    finite_differencing_derivative(
        [chaboche](const LabeledVector & x) { return chaboche.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor(), 2e-5));
  }
}
