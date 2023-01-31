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
#include "neml2/models/solid_mechanics/PlasticStrainRate.h"

using namespace neml2;

TEST_CASE("PlasticStrainRate", "[PlasticStrainRate]")
{
  TorchSize nbatch = 10;
  auto pstrainrate = PlasticStrainRate("plastic_strain_rate");

  SECTION("model definition")
  {
    REQUIRE(pstrainrate.input().has_subaxis("state"));
    REQUIRE(pstrainrate.input().subaxis("state").has_variable<SymR2>("plastic_flow_direction"));
    REQUIRE(pstrainrate.input().subaxis("state").has_variable<Scalar>("hardening_rate"));

    REQUIRE(pstrainrate.output().has_subaxis("state"));
    REQUIRE(pstrainrate.output().subaxis("state").has_variable<SymR2>("plastic_strain_rate"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, pstrainrate.input());
    // Well, in principle we should give Np:Np = 3/2, but whatever
    auto Np = SymR2::init(0, 0, 0, 1, 0.3, 0.8).batch_expand(nbatch);
    in.slice("state").set(Np, "plastic_flow_direction");
    in.slice("state").set(Scalar(0.01, nbatch), "hardening_rate");

    auto exact = pstrainrate.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, pstrainrate.output(), pstrainrate.input());
    finite_differencing_derivative(
        [pstrainrate](const LabeledVector & x) { return pstrainrate.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
