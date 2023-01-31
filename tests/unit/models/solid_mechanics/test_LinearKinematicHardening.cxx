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
#include "neml2/models/solid_mechanics/LinearKinematicHardening.h"

using namespace neml2;

TEST_CASE("LinearKinematicHardening", "[LinearKinematicHardening]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        LinearKinematicHardening::expected_params() +
                            ParameterSet(KS{"name", "kinharden"},
                                         KS{"type", "LinearKinematicHardening"},
                                         KR{"H", 1000}));

  auto & kinharden = Factory::get_object<LinearKinematicHardening>("Models", "kinharden");

  SECTION("model definition")
  {
    REQUIRE(kinharden.input().has_subaxis("state"));
    REQUIRE(kinharden.input().subaxis("state").has_subaxis("internal_state"));
    REQUIRE(kinharden.input()
                .subaxis("state")
                .subaxis("internal_state")
                .has_variable<SymR2>("plastic_strain"));
    REQUIRE(kinharden.output().has_subaxis("state"));
    REQUIRE(kinharden.output().subaxis("state").has_subaxis("hardening_interface"));
    REQUIRE(kinharden.output()
                .subaxis("state")
                .subaxis("hardening_interface")
                .has_variable<SymR2>("kinematic_hardening"));
  }

  SECTION("model derivatives")
  {
    TorchSize nbatch = 10;
    LabeledVector in(nbatch, kinharden.input());
    auto ep = SymR2::init(0.05, -0.01, 0.02, 0.04, 0.03, -0.06).batch_expand(nbatch);
    in.slice("state").slice("internal_state").set(ep, "plastic_strain");

    auto exact = kinharden.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, kinharden.output(), kinharden.input());
    finite_differencing_derivative(
        [kinharden](const LabeledVector & x) { return kinharden.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
