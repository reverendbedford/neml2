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
#include "neml2/models/solid_mechanics/VoceIsotropicHardening.h"

using namespace neml2;

TEST_CASE("VoceIsotropicHardening", "[VoceIsotropicHardening]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        VoceIsotropicHardening::expected_params() +
                            ParameterSet(KS{"name", "isoharden"},
                                         KS{"type", "VoceIsotropicHardening"},
                                         KR{"saturated_hardening", 100},
                                         KR{"saturation_rate", 1.1}));

  auto & isoharden = Factory::get_object<VoceIsotropicHardening>("Models", "isoharden");

  SECTION("model definition")
  {
    REQUIRE(isoharden.input().has_subaxis("state"));
    REQUIRE(isoharden.input().subaxis("state").has_subaxis("internal_state"));
    REQUIRE(isoharden.input()
                .subaxis("state")
                .subaxis("internal_state")
                .has_variable<Scalar>("equivalent_plastic_strain"));
    REQUIRE(isoharden.output().has_subaxis("state"));
    REQUIRE(isoharden.output().subaxis("state").has_subaxis("hardening_interface"));
    REQUIRE(isoharden.output()
                .subaxis("state")
                .subaxis("hardening_interface")
                .has_variable<Scalar>("isotropic_hardening"));
  }

  SECTION("model derivatives")
  {
    TorchSize nbatch = 10;
    LabeledVector in(nbatch, isoharden.input());
    in.slice("state").slice("internal_state").set(Scalar(0.1, nbatch), "equivalent_plastic_strain");

    auto exact = isoharden.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, isoharden.output(), isoharden.input());
    finite_differencing_derivative(
        [isoharden](const LabeledVector & x) { return isoharden.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
