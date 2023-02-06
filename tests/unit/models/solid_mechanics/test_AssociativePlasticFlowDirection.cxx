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
#include "neml2/models/solid_mechanics/YieldFunction.h"
#include "neml2/models/solid_mechanics/J2StressMeasure.h"
#include "neml2/models/solid_mechanics/AssociativePlasticFlowDirection.h"

using namespace neml2;

TEST_CASE("AssociativePlasticFlowDirection", "[AssociativePlasticFlowDirection]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        J2StressMeasure::expected_params() +
                            ParameterSet(KS{"name", "j2"}, KS{"type", "J2StressMeasure"}));
  factory.create_object("Models",
                        IsotropicHardeningYieldFunction::expected_params() +
                            ParameterSet(KS{"name", "yield"},
                                         KS{"type", "IsotropicHardeningYieldFunction"},
                                         KS{"stress_measure", "j2"},
                                         KR{"yield_stress", 10}));
  factory.create_object("Models",
                        AssociativePlasticFlowDirection::expected_params() +
                            ParameterSet(KS{"name", "direction"},
                                         KS{"type", "AssociativePlasticFlowDirection"},
                                         KS{"yield_function", "yield"}));

  auto & yield = Factory::get_object<IsotropicHardeningYieldFunction>("Models", "yield");
  auto & direction = Factory::get_object<AssociativePlasticFlowDirection>("Models", "direction");

  SECTION("model definition")
  {
    // My input should be sufficient for me to evaluate the yield function, hence
    REQUIRE(direction.input() == yield.input());

    REQUIRE(direction.output().has_subaxis("state"));
    REQUIRE(direction.output().subaxis("state").has_variable<SymR2>("plastic_flow_direction"));
  }

  SECTION("model derivatives")
  {
    TorchSize nbatch = 10;
    LabeledVector in(nbatch, direction.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").slice("hardening_interface").set(Scalar(200, nbatch), "isotropic_hardening");
    in.slice("state").set(M, "mandel_stress");

    auto exact = direction.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, direction.output(), direction.input());
    finite_differencing_derivative(
        [direction](const LabeledVector & x) { return direction.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
