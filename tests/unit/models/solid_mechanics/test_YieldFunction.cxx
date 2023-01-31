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

using namespace neml2;

TEST_CASE("J2IsotropicYieldFunction", "[J2IsotropicYieldFunction]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        J2StressMeasure::expected_params() +
                            ParameterSet(KS{"name", "j2"}, KS{"type", "J2StressMeasure"}));
  factory.create_object("Models",
                        YieldFunction::expected_params() +
                            ParameterSet(KS{"name", "yield"},
                                         KS{"type", "YieldFunction"},
                                         KS{"stress_measure", "j2"},
                                         KR{"yield_stress", 50},
                                         KB{"with_isotropic_hardening", true},
                                         KB{"with_kinematic_hardening", false}));

  auto & yield = Factory::get_object<YieldFunction>("Models", "yield");

  SECTION("model definition")
  {
    REQUIRE(yield.input().has_subaxis("state"));
    REQUIRE(yield.input().subaxis("state").has_variable<SymR2>("mandel_stress"));
    REQUIRE(yield.input().has_subaxis("state"));
    REQUIRE(yield.input().subaxis("state").has_subaxis("hardening_interface"));
    REQUIRE(yield.input()
                .subaxis("state")
                .subaxis("hardening_interface")
                .has_variable<Scalar>("isotropic_hardening"));
    REQUIRE(yield.output().has_subaxis("state"));
    REQUIRE(yield.output().subaxis("state").has_variable<Scalar>("yield_function"));
  }

  SECTION("model derivatives")
  {
    TorchSize nbatch = 10;
    LabeledVector in(nbatch, yield.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").slice("hardening_interface").set(Scalar(200, nbatch), "isotropic_hardening");
    in.slice("state").set(M, "mandel_stress");

    auto exact = yield.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, yield.output(), yield.input());
    finite_differencing_derivative(
        [yield](const LabeledVector & x) { return yield.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));

    auto exactd2 = yield.d2value(in);
    auto numericald2 = LabeledTensor<1, 3>(nbatch, yield.output(), yield.input(), yield.input());
    finite_differencing_derivative(
        [yield](const LabeledVector & x) { return yield.dvalue(x); }, in, numericald2);

    REQUIRE(torch::allclose(exactd2.tensor(), numericald2.tensor()));
  }
}

TEST_CASE("J2PerfectYieldFunction", "[J2PerfectYieldFunction]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        J2StressMeasure::expected_params() +
                            ParameterSet(KS{"name", "j2"}, KS{"type", "J2StressMeasure"}));
  factory.create_object("Models",
                        YieldFunction::expected_params() +
                            ParameterSet(KS{"name", "yield"},
                                         KS{"type", "YieldFunction"},
                                         KS{"stress_measure", "j2"},
                                         KR{"yield_stress", 50},
                                         KB{"with_isotropic_hardening", false},
                                         KB{"with_kinematic_hardening", false}));

  auto & yield = Factory::get_object<YieldFunction>("Models", "yield");

  SECTION("model definition")
  {
    REQUIRE(yield.input().has_subaxis("state"));
    REQUIRE(yield.input().subaxis("state").has_variable<SymR2>("mandel_stress"));
    REQUIRE(yield.output().has_subaxis("state"));
    REQUIRE(yield.output().subaxis("state").has_variable<Scalar>("yield_function"));
  }

  SECTION("model derivatives")
  {
    TorchSize nbatch = 10;
    LabeledVector in(nbatch, yield.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(M, "mandel_stress");

    auto exact = yield.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, yield.output(), yield.input());
    finite_differencing_derivative(
        [yield](const LabeledVector & x) { return yield.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));

    auto exactd2 = yield.d2value(in);
    auto numericald2 = LabeledTensor<1, 3>(nbatch, yield.output(), yield.input(), yield.input());
    finite_differencing_derivative(
        [yield](const LabeledVector & x) { return yield.dvalue(x); }, in, numericald2);

    REQUIRE(torch::allclose(exactd2.tensor(), numericald2.tensor()));
  }
}

TEST_CASE("J2KinematicYieldFunction", "[J2KinematicYieldFunction]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        J2StressMeasure::expected_params() +
                            ParameterSet(KS{"name", "j2"}, KS{"type", "J2StressMeasure"}));
  factory.create_object("Models",
                        YieldFunction::expected_params() +
                            ParameterSet(KS{"name", "yield"},
                                         KS{"type", "YieldFunction"},
                                         KS{"stress_measure", "j2"},
                                         KR{"yield_stress", 50},
                                         KB{"with_isotropic_hardening", false},
                                         KB{"with_kinematic_hardening", true}));

  auto & yield = Factory::get_object<YieldFunction>("Models", "yield");

  SECTION("model definition")
  {
    REQUIRE(yield.input().has_subaxis("state"));
    REQUIRE(yield.input().subaxis("state").has_variable<SymR2>("mandel_stress"));
    REQUIRE(yield.input().has_subaxis("state"));
    REQUIRE(yield.input().subaxis("state").has_subaxis("hardening_interface"));
    REQUIRE(yield.input()
                .subaxis("state")
                .subaxis("hardening_interface")
                .has_variable<SymR2>("kinematic_hardening"));
    REQUIRE(yield.output().has_subaxis("state"));
    REQUIRE(yield.output().subaxis("state").has_variable<Scalar>("yield_function"));
  }

  SECTION("model derivatives")
  {
    TorchSize nbatch = 10;
    LabeledVector in(nbatch, yield.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(M, "mandel_stress");
    auto X = SymR2::init(50, -25, 10, 20, 30, 40).batch_expand(nbatch);
    in.slice("state").slice("hardening_interface").set(X, "kinematic_hardening");

    auto exact = yield.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, yield.output(), yield.input());
    finite_differencing_derivative(
        [yield](const LabeledVector & x) { return yield.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor(), 2e-5));

    auto exactd2 = yield.d2value(in);
    auto numericald2 = LabeledTensor<1, 3>(nbatch, yield.output(), yield.input(), yield.input());
    finite_differencing_derivative(
        [yield](const LabeledVector & x) { return yield.dvalue(x); }, in, numericald2);

    REQUIRE(torch::allclose(exactd2.tensor(), numericald2.tensor()));
  }
}

TEST_CASE("J2IsotropicAndKinematicYieldFunction", "[J2IsotropicAndKinematicYieldFunction]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        J2StressMeasure::expected_params() +
                            ParameterSet(KS{"name", "j2"}, KS{"type", "J2StressMeasure"}));
  factory.create_object("Models",
                        YieldFunction::expected_params() +
                            ParameterSet(KS{"name", "yield"},
                                         KS{"type", "YieldFunction"},
                                         KS{"stress_measure", "j2"},
                                         KR{"yield_stress", 50},
                                         KB{"with_isotropic_hardening", true},
                                         KB{"with_kinematic_hardening", true}));

  auto & yield = Factory::get_object<YieldFunction>("Models", "yield");

  SECTION("model definition")
  {
    REQUIRE(yield.input().has_subaxis("state"));
    REQUIRE(yield.input().subaxis("state").has_variable<SymR2>("mandel_stress"));
    REQUIRE(yield.input().has_subaxis("state"));
    REQUIRE(yield.input().subaxis("state").has_subaxis("hardening_interface"));
    REQUIRE(yield.input()
                .subaxis("state")
                .subaxis("hardening_interface")
                .has_variable<Scalar>("isotropic_hardening"));
    REQUIRE(yield.input()
                .subaxis("state")
                .subaxis("hardening_interface")
                .has_variable<SymR2>("kinematic_hardening"));
    REQUIRE(yield.output().has_subaxis("state"));
    REQUIRE(yield.output().subaxis("state").has_variable<Scalar>("yield_function"));
  }

  SECTION("model derivatives")
  {
    TorchSize nbatch = 10;
    LabeledVector in(nbatch, yield.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(M, "mandel_stress");
    auto X = SymR2::init(50, -25, 10, 20, 30, 40).batch_expand(nbatch);
    in.slice("state").slice("hardening_interface").set(Scalar(50, nbatch), "isotropic_hardening");
    in.slice("state").slice("hardening_interface").set(X, "kinematic_hardening");

    auto exact = yield.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, yield.output(), yield.input());
    finite_differencing_derivative(
        [yield](const LabeledVector & x) { return yield.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor(), 2e-5));

    auto exactd2 = yield.d2value(in);
    auto numericald2 = LabeledTensor<1, 3>(nbatch, yield.output(), yield.input(), yield.input());
    finite_differencing_derivative(
        [yield](const LabeledVector & x) { return yield.dvalue(x); }, in, numericald2);

    REQUIRE(torch::allclose(exactd2.tensor(), numericald2.tensor()));
  }
}
