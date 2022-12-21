#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "neml2/models/solid_mechanics/J2StressMeasure.h"
#include "neml2/models/solid_mechanics/YieldFunction.h"
#include "neml2/models/solid_mechanics/IsotropicHardeningYieldFunction.h"
#include "neml2/models/solid_mechanics/KinematicHardeningYieldFunction.h"
#include "neml2/models/solid_mechanics/IsotropicAndKinematicHardeningYieldFunction.h"
#include "neml2/models/solid_mechanics/PerfectlyPlasticYieldFunction.h"

using namespace neml2;

TEST_CASE("J2IsotropicYieldFunction", "[J2IsotropicYieldFunction]")
{
  TorchSize nbatch = 10;
  Scalar s0 = 50.0;
  auto sm = std::make_shared<J2StressMeasure>("stress_measure");
  auto yield = IsotropicHardeningYieldFunction("yield_function", sm, s0);

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
  TorchSize nbatch = 10;
  Scalar s0 = 50.0;
  auto sm = std::make_shared<J2StressMeasure>("stress_measure");
  auto yield = PerfectlyPlasticYieldFunction("yield_function", sm, s0);

  SECTION("model definition")
  {
    REQUIRE(yield.input().has_subaxis("state"));
    REQUIRE(yield.input().subaxis("state").has_variable<SymR2>("mandel_stress"));
    REQUIRE(yield.output().has_subaxis("state"));
    REQUIRE(yield.output().subaxis("state").has_variable<Scalar>("yield_function"));
  }

  SECTION("model derivatives")
  {
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
  TorchSize nbatch = 10;
  Scalar s0 = 50.0;
  auto sm = std::make_shared<J2StressMeasure>("stress_measure");
  auto yield = KinematicHardeningYieldFunction("yield_function", sm, s0);

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
  TorchSize nbatch = 10;
  Scalar s0 = 50.0;
  auto sm = std::make_shared<J2StressMeasure>("stress_measure");
  auto yield = IsotropicAndKinematicHardeningYieldFunction("yield_function", sm, s0);

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
