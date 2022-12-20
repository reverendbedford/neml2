#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "models/solid_mechanics/J2StressMeasure.h"
#include "models/solid_mechanics/YieldFunction.h"
#include "models/solid_mechanics/ChabochePlasticHardening.h"

using namespace neml2;

TEST_CASE("Chaboche model definition", "[Chaboche]")
{
  TorchSize nbatch = 10;
  Scalar s0 = 25.0;
  auto sm = std::make_shared<J2StressMeasure>("stress_measure");
  auto yield = std::make_shared<YieldFunction>("yield_function", sm, s0, false, true);
  Scalar C = 1000.0;
  Scalar g = 10.0;
  Scalar A = 1.0e-6;
  Scalar a = 2.1;
  auto chaboche = ChabochePlasticHardening("backstress_1", C, g, A, a, yield);

  SECTION("model definition")
  {
    // My input should be sufficient for me to evaluate the yield function, hence
    REQUIRE(chaboche.input().has_subaxis("state"));
    REQUIRE(chaboche.input().subaxis("state").has_variable<Scalar>("hardening_rate"));
    REQUIRE(chaboche.input().subaxis("state").has_variable<SymR2>("mandel_stress"));
    REQUIRE(chaboche.input().subaxis("state").has_subaxis("hardening_interface"));
    REQUIRE(chaboche.input().subaxis("state").subaxis("hardening_interface").has_variable<SymR2>("kinematic_hardening"));
    REQUIRE(chaboche.input().subaxis("state").has_subaxis("internal_state"));
    REQUIRE(chaboche.input().subaxis("state").subaxis("internal_state").has_variable<SymR2>("backstress"));

    REQUIRE(chaboche.output().has_subaxis("state"));
    REQUIRE(
        chaboche.output().subaxis("state").subaxis("internal_state").has_variable<SymR2>("backstress_rate"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, chaboche.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    auto X = SymR2::init(-25,30,10,-15,20,25).batch_expand(nbatch);
    auto Xi = SymR2::init(-10, 15, 5, -7, 15, 20).batch_expand(nbatch);
    in.slice("state").set(Scalar(0.01, nbatch), "hardening_rate");
    in.slice("state").slice("hardening_interface").set(X, "kinematic_hardening");
    in.slice("state").set(M, "mandel_stress");
    in.slice("state").slice("internal_state").set(Xi, "backstress");

    auto exact = chaboche.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, chaboche.output(), chaboche.input());
    finite_differencing_derivative(
        [chaboche](const LabeledVector & x) { return chaboche.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor(), 2e-5));
  }
}
