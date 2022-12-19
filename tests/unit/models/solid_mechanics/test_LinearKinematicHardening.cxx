#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "models/solid_mechanics/LinearKinematicHardening.h"

using namespace neml2;

TEST_CASE("LinearKinematicHardening", "[LinearKinematicHardening]")
{
  TorchSize nbatch = 10;
  Scalar H = 1000.0;
  auto kinharden = LinearKinematicHardening("hardening", H);

  SECTION("model definition")
  {
    REQUIRE(kinharden.input().has_subaxis("state"));
    REQUIRE(kinharden.input().subaxis("state").has_subaxis("internal_state"));
    REQUIRE(kinharden.input().subaxis("state").subaxis("internal_state").has_variable<SymR2>("plastic_strain"));
    REQUIRE(kinharden.output().has_subaxis("state"));
    REQUIRE(kinharden.output().subaxis("state").has_subaxis("hardening_interface"));
    REQUIRE(kinharden.output().subaxis("state").subaxis("hardening_interface").has_variable<SymR2>("kinematic_hardening"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, kinharden.input());
    auto ep = SymR2::init(0.05,-0.01,0.02,0.04,0.03,-0.06).batch_expand(nbatch);
    in.slice("state").slice("internal_state").set(ep, "plastic_strain");

    auto exact = kinharden.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, kinharden.output(), kinharden.input());
    finite_differencing_derivative(
        [kinharden](const LabeledVector & x) { return kinharden.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
