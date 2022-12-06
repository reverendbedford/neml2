#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "models/solid_mechanics/NoKinematicHardening.h"

TEST_CASE("NoKinematicHardening", "[NoKinematicHardening]")
{
  TorchSize nbatch = 10;
  auto kinharden = NoKinematicHardening("kinematic_hardening");

  SECTION("model definition")
  {
    REQUIRE(kinharden.input().has_subaxis("state"));
    REQUIRE(kinharden.input().subaxis("state").has_variable<SymR2>("cauchy_stress"));
    REQUIRE(kinharden.output().has_subaxis("state"));
    REQUIRE(kinharden.output().subaxis("state").has_variable<SymR2>("mandel_stress"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, kinharden.input());
    auto S = SymR2::init(100, 110, 100, 100, 100, 100).expand_batch(nbatch);
    in.slice(0, "state").set(S, "cauchy_stress");

    auto exact = kinharden.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, kinharden.output(), kinharden.input());
    finite_differencing_derivative(
        [kinharden](const LabeledVector & x) { return kinharden.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
