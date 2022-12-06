#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"

TEST_CASE("LinearIsotropicHardening", "[LinearIsotropicHardening]")
{
  TorchSize nbatch = 10;
  Scalar s0 = 100.0;
  Scalar K = 1000.0;
  auto isoharden = LinearIsotropicHardening("hardening", s0, K);

  SECTION("model definition")
  {
    REQUIRE(isoharden.input().has_subaxis("state"));
    REQUIRE(isoharden.input().subaxis("state").has_variable<Scalar>("equivalent_plastic_strain"));
    REQUIRE(isoharden.output().has_subaxis("state"));
    REQUIRE(isoharden.output().subaxis("state").has_variable<Scalar>("isotropic_hardening"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, isoharden.input());
    in.slice(0, "state").set(Scalar(0.1, nbatch), "equivalent_plastic_strain");

    auto exact = isoharden.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, isoharden.output(), isoharden.input());
    finite_differencing_derivative(
        [isoharden](const LabeledVector & x) { return isoharden.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
