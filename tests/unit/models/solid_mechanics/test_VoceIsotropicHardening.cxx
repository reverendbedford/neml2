#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "neml2/models/solid_mechanics/VoceIsotropicHardening.h"

using namespace neml2;

TEST_CASE("VoceIsotropicHardening", "[VoceIsotropicHardening]")
{
  TorchSize nbatch = 10;
  Scalar R = 100.0;
  Scalar d = 1.1;
  auto isoharden = VoceIsotropicHardening("hardening", R, d);

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
    LabeledVector in(nbatch, isoharden.input());
    in.slice("state").slice("internal_state").set(Scalar(0.1, nbatch), "equivalent_plastic_strain");

    auto exact = isoharden.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, isoharden.output(), isoharden.input());
    finite_differencing_derivative(
        [isoharden](const LabeledVector & x) { return isoharden.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
