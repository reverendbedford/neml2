#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "models/solid_mechanics/J2IsotropicYieldFunction.h"
#include "models/solid_mechanics/AssociativeIsotropicPlasticHardening.h"

using namespace neml2;

TEST_CASE("AssociativeIsotropicPlasticHardening", "[AssociativeIsotropicPlasticHardening]")
{
  TorchSize nbatch = 10;
  Scalar s0 = 100.0;
  auto yield = std::make_shared<J2IsotropicYieldFunction>("yield_function", s0);
  auto eprate = AssociativeIsotropicPlasticHardening("ep_rate", yield);

  SECTION("model definition")
  {
    // My input should be sufficient for me to evaluate the yield function, hence
    REQUIRE(eprate.input().has_subaxis("state"));
    REQUIRE(eprate.input().subaxis("state").has_variable<Scalar>("hardening_rate"));
    REQUIRE(eprate.input().subaxis("state").has_variable<SymR2>("mandel_stress"));
    REQUIRE(eprate.input().subaxis("state").has_subaxis("hardening_interface"));
    REQUIRE(eprate.input().subaxis("state").subaxis("hardening_interface").has_variable<Scalar>("isotropic_hardening"));

    REQUIRE(eprate.output().has_subaxis("state"));
    REQUIRE(
        eprate.output().subaxis("state").subaxis("internal_state").has_variable<Scalar>("equivalent_plastic_strain_rate"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, eprate.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(Scalar(0.01, nbatch), "hardening_rate");
    in.slice("state").slice("hardening_interface").set(Scalar(200, nbatch), "isotropic_hardening");
    in.slice("state").set(M, "mandel_stress");

    auto exact = eprate.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, eprate.output(), eprate.input());
    finite_differencing_derivative(
        [eprate](const LabeledVector & x) { return eprate.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
