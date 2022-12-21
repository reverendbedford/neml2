#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "neml2/models/solid_mechanics/IsotropicMandelStress.h"

using namespace neml2;

TEST_CASE("IsotropicMandelStress", "[IsotropicMandelStress]")
{
  TorchSize nbatch = 10;
  auto mandel_stress = IsotropicMandelStress("mandel_stress");

  SECTION("model definition")
  {
    REQUIRE(mandel_stress.input().has_subaxis("state"));
    REQUIRE(mandel_stress.input().subaxis("state").has_variable<SymR2>("cauchy_stress"));
    REQUIRE(mandel_stress.output().has_subaxis("state"));
    REQUIRE(mandel_stress.output().subaxis("state").has_variable<SymR2>("mandel_stress"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, mandel_stress.input());
    auto S = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(S, "cauchy_stress");

    auto exact = mandel_stress.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, mandel_stress.output(), mandel_stress.input());
    finite_differencing_derivative(
        [mandel_stress](const LabeledVector & x) { return mandel_stress.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
