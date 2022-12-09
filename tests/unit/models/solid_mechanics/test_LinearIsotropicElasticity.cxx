#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "models/solid_mechanics/LinearIsotropicElasticity.h"

using namespace neml2;

TEST_CASE("LinearIsotropicElasticity", "[LinearIsotropicElasticity]")
{
  TorchSize nbatch = 10;
  Scalar E = 100;
  Scalar nu = 0.3;
  auto elasticity = LinearIsotropicElasticity("elasticity", E, nu);

  SECTION("model definition")
  {
    REQUIRE(elasticity.input().has_subaxis("state"));
    REQUIRE(elasticity.input().subaxis("state").has_variable<SymR2>("elastic_strain"));
    REQUIRE(elasticity.output().has_subaxis("state"));
    REQUIRE(elasticity.output().subaxis("state").has_variable<SymR2>("cauchy_stress"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, elasticity.input());
    auto Ee = SymR2::init(0.09, 0.04, 0).expand_batch(nbatch);
    in.slice(0, "state").set(Ee, "elastic_strain");

    auto exact = elasticity.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, elasticity.output(), elasticity.input());
    finite_differencing_derivative(
        [elasticity](const LabeledVector & x) { return elasticity.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
