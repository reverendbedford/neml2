#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "models/solid_mechanics/ElasticStrain.h"

using namespace neml2;

TEST_CASE("ElasticStrain", "[ElasticStrain]")
{
  TorchSize nbatch = 10;
  auto estrain = ElasticStrain("elastic_strain");

  SECTION("model definition")
  {
    REQUIRE(estrain.input().has_subaxis("state"));
    REQUIRE(estrain.input().subaxis("state").has_variable<SymR2>("plastic_strain"));
    REQUIRE(estrain.input().has_subaxis("forces"));
    REQUIRE(estrain.input().subaxis("forces").has_variable<SymR2>("total_strain"));
    REQUIRE(estrain.output().has_subaxis("state"));
    REQUIRE(estrain.output().subaxis("state").has_variable<SymR2>("elastic_strain"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, estrain.input());
    auto E = SymR2::init(0.1, 0.05, 0).batch_expand(nbatch);
    auto Ep = SymR2::init(0.01, 0.01, 0).batch_expand(nbatch);
    in.slice(0, "forces").set(E, "total_strain");
    in.slice(0, "state").set(Ep, "plastic_strain");

    auto exact = estrain.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, estrain.output(), estrain.input());
    finite_differencing_derivative(
        [estrain](const LabeledVector & x) { return estrain.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
