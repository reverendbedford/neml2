#include <catch2/catch.hpp>

#include "neml2/base/HITParser.h"
#include "neml2/models/SumModel.h"

using namespace neml2;

TEST_CASE("manufacture", "[Factory]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  ParameterCollection all_params;
  all_params["Models"]["example"] =
      ScalarSumModel::expected_params() +
      ParameterSet(KS{"name", "example"},
                   KS{"type", "ScalarSumModel"},
                   KVVS{"from_var", {{"state", "A"}, {"state", "substate", "B"}}},
                   KVS{"to_var", {"state", "outsub", "C"}});

  factory.manufacture(all_params);
  auto & summodel = Factory::get_object<ScalarSumModel>("Models", "example");

  SECTION("model definition")
  {
    REQUIRE(summodel.input().has_subaxis("state"));
    REQUIRE(summodel.input().subaxis("state").has_subaxis("substate"));
    REQUIRE(summodel.input().subaxis("state").has_variable<Scalar>("A"));
    REQUIRE(summodel.input().subaxis("state").subaxis("substate").has_variable<Scalar>("B"));

    REQUIRE(summodel.output().has_subaxis("state"));
    REQUIRE(summodel.output().subaxis("state").has_subaxis("outsub"));
    REQUIRE(summodel.output().subaxis("state").subaxis("outsub").has_variable<Scalar>("C"));
  }
}
