#include <catch2/catch.hpp>

#include "neml2/base/ParameterSet.h"
#include "neml2/models/solid_mechanics/LinearIsotropicHardening.h"

using namespace neml2;

TEST_CASE("get and set", "[ParameterSet]")
{
  ParameterSet params;
  params.set<double>("p1") = 1.5;
  params.set<std::string>("p2") = "foo";
  params.set<unsigned int>("p3") = 3;
  params.set<std::vector<std::string>>("p4") = {"a", "b", "c", "d", "e"};
  params.set<std::vector<double>>("p5") = {1.2, -1.1, 100, 5.3};

  SECTION("size") { REQUIRE(params.size() == 5); }

  SECTION("get")
  {
    REQUIRE(params.get<double>("p1") == Approx(1.5));
    REQUIRE(params.get<std::string>("p2") == "foo");
    REQUIRE(params.get<unsigned int>("p3") == 3);
    REQUIRE_THAT(params.get<std::vector<std::string>>("p4"),
                 Catch::Matchers::Equals(std::vector<std::string>{"a", "b", "c", "d", "e"}));
    REQUIRE_THAT(params.get<std::vector<double>>("p5"),
                 Catch::Matchers::Approx(std::vector<double>{1.2, -1.1, 100, 5.3}));
  }
}
