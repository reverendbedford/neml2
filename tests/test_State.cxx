#include <catch2/catch.hpp>

#include "State.h"

TEST_CASE("Can setup, get, and set State with basic objects", "[State]")
{
  StateInfo s;
  s.add<BatchedSymR2>("symmetric1");
  s.add<BatchedScalar>("scalar1");
  s.add<BatchedSymR2>("symmetric2");
  s.add<BatchedScalar>("scalar2");

  TorchSize bs = 10;
  State test(s, bs);

  SECTION("Set and get a particular value")
  {
    test.set<BatchedScalar>("scalar1", BatchedScalar(torch::ones({bs, 1})));
    REQUIRE(torch::sum(test.get<BatchedScalar>("scalar1")).item<double>() == Approx(10));
    test.set<BatchedSymR2>("symmetric1", BatchedSymR2(torch::ones({bs, 6})));
    REQUIRE(torch::sum(test.get<BatchedSymR2>("symmetric1")).item<double>() == Approx(60));
  }
}

TEST_CASE("Operations on a substate affect the parent", "[State]")
{
  StateInfo secondary;
  secondary.add<BatchedScalar>("scalar_in_secondary");
  StateInfo primary;
  primary.add<BatchedScalar>("scalar_in_primary");
  primary.add_substate("secondary", secondary);

  TorchSize bs = 10;
  State test(primary, bs);

  SECTION("Set through the primary, check secondary")
  {
    test.set<BatchedScalar>("scalar_in_secondary", BatchedScalar(torch::ones({bs, 1})));
    State secondary_state = test.get_substate("secondary");
    REQUIRE(torch::sum(secondary_state.get<BatchedScalar>("scalar_in_secondary")).item<double>() ==
            Approx(10));
  }

  SECTION("Set through the secondary, check primary")
  {
    State secondary_state = test.get_substate("secondary");
    secondary_state.set<BatchedScalar>("scalar_in_secondary", BatchedScalar(torch::ones({bs, 1})));
    BatchedScalar new_value = test.get<BatchedScalar>("scalar_in_secondary");
    REQUIRE(torch::sum(new_value).item<double>() == Approx(10));
  }
}
