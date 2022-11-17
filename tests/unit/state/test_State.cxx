#include <catch2/catch.hpp>

#include "state/State.h"

TEST_CASE("Can setup, get, and set State with basic objects", "[State]")
{
  StateInfo s;
  s.add<SymR2>("symmetric1");
  s.add<Scalar>("scalar1");
  s.add<SymR2>("symmetric2");
  s.add<Scalar>("scalar2");

  TorchSize bs = 10;
  State test(s, bs);

  SECTION("Set and get a particular value")
  {
    test.set<Scalar>("scalar1", Scalar(1, bs));
    REQUIRE(torch::sum(test.get<Scalar>("scalar1")).item<double>() == Approx(10));
    test.set<SymR2>("symmetric1", SymR2(torch::ones({bs, 6})));
    REQUIRE(torch::sum(test.get<SymR2>("symmetric1")).item<double>() == Approx(60));
  }
}

TEST_CASE("Can rename state variables", "[State]")
{
  StateInfo s;
  s.add<SymR2>("symmetric1");
  s.add<Scalar>("scalar1");
  s.add<SymR2>("symmetric2");
  s.add<Scalar>("scalar2");

  TorchSize bs = 10;
  State test(s, bs);

  REQUIRE(test.info().items() ==
          std::vector<std::string>({"scalar1", "scalar2", "symmetric1", "symmetric2"}));
  test.rename("symmetric2", "blah");
  REQUIRE(test.info().items() ==
          std::vector<std::string>({"blah", "scalar1", "scalar2", "symmetric1"}));
}

TEST_CASE("Operations on a substate affect the parent", "[State]")
{
  StateInfo secondary;
  secondary.add<Scalar>("scalar_in_secondary");
  StateInfo primary;
  primary.add<Scalar>("scalar_in_primary");
  primary.add_substate("secondary", secondary);

  TorchSize bs = 10;
  State test(primary, bs);

  SECTION("Set through the primary, check secondary")
  {
    test.set<Scalar>("scalar_in_secondary", Scalar(1, bs));
    State secondary_state = test.get_substate("secondary");
    REQUIRE(torch::sum(secondary_state.get<Scalar>("scalar_in_secondary")).item<double>() ==
            Approx(10));
  }

  SECTION("Set through the secondary, check primary")
  {
    State secondary_state = test.get_substate("secondary");
    secondary_state.set<Scalar>("scalar_in_secondary", Scalar(1, bs));
    Scalar new_value = test.get<Scalar>("scalar_in_secondary");
    REQUIRE(torch::sum(new_value).item<double>() == Approx(10));
  }
}

TEST_CASE("Cloned copies of state are actual deep copies")
{
  StateInfo s;
  s.add<Scalar>("scalar1");

  TorchSize bs = 10;
  State test(s, bs);
  test.set<Scalar>("scalar1", Scalar(1, bs));

  State copy = test.clone();
  copy.set<Scalar>("scalar1", Scalar(2, bs));

  REQUIRE(torch::allclose(test.get<Scalar>("scalar1"), torch::ones({bs, 1}, TorchDefaults)));
  REQUIRE(torch::allclose(copy.get<Scalar>("scalar1"),
                          Scalar(torch::ones({bs, 1}, TorchDefaults) * 2.0)));
}

TEST_CASE("We can remove objects from State", "[State]")
{
  StateInfo s;
  s.add<SymR2>("symmetric1");
  s.add<Scalar>("scalar1");
  s.add<SymR2>("symmetric2");
  s.add<Scalar>("scalar2");

  TorchSize bs = 10;
  State test(s, bs);

  REQUIRE(test.info().items() ==
          std::vector<std::string>({"scalar1", "scalar2", "symmetric1", "symmetric2"}));

  SECTION(".remove gives the correct state")
  {
    State test2 = test.remove("symmetric1");
    REQUIRE(test2.info().items() == std::vector<std::string>({"scalar1", "scalar2", "symmetric2"}));
    REQUIRE(test2.tensor().sizes() == TorchShape({bs, 8}));
  }

  SECTION(".remove modifies the right values")
  {
    State test2 = test.remove("symmetric1");
    auto val = torch::ones({bs, 1}, TorchDefaults);
    test2.set<Scalar>("scalar2", val);
    REQUIRE(torch::allclose(test2.get<Scalar>("scalar2"), val));
  }
}
