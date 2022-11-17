#include <catch2/catch.hpp>

#include "tensors/Scalar.h"
#include "state/StateInfo.h"

TEST_CASE("Add to StateInfo", "[StateInfo]")
{
  // Empty
  StateInfo test;

  // Add a Scalar
  test.add<Scalar>("scalar");
  REQUIRE(test.nitems() == 1);
  REQUIRE(test.size_storage() == 1);
  REQUIRE(test.nsubstates() == 0);

  // Add a SymR2
  test.add<SymR2>("symmetric");
  REQUIRE(test.nitems() == 2);
  REQUIRE(test.size_storage() == 7);
  REQUIRE(test.nsubstates() == 0);

  // Check the actual size
  REQUIRE(test.required_shape(10) == TorchShape({10, 7}));
}

TEST_CASE("Can rename variables", "[StateInfo]")
{
  // Empty
  StateInfo test;

  // Add a Scalar
  test.add<Scalar>("scalar");

  // Add a SymR2
  test.add<SymR2>("symmetric");

  // Check the actual size
  REQUIRE(test.items() == std::vector<std::string>({"scalar", "symmetric"}));
  test.rename("scalar", "blah");
  REQUIRE(test.items() == std::vector<std::string>({"blah", "symmetric"}));
}

TEST_CASE("Test that we can add objects through substate", "[StateInfo]")
{
  // Empty
  StateInfo test;

  // Add a Scalar
  test.add<Scalar>("scalar");
  REQUIRE(test.nitems() == 1);
  REQUIRE(test.size_storage() == 1);
  REQUIRE(test.nsubstates() == 0);

  // Substate with 2 symmetric tensors and a scalar
  StateInfo substate;
  substate.add<SymR2>("one_sub_sym");
  substate.add<Scalar>("one_sub_scalar");
  substate.add<SymR2>("two_sub_symmetric");

  test.add_substate("substate", substate);

  // Shapes must be correct
  REQUIRE(test.nitems() == 4);
  REQUIRE(test.size_storage() == 14);
  REQUIRE(test.nsubstates() == 1);
  REQUIRE(test.required_shape(10) == TorchShape({10, 14}));
}
