#include <catch2/catch.hpp>

#include "BatchedSymR2.h"
#include "Scalar.h"
#include "StateInfo.h"

TEST_CASE("Test that we can add all the required objects to StateInfo "
          "and that the sizes are correct as we do so",
          "[StateInfo]")
{
  // Empty
  StateInfo test;

  // Add a BatchedScalar
  test.add<BatchedScalar>("scalar");
  REQUIRE(test.nitems() == 1);
  REQUIRE(test.size_storage() == 1);
  REQUIRE(test.nsubstates() == 0);

  // Add a BatchedSymR2
  test.add<BatchedSymR2>("symmetric");
  REQUIRE(test.nitems() == 2);
  REQUIRE(test.size_storage() == 7);
  REQUIRE(test.nsubstates() == 0);

  // Check the actual size
  REQUIRE(test.required_shape(10) == TorchShape({10, 7}));
}

TEST_CASE("Test that we can add objects through substate", "[StateInfo]")
{
  // Empty
  StateInfo test;

  // Add a BatchedScalar
  test.add<BatchedScalar>("scalar");
  REQUIRE(test.nitems() == 1);
  REQUIRE(test.size_storage() == 1);
  REQUIRE(test.nsubstates() == 0);

  // Substate with 2 symmetric tensors and a scalar
  StateInfo substate;
  substate.add<BatchedSymR2>("one_sub_sym");
  substate.add<BatchedScalar>("one_sub_scalar");
  substate.add<BatchedSymR2>("two_sub_symmetric");

  test.add_substate("substate", substate);

  // Shapes must be correct
  REQUIRE(test.nitems() == 4);
  REQUIRE(test.size_storage() == 14);
  REQUIRE(test.nsubstates() == 1);
  REQUIRE(test.required_shape(10) == TorchShape({10, 14}));
}
