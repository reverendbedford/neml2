#include <catch2/catch.hpp>

#include "Scalar.h"

TEST_CASE("Setup from scalar types", "[Scalar]")
{
  SECTION("Set from scalar type")
  {
    Scalar t = 2.5;
    REQUIRE(t.value() == Approx(2.5));
  }
}

TEST_CASE("Scalar can't be created with semantically non-scalar tensors", "[Scalar]")
{
  // This can't happen as the tensor dimension is not (1,)
  REQUIRE_THROWS(Scalar(torch::zeros({2, 2}, TorchDefaults)));
}
