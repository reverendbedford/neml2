#include <catch2/catch.hpp>

#include "Scalar.h"

TEST_CASE("Setup from scalar types",
          "[Scalar]") {

  SECTION("Set from scalar type") {
    Scalar t = 2.5;
    REQUIRE(t.value() == Approx(2.5));
  }

}
