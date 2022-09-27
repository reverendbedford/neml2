#include <catch2/catch.hpp>

#include "BatchedSymSymR4.h"
#include "State.h"
#include "StateDerivative.h"

TEST_CASE("All combinations of valid State objects can be accessed", "[StateDerivative]")
{
  StateInfo first;
  first.add<BatchedScalar>("one");
  first.add<BatchedSymR2>("two");

  StateInfo second;
  second.add<BatchedSymR2>("three");
  second.add<BatchedScalar>("four");

  TorchSize bs = 10;

  StateDerivative test(first, second, bs);

  SECTION("Scalar/Scalar")
  {
    test.set<BatchedScalar>("one", "four", BatchedScalar(torch::ones({bs, 1})));
    BatchedScalar value = test.get<BatchedScalar>("one", "four");
    REQUIRE(value.batch_size() == bs);
    REQUIRE(torch::sum(value).item<double>() == Approx(10));
  }

  SECTION("SymR2/Scalar")
  {
    test.set<BatchedSymR2>("two", "four", BatchedSymR2(torch::ones({bs, 6})));
    BatchedSymR2 value = test.get<BatchedSymR2>("two", "four");
    REQUIRE(value.batch_size() == bs);
    REQUIRE(torch::sum(value).item<double>() == Approx(60));
  }

  SECTION("Scalar/SymR2")
  {
    test.set<BatchedSymR2>("one", "three", BatchedSymR2(torch::ones({bs, 6})));
    BatchedSymR2 value = test.get<BatchedSymR2>("one", "three");
    REQUIRE(value.batch_size() == bs);
    REQUIRE(torch::sum(value).item<double>() == Approx(60));
  }

  SECTION("SymR2/SymR2")
  {
    test.set<BatchedSymSymR4>("two", "three", BatchedSymSymR4(torch::ones({bs, 6, 6})));
    BatchedSymSymR4 value = test.get<BatchedSymSymR4>("two", "three");
    REQUIRE(value.batch_size() == bs);
    REQUIRE(torch::sum(value).item<double>() == Approx(360));
  }
}
