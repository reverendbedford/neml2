#include <catch2/catch.hpp>

#include "tensors/SymSymR4.h"
#include "state/State.h"
#include "state/StateDerivative.h"

TEST_CASE("All combinations of valid State objects can be accessed", "[StateDerivative]")
{
  StateInfo first;
  first.add<Scalar>("one");
  first.add<SymR2>("two");

  StateInfo second;
  second.add<SymR2>("three");
  second.add<Scalar>("four");

  TorchSize bs = 10;

  StateDerivative test(first, second, bs);

  SECTION("Scalar/Scalar")
  {
    test.set<Scalar>("one", "four", Scalar(torch::ones({bs, 1})));
    Scalar value = test.get<Scalar>("one", "four");
    REQUIRE(value.batch_sizes()[0] == bs);
    REQUIRE(torch::sum(value).item<double>() == Approx(10));
  }

  SECTION("SymR2/Scalar")
  {
    test.set<SymR2>("two", "four", SymR2(torch::ones({bs, 6})));
    SymR2 value = test.get<SymR2>("two", "four");
    REQUIRE(value.batch_sizes()[0] == bs);
    REQUIRE(torch::sum(value).item<double>() == Approx(60));
  }

  SECTION("Scalar/SymR2")
  {
    test.set<SymR2>("one", "three", SymR2(torch::ones({bs, 6})));
    SymR2 value = test.get<SymR2>("one", "three");
    REQUIRE(value.batch_sizes()[0] == bs);
    REQUIRE(torch::sum(value).item<double>() == Approx(60));
  }

  SECTION("SymR2/SymR2")
  {
    test.set<SymSymR4>("two", "three", SymSymR4(torch::ones({bs, 6, 6})));
    SymSymR4 value = test.get<SymSymR4>("two", "three");
    REQUIRE(value.batch_sizes()[0] == bs);
    REQUIRE(torch::sum(value).item<double>() == Approx(360));
  }
}
