#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "models/solid_mechanics/J2IsotropicYieldFunction.h"

using namespace neml2;

TEST_CASE("J2IsotropicYieldFunction", "[J2IsotropicYieldFunction]")
{
  TorchSize nbatch = 10;
  auto yield = J2IsotropicYieldFunction("yield_function");

  SECTION("model definition")
  {
    REQUIRE(yield.input().has_subaxis("state"));
    REQUIRE(yield.input().subaxis("state").has_variable<SymR2>("mandel_stress"));
    REQUIRE(yield.input().has_subaxis("state"));
    REQUIRE(yield.input().subaxis("state").has_variable<Scalar>("isotropic_hardening"));
    REQUIRE(yield.output().has_subaxis("state"));
    REQUIRE(yield.output().subaxis("state").has_variable<Scalar>("yield_function"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, yield.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(Scalar(200, nbatch), "isotropic_hardening");
    in.slice("state").set(M, "mandel_stress");

    auto exact = yield.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, yield.output(), yield.input());
    finite_differencing_derivative(
        [yield](const LabeledVector & x) { return yield.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
