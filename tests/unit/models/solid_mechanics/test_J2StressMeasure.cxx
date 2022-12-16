#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "models/solid_mechanics/J2StressMeasure.h"

using namespace neml2;

TEST_CASE("J2StressMeasure", "[J2StressMeasure]")
{
  TorchSize nbatch = 10;
  Scalar s0 = 50.0;
  auto sm = J2StressMeasure("stress_measure");

  SECTION("model definition")
  {
    REQUIRE(sm.input().has_subaxis("state"));
    REQUIRE(sm.input().subaxis("state").has_variable<SymR2>("overstress"));
    REQUIRE(sm.output().has_subaxis("state"));
    REQUIRE(sm.output().subaxis("state").has_variable<Scalar>("stress_measure"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, sm.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(M, "overstress");

    auto exact = sm.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, sm.output(), sm.input());
    finite_differencing_derivative(
        [sm](const LabeledVector & x) { return sm.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));

    auto exactd2 = sm.d2value(in);
    auto numericald2 = LabeledTensor<1, 3>(nbatch, sm.output(), sm.input(), sm.input());
    finite_differencing_derivative(
        [sm](const LabeledVector & x) { return sm.dvalue(x); }, in, numericald2);
    
    REQUIRE(torch::allclose(exactd2.tensor(), numericald2.tensor()));
  }
}
