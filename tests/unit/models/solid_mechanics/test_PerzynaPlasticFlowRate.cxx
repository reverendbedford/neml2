#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "neml2/models/solid_mechanics/PerzynaPlasticFlowRate.h"

using namespace neml2;

TEST_CASE("PerzynaPlasticFlowRate", "[PerzynaPlasticFlowRate]")
{
  TorchSize nbatch = 10;
  Scalar eta = 150;
  Scalar n = 6;
  auto eprate = PerzynaPlasticFlowRate("plastic_flow_rate", eta, n);

  SECTION("model definition")
  {
    REQUIRE(eprate.input().has_subaxis("state"));
    REQUIRE(eprate.input().subaxis("state").has_variable<Scalar>("yield_function"));
    REQUIRE(eprate.output().has_subaxis("state"));
    REQUIRE(eprate.output().subaxis("state").has_variable<Scalar>("hardening_rate"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, eprate.input());
    in.slice("state").set(Scalar(1.3, nbatch), "yield_function");

    auto exact = eprate.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, eprate.output(), eprate.input());
    finite_differencing_derivative(
        [eprate](const LabeledVector & x) { return eprate.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
