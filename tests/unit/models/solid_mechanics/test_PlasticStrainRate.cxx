#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "neml2/models/solid_mechanics/PlasticStrainRate.h"

using namespace neml2;

TEST_CASE("PlasticStrainRate", "[PlasticStrainRate]")
{
  TorchSize nbatch = 10;
  auto pstrainrate = PlasticStrainRate("plastic_strain_rate");

  SECTION("model definition")
  {
    REQUIRE(pstrainrate.input().has_subaxis("state"));
    REQUIRE(pstrainrate.input().subaxis("state").has_variable<SymR2>("plastic_flow_direction"));
    REQUIRE(pstrainrate.input().subaxis("state").has_variable<Scalar>("hardening_rate"));

    REQUIRE(pstrainrate.output().has_subaxis("state"));
    REQUIRE(pstrainrate.output().subaxis("state").has_variable<SymR2>("plastic_strain_rate"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, pstrainrate.input());
    // Well, in principle we should give Np:Np = 3/2, but whatever
    auto Np = SymR2::init(0, 0, 0, 1, 0.3, 0.8).batch_expand(nbatch);
    in.slice("state").set(Np, "plastic_flow_direction");
    in.slice("state").set(Scalar(0.01, nbatch), "hardening_rate");

    auto exact = pstrainrate.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, pstrainrate.output(), pstrainrate.input());
    finite_differencing_derivative(
        [pstrainrate](const LabeledVector & x) { return pstrainrate.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
