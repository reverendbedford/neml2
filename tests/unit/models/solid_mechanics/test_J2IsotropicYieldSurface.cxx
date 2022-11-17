#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "models/solid_mechanics/J2IsotropicYieldSurface.h"

TEST_CASE("J2IsotropicYieldSurface defined correctly", "[J2IsotropicYieldSurface]")
{
  auto surface = J2IsotropicYieldSurface();
  TorchSize batch = 10;
  StateInfo interface = surface.interface();
  State state(interface, batch);

  auto s = SymR2::init(100, 110, 100, 100, 100, 100).expand_batch(batch);
  state.set<SymR2>("stress", s);
  state.set<Scalar>("isotropic_hardening", Scalar(10, batch));

  SECTION("df_ds")
  {
    auto direct = surface.df_ds(state);
    auto numerical = scalar_derivative(
        std::bind(&J2IsotropicYieldSurface::f, surface, std::placeholders::_1), state);
    REQUIRE(torch::allclose(direct.tensor(), numerical.tensor()));
  }

  SECTION("d2f_ds2")
  {
    auto direct = surface.d2f_ds2(state);
    auto numerical = state_derivative(
        std::bind(&J2IsotropicYieldSurface::df_ds, surface, std::placeholders::_1), state);
    REQUIRE(torch::allclose(direct.tensor(), numerical.tensor()));
  }
}
