#include "utils.h"
#include <catch2/catch.hpp>

#include "models/solid_mechanics/AssociativeInelasticFlowDirection.h"
#include "models/solid_mechanics/J2IsotropicYieldSurface.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"

TEST_CASE("AssociativeInelasticFlowDirection defined correctly",
          "[AssociativeInelasticFlowDirection]")
{
  // Basic setup
  TorchSize batch = 10;
  auto surface = J2IsotropicYieldSurface();
  auto s0 = 100.0;
  Scalar K = 1000.0;
  auto hardening = LinearIsotropicHardening(s0, K);
  auto model = AssociativeInelasticFlowDirection(surface, hardening);

  StateInfo input;
  input.add<SymR2>("stress");
  input.add<Scalar>("equivalent_plastic_strain");
  State state(input, batch);
  auto s = SymR2::init(100, 110, 100, 100, 100, 100).expand_batch(batch);
  state.set<SymR2>("stress", s);
  state.set<Scalar>("equivalent_plastic_strain", Scalar(0.1, batch));

  SECTION("produces flow direction")
  {
    auto test = model.value(state);
    REQUIRE(test.info().items() == std::vector<std::string>({"flow_direction"}));
  }

  SECTION("derivative")
  {
    auto direct = model.dvalue(state);
    auto numerical = utils::state_derivative(
        std::bind(&AssociativeInelasticFlowDirection::value, model, std::placeholders::_1), state);
    REQUIRE(torch::allclose(direct.tensor(), numerical.tensor()));
  }
}
