#include "TestUtils.h"
#include <catch2/catch.hpp>

#include "models/solid_mechanics/J2IsotropicYieldSurface.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"
#include "models/solid_mechanics/PerzynaInelasticFlowRate.h"

TEST_CASE("PerzynaInelasticFlowRate defined correctly", "[PerzynaInelasticFlowRate]")
{
  // Basic setup
  TorchSize batch = 10;
  auto surface = J2IsotropicYieldSurface();
  Scalar s0 = 100.0;
  Scalar K = 1000.0;
  auto hardening = LinearIsotropicHardening(s0, K);
  Scalar eta = 150.0;
  Scalar n = 6.0;
  auto model = PerzynaInelasticFlowRate(eta, n, surface, hardening);

  StateInfo input;
  input.add<SymR2>("stress");
  input.add<Scalar>("equivalent_plastic_strain");
  State state(input, batch);
  auto s = SymR2::init(100, 110, 100, 100, 100, 100).expand_batch(batch);
  state.set<SymR2>("stress", s);
  state.set<Scalar>("equivalent_plastic_strain", Scalar(0.1, batch));

  SECTION(" defines flow rate")
  {
    auto fr = model.value(state);
    REQUIRE(fr.info().items() == std::vector<std::string>({"flow_rate"}));
  }

  SECTION(" defines flow rate derivative")
  {
    auto dfr = model.dvalue(state);
    REQUIRE(dfr.info_A().items() == std::vector<std::string>({"flow_rate"}));
    REQUIRE(dfr.info_B().items() ==
            std::vector<std::string>({"equivalent_plastic_strain", "stress"}));
  }

  SECTION(" derivative is correct")
  {
    auto direct = model.dvalue(state);
    auto numerical = state_derivative(
        std::bind(&PerzynaInelasticFlowRate::value, model, std::placeholders::_1), state);
    REQUIRE(torch::allclose(direct.tensor(), numerical.tensor(), 1e-4));
  }
}
