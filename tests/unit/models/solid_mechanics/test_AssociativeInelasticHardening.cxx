#include "TestUtils.h"
#include <catch2/catch.hpp>

#include "models/solid_mechanics/AssociativeInelasticHardening.h"
#include "models/solid_mechanics/J2IsotropicYieldSurface.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"
#include "models/solid_mechanics/PerzynaInelasticFlowRate.h"

TEST_CASE("AssociativeInelasticHardening defined correctly", "[AssociativeInelasticHardening]")
{
  // Basic setup
  TorchSize batch = 10;
  auto surface = J2IsotropicYieldSurface();
  auto s0 = 100.0;
  Scalar K = 1000.0;
  auto hardening = LinearIsotropicHardening(s0, K);
  Scalar eta = 150.0;
  Scalar n = 6.0;
  auto rate = PerzynaInelasticFlowRate(eta, n, surface, hardening);
  auto model = AssociativeInelasticHardening(surface, hardening, rate);

  StateInfo input;
  input.add<SymR2>("stress");
  input.add<Scalar>("equivalent_plastic_strain");
  State state(input, batch);
  auto s = SymR2::init(100, 110, 100, 100, 100, 100).expand_batch(batch);
  state.set<SymR2>("stress", s);
  state.set<Scalar>("equivalent_plastic_strain", Scalar(0.1, batch));

  SECTION(" defines equivalent_plastic_strain_rate")
  {
    auto res = model.value(state);
    REQUIRE(res.info().items() == std::vector<std::string>({"equivalent_plastic_strain_rate"}));
  }

  SECTION(" derivative is correct and has the correct names")
  {
    auto exact = model.dvalue(state);
    auto numerical = state_derivative(
        std::bind(&AssociativeInelasticHardening::value, model, std::placeholders::_1), state);

    // Check on the names of everything
    REQUIRE(exact.info_A().items() == std::vector<std::string>({"equivalent_plastic_strain_rate"}));
    REQUIRE(exact.info_B().items() ==
            std::vector<std::string>({"equivalent_plastic_strain", "stress"}));

    // Check the actual values
    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor(), 1e-4));
  }
}
