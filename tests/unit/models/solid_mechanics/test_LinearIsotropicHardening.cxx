#include "TestUtils.h"
#include <catch2/catch.hpp>

#include "models/solid_mechanics/LinearIsotropicHardening.h"

TEST_CASE("Derivative of map is correct", "[LinearIsotropicHardening]")
{
  // Setup a reasonable problem
  // Basic setup
  TorchSize batch = 10;
  Scalar s0 = 100.0;
  Scalar K = 1000.0;

  auto model = LinearIsotropicHardening(s0, K);

  StateInfo input;
  input.add<SymR2>("stress");
  input.add<Scalar>("equivalent_plastic_strain");
  State state(input, batch);
  auto s = SymR2::init(100, 110, 100, 100, 100, 100).expand_batch(batch);
  state.set<SymR2>("stress", s);
  state.set<Scalar>("equivalent_plastic_strain", Scalar(0.1, batch));

  SECTION("test derivative")
  {
    auto direct = model.dvalue(state);
    auto numerical = state_derivative(
        std::bind(&LinearIsotropicHardening::value, model, std::placeholders::_1), state);
    REQUIRE(torch::allclose(direct.tensor(), numerical.tensor()));
  }
}
