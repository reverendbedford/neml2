#include <catch2/catch.hpp>

#include "TestUtils.h"

#include "models/solid_mechanics/AssociativeInelasticFlowDirection.h"
#include "models/solid_mechanics/AssociativeInelasticHardening.h"
#include "models/solid_mechanics/InelasticModel.h"
#include "models/solid_mechanics/J2IsotropicYieldSurface.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"
#include "models/solid_mechanics/PerzynaInelasticFlowRate.h"

TEST_CASE("InelasticModel defined correctly", "[InelasticModel]")
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
  auto direction = AssociativeInelasticFlowDirection(surface, hardening);
  auto hmodel = AssociativeInelasticHardening(surface, hardening, rate);

  Scalar E = 100000.0;
  Scalar nu = 0.3;

  auto C = SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {E, nu});

  auto model = InelasticModel(C, rate, direction, hmodel);

  State state(model.state(), batch);
  auto s = SymR2::init(100, 110, 100, 100 / utils::sqrt2, 100 / utils::sqrt2, 100 / utils::sqrt2)
               .expand_batch(batch);
  state.set<SymR2>("stress", s);
  Scalar ep(0.1, batch);
  state.set<Scalar>("equivalent_plastic_strain", ep);

  State force_rate(model.forces().remove("time").add_suffix("_rate"), batch);
  auto erate = SymR2::init(0.1, 0, 0, 0, 0, 0).expand_batch(batch);
  force_rate.set<SymR2>("strain_rate", erate);

  State force(model.forces(), batch);
  auto strain = SymR2::init(0.01, 0, 0, 0, 0, 0).expand_batch(batch);
  Scalar time(1, batch);
  force.set<SymR2>("strain", strain);
  force.set<Scalar>("time", time);

  SECTION("state rate has the correct output defined")
  {
    auto state_rate = model.value({state, force, force_rate});
    REQUIRE(state_rate.info().items() ==
            std::vector<std::string>({"equivalent_plastic_strain_rate", "stress_rate"}));
  }

  // Get all the derivatives
  auto num_derivs = state_derivatives(
      std::bind(&InelasticModel::value, model, std::placeholders::_1), {state, force, force_rate});
  auto exact_derivs = model.dvalue({state, force, force_rate});

  SECTION("derivative with respect to state is correct")
  {
    REQUIRE(torch::allclose(num_derivs[0].tensor(), exact_derivs[0].tensor(), 1e-4));
  }

  SECTION("derivative with respect to forces is correct")
  {
    REQUIRE(torch::allclose(num_derivs[1].tensor(), exact_derivs[1].tensor()));
  }

  SECTION("derivative with respect to forces_rate is correct")
  {
    REQUIRE(torch::allclose(num_derivs[2].tensor(), exact_derivs[2].tensor()));
  }
}
