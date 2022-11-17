#include <catch2/catch.hpp>

#include "utils.h"

#include "models/solid_mechanics/AssociativeInelasticFlowDirection.h"
#include "models/solid_mechanics/AssociativeInelasticHardening.h"
#include "models/solid_mechanics/InelasticModel.h"
#include "models/IntegratedRateFormModel.h"
#include "models/solid_mechanics/J2IsotropicYieldSurface.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"
#include "models/solid_mechanics/PerzynaInelasticFlowRate.h"

TEST_CASE("IntegratedRateFormModel defined correctly", "[IntegratedRateFormModel]")
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

  auto imodel = InelasticModel(C, rate, direction, hmodel);

  auto model = IntegratedRateFormModel(imodel);

  State state_np1(model.state(), batch);
  auto s = SymR2::init(100, 110, 100, 100 / utils::sqrt2, 100 / utils::sqrt2, 100 / utils::sqrt2)
               .expand_batch(batch);
  state_np1.set<SymR2>("stress", s);
  Scalar ep(0.1, batch);
  state_np1.set<Scalar>("equivalent_plastic_strain", ep);

  State state_n(model.state(), batch);
  state_n.set<SymR2>("stress", SymR2::init(0).expand_batch(batch));
  state_n.set<Scalar>("equivalent_plastic_strain", Scalar(0, batch));

  State force_np1(model.forces(), batch);
  auto strain = SymR2::init(0.01, 0, 0, 0, 0, 0).expand_batch(batch);
  Scalar time(1.2, batch);
  force_np1.set<SymR2>("strain", strain);
  force_np1.set<Scalar>("time", time);

  State force_n(model.forces(), batch);
  force_n.set<SymR2>("strain", SymR2::init(0).expand_batch(batch));
  force_n.set<Scalar>("time", Scalar(0, batch));

  SECTION(" implicit function defined correctly")
  {
    auto R = model.value({state_np1, force_np1, state_n, force_n});
    REQUIRE(R.info().items() == std::vector<std::string>({"equivalent_plastic_strain", "stress"}));
    REQUIRE(R.tensor().sizes() == TorchShape({batch, 7}));

    std::cout << R.tensor().batch_index({0}) << std::endl;
    exit(0);
  }

  auto exact_derivs = model.dvalue({state_np1, force_np1, state_n, force_n});
  auto num_derivs = utils::state_derivatives(
      std::bind(&IntegratedRateFormModel::value, model, std::placeholders::_1),
      {state_np1, force_np1, state_n, force_n});

  SECTION(" derivative with respect to current state is correct")
  {
    REQUIRE(torch::allclose(exact_derivs[0].tensor(), num_derivs[0].tensor(), 1e-4));
  }

  SECTION(" derivative with respect to current force is correct")
  {
    REQUIRE(torch::allclose(exact_derivs[1].tensor(), num_derivs[1].tensor(), 1e-4));
  }

  SECTION(" derivative with respect to previous state is correct")
  {
    REQUIRE(torch::allclose(exact_derivs[2].tensor(), num_derivs[2].tensor(), 1e-4));
  }

  SECTION(" derivative with respect to previous force is correct")
  {
    REQUIRE(torch::allclose(exact_derivs[3].tensor(), num_derivs[3].tensor()));
  }

  SECTION(" residual gives the correct values")
  {
    model.setup({force_np1, state_n, force_n});
    auto R1 = model.value({state_np1, force_np1, state_n, force_n});
    auto R2 = model.residual(state_np1.tensor());
    REQUIRE(torch::allclose(R1.tensor(), R2));
  }

  SECTION(" jacobian gives the correct value")
  {
    model.setup({force_np1, state_n, force_n});
    auto J1 = model.dvalue({state_np1, force_np1, state_n, force_n})[0];
    auto J2 = model.jacobian(state_np1.tensor());
    REQUIRE(torch::allclose(J1.tensor(), J2));
  }

  SECTION(" trial_state gives the correct values")
  {
    model.setup({force_np1, state_n, force_n});
    auto s1 = model.trial_state();
    REQUIRE(torch::allclose(s1.tensor(), state_n.tensor()));
  }
}
