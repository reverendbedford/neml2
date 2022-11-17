#include <catch2/catch.hpp>

#include "utils.h"

#include "models/solid_mechanics/AssociativeInelasticFlowDirection.h"
#include "models/solid_mechanics/AssociativeInelasticHardening.h"
#include "models/solid_mechanics/InelasticModel.h"
#include "models/IntegratedRateFormModel.h"
#include "models/solid_mechanics/J2IsotropicYieldSurface.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"
#include "solvers/NewtonNonlinearSolver.h"
#include "models/solid_mechanics/PerzynaInelasticFlowRate.h"
#include "models/SolveImplicitFunctionModel.h"

TEST_CASE("SolveImplicitFunctionModel defined correctly", "[SolveImplicitFunctionModel]")
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

  auto rmodel = IntegratedRateFormModel(imodel);

  NonlinearSolverParameters params;
  NewtonNonlinearSolver solver(params);

  auto model = SolveImplicitFunctionModel(rmodel, solver);

  State state_n(model.state(), batch);
  state_n.set<SymR2>("stress", SymR2::init(30, 0, 0).expand_batch(batch));
  state_n.set<Scalar>("equivalent_plastic_strain", Scalar(1e-3, batch));

  State force_np1(model.forces(), batch);
  force_np1.set<SymR2>("strain", SymR2::init(2e-3, 0, 0).expand_batch(batch));
  force_np1.set<Scalar>("time", Scalar(1.2, batch));

  State force_n(model.forces(), batch);
  force_n.set<SymR2>("strain", SymR2::init(0).expand_batch(batch));
  force_n.set<Scalar>("time", Scalar(0, batch));

  SECTION(" state update completes without error")
  {
    auto state_np1 = model.value({force_np1, state_n, force_n});
  }

  // I don't like this, but you need to call the update first
  auto state_np1 = model.value({force_np1, state_n, force_n});
  // Now I can get the tangents
  auto tangents = model.dvalue({force_np1, state_n, force_n});
  // Need to wrap this in a lambda because of the update...
  auto num_tangents = utils::state_derivatives(
      std::bind(&SolveImplicitFunctionModel::value, model, std::placeholders::_1),
      {force_np1, state_n, force_n},
      1e-6,
      1e-12);

  SECTION(" dstate_np1/dforce_np1 correct")
  {
    REQUIRE(torch::allclose(num_tangents[0].tensor(), tangents[0].tensor(), 1e-4));
  }

  SECTION(" dstate_np1/dstate_n correct")
  {
    REQUIRE(torch::allclose(num_tangents[1].tensor(), tangents[1].tensor(), 1e-4, 1e-1));
  }

  SECTION(" dstate_np1/dforce_n correct")
  {
    REQUIRE(torch::allclose(num_tangents[2].tensor(), tangents[2].tensor(), 1e-4, 1e-1));
  }
}
