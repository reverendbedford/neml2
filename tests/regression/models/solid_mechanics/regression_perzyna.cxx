#include <catch2/catch.hpp>

#include "UniaxialStrainStructuralDriver.h"

#include "models/solid_mechanics/AssociativeInelasticFlowDirection.h"
#include "models/solid_mechanics/AssociativeInelasticHardening.h"
#include "models/solid_mechanics/InelasticModel.h"
#include "models/solid_mechanics/PerzynaInelasticFlowRate.h"
#include "models/IntegratedRateFormModel.h"
#include "models/solid_mechanics/J2IsotropicYieldSurface.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"
#include "solvers/NewtonNonlinearSolver.h"
#include "models/solid_mechanics/PerzynaInelasticFlowRate.h"
#include "models/SolveImplicitFunctionModel.h"

TEST_CASE("Uniaxial strain regression test for Perzyna model", "[StructuralRegressionTests]")
{
  // Model name
  std::string name = "perzyna_structural_model";

  // Basic setup
  auto surface = J2IsotropicYieldSurface();
  auto s0 = 5.0;
  Scalar K = 1000.0;
  auto hardening = LinearIsotropicHardening(s0, K);
  Scalar eta = 100.0;
  Scalar n = 2.0;
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

  // Unit test parameters
  TorchSize nbatch = 20;
  TorchSize nsteps = 100;
  Real max_strain = 0.20;
  Real min_rate = -6;
  Real max_rate = 0;

  auto max_strains = torch::full({nbatch}, max_strain, TorchDefaults);
  auto strain_rates = torch::logspace(min_rate, max_rate, nbatch, 10.0, TorchDefaults);

  // Run the unit test
  UniaxialStrainStructuralDriver driver(model, max_strains, strain_rates);
  driver.run(nsteps);

  // Save the results (for now just comment out)
  // write_csv(driver.forces(), name+"_forces.csv");
  // write_csv(driver.states(), name+"_states.csv");

  // torch::save(driver.forces(), name+"_forces.pt");
  // torch::save(driver.states(), name+"_states.pt");

  // Load the regression test data back in
  auto forces_ref = torch::zeros_like(driver.forces());
  torch::load(forces_ref, "regression/models/solid_mechanics/" + name + "_forces.pt");
  auto states_ref = torch::zeros_like(driver.states());
  torch::load(states_ref, "regression/models/solid_mechanics/" + name + "_states.pt");

  REQUIRE(torch::allclose(driver.forces(), forces_ref));
  REQUIRE(torch::allclose(driver.states(), states_ref));
}
