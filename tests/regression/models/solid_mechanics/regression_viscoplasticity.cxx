#include <catch2/catch.hpp>

#include <fstream>

#include "models/ComposedModel.h"
#include "models/solid_mechanics/ElasticStrain.h"
#include "models/solid_mechanics/LinearIsotropicElasticity.h"
#include "models/solid_mechanics/NoKinematicHardening.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"
#include "models/solid_mechanics/J2IsotropicYieldFunction.h"
#include "models/solid_mechanics/AssociativePlasticFlowDirection.h"
#include "models/solid_mechanics/AssociativePlasticHardening.h"
#include "models/solid_mechanics/PerzynaPlasticFlowRate.h"
#include "models/solid_mechanics/PlasticStrainRate.h"
#include "models/ImplicitTimeIntegration.h"
#include "models/ImplicitUpdate.h"
#include "models/IdentityMap.h"
#include "models/forces/QuasiStaticForce.h"
#include "models/forces/ForceRate.h"
#include "solvers/NewtonNonlinearSolver.h"
#include "UniaxialStrainStructuralDriver.h"

TEST_CASE("Uniaxial strain regression test", "[StructuralRegressionTests]")
{
  TorchSize nbatch = 20;
  Scalar E = 1e5;
  Scalar nu = 0.3;
  Scalar s0 = 5;
  Scalar K = 1000;
  Scalar eta = 100;
  Scalar n = 2;
  auto Erate = ForceRate<SymR2>("total_strain");
  auto Eerate = ElasticStrainRate("elastic_strain_rate");
  auto elasticity = LinearIsotropicElasticity<true>("elasticity", E, nu);
  auto kinharden = NoKinematicHardening("kinematic_hardening");
  auto isoharden = LinearIsotropicHardening("isotropic_hardening", s0, K);
  auto yield = J2IsotropicYieldFunction("yield_function");
  auto direction = AssociativePlasticFlowDirection("plastic_flow_direction", yield);
  auto eprate = AssociativePlasticHardening("ep_rate", yield);
  auto hrate = PerzynaPlasticFlowRate("hardening_rate", eta, n);
  auto Eprate = PlasticStrainRate("plastic_strain_rate");

  // All these dependency registration thingy can be predefined.
  auto rate = ComposedModel("rate");
  rate.registerModel(Erate);
  rate.registerModel(Eerate);
  rate.registerModel(elasticity);
  rate.registerModel(kinharden);
  rate.registerModel(isoharden);
  rate.registerModel(yield);
  rate.registerModel(direction);
  rate.registerModel(hrate);
  rate.registerModel(eprate);
  rate.registerModel(Eprate);
  rate.registerDependency("total_strain", "elastic_strain_rate");
  rate.registerDependency("kinematic_hardening", "yield_function");
  rate.registerDependency("isotropic_hardening", "yield_function");
  rate.registerDependency("kinematic_hardening", "plastic_flow_direction");
  rate.registerDependency("isotropic_hardening", "plastic_flow_direction");
  rate.registerDependency("kinematic_hardening", "ep_rate");
  rate.registerDependency("isotropic_hardening", "ep_rate");
  rate.registerDependency("hardening_rate", "ep_rate");
  rate.registerDependency("yield_function", "hardening_rate");
  rate.registerDependency("hardening_rate", "plastic_strain_rate");
  rate.registerDependency("plastic_flow_direction", "plastic_strain_rate");
  rate.registerDependency("plastic_strain_rate", "elastic_strain_rate");
  rate.registerDependency("elastic_strain_rate", "elasticity");

  auto implicit_rate = ImplicitTimeIntegration("implicit_time_integration", rate);
  auto solver = NewtonNonlinearSolver({/*atol =*/1e-10,
                                       /*rtol =*/1e-8,
                                       /*miters =*/100,
                                       /*verbose=*/false});
  auto model = ImplicitUpdate("viscoplasticity", implicit_rate, solver);

  TorchSize nsteps = 100;
  Real max_strain = 0.10;
  Real min_time = -1;
  Real max_time = 5;
  Scalar max_strains = torch::full({nbatch}, max_strain, TorchDefaults).unsqueeze(-1);
  Scalar end_times = torch::logspace(min_time, max_time, nbatch, 10, TorchDefaults).unsqueeze(-1);
  UniaxialStrainStructuralDriver driver(model, max_strains, end_times, nsteps);
  auto [all_inputs, all_outputs] = driver.run();

  std::ofstream ofile;
  std::string fname = "regression/models/solid_mechanics/" + model.name();

  // I use this to write csv for visualization purposes.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // for (TorchSize batch = 0; batch < nbatch; batch++)
  // {
  //   ofile.open(fname + "_forces_batch_" + utils::stringify(batch) + ".csv");
  //   for (size_t i = 0; i < all_inputs.size(); i++)
  //     LabeledVector(all_inputs[i].slice(0, "forces")).write(ofile, ",", batch, i == 0);
  //   ofile.close();

  //   ofile.open(fname + "_state_batch_" + utils::stringify(batch) + ".csv");
  //   for (size_t i = 0; i < all_outputs.size(); i++)
  //     LabeledVector(all_outputs[i].slice(0, "state")).write(ofile, ",", batch, i == 0);
  //   ofile.close();
  // }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  auto inputs =
      torch::zeros(utils::add_shapes(TorchShape({nsteps}), all_inputs[0].tensor().sizes()));
  auto outputs =
      torch::zeros(utils::add_shapes(TorchShape({nsteps}), all_outputs[0].tensor().sizes()));
  for (TorchSize i = 0; i < nsteps; i++)
  {
    inputs.index_put_({i, torch::indexing::Ellipsis}, all_inputs[i].tensor());
    outputs.index_put_({i, torch::indexing::Ellipsis}, all_outputs[i].tensor());
  }

  // Below is what I used to save the results
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // torch::save(inputs, fname + "_inputs.pt");
  // torch::save(outputs, fname + "_outputs.pt");
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Load it back
  auto inputs_ref =
      torch::zeros(utils::add_shapes(TorchShape({nsteps}), all_inputs[0].tensor().sizes()));
  auto outputs_ref =
      torch::zeros(utils::add_shapes(TorchShape({nsteps}), all_outputs[0].tensor().sizes()));
  torch::load(inputs_ref, fname + "_inputs.pt");
  torch::load(outputs_ref, fname + "_outputs.pt");

  REQUIRE(torch::allclose(inputs, inputs_ref));
  REQUIRE(torch::allclose(outputs, outputs_ref));
}
