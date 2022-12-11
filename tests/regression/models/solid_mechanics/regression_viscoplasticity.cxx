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

using namespace neml2;

TEST_CASE("Uniaxial strain regression test", "[StructuralRegressionTests]")
{
  NonlinearSolverParameters params = {/*atol =*/1e-10,
                                      /*rtol =*/1e-8,
                                      /*miters =*/100,
                                      /*verbose=*/false};

  TorchSize nbatch = 20;
  Scalar E = 1e5;
  Scalar nu = 0.3;
  Scalar s0 = 5;
  Scalar K = 1000;
  Scalar eta = 100;
  Scalar n = 2;
  auto Erate = std::make_shared<ForceRate<SymR2>>("total_strain");
  auto Eerate = std::make_shared<ElasticStrainRate>("elastic_strain_rate");
  auto elasticity = std::make_shared<LinearIsotropicElasticityRate>("elasticity", E, nu);
  auto kinharden = std::make_shared<NoKinematicHardening>("kinematic_hardening");
  auto isoharden = std::make_shared<LinearIsotropicHardening>("isotropic_hardening", s0, K);
  auto yield = std::make_shared<J2IsotropicYieldFunction>("yield_function");
  auto direction =
      std::make_shared<AssociativePlasticFlowDirection>("plastic_flow_direction", yield);
  auto eprate = std::make_shared<AssociativePlasticHardening>("ep_rate", yield);
  auto hrate = std::make_shared<PerzynaPlasticFlowRate>("hardening_rate", eta, n);
  auto Eprate = std::make_shared<PlasticStrainRate>("plastic_strain_rate");

  // All these dependency registration thingy can be predefined.
  std::vector<std::pair<std::shared_ptr<Model>, std::shared_ptr<Model>>> dependencies = {
      {Erate, Eerate},
      {kinharden, yield},
      {isoharden, yield},
      {kinharden, direction},
      {isoharden, direction},
      {kinharden, eprate},
      {isoharden, eprate},
      {hrate, eprate},
      {yield, hrate},
      {hrate, Eprate},
      {direction, Eprate},
      {Eprate, Eerate},
      {Eerate, elasticity}};
  auto rate = std::make_shared<ComposedModel>("rate", dependencies);

  auto implicit_rate = std::make_shared<ImplicitTimeIntegration>("implicit_time_integration", rate);
  auto solver = std::make_shared<NewtonNonlinearSolver>(params);
  auto model = std::make_shared<ImplicitUpdate>("viscoplasticity", implicit_rate, solver);

  TorchSize nsteps = 100;
  Real max_strain = 0.10;
  Real min_time = -1;
  Real max_time = 5;
  Scalar max_strains = torch::full({nbatch}, max_strain, TorchDefaults).unsqueeze(-1);
  Scalar end_times = torch::logspace(min_time, max_time, nbatch, 10, TorchDefaults).unsqueeze(-1);
  UniaxialStrainStructuralDriver driver(*model, max_strains, end_times, nsteps);
  auto [all_inputs, all_outputs] = driver.run();

  std::ofstream ofile;
  std::string fname = "regression/models/solid_mechanics/" + model->name();

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
  std::cout << "ARG" << std::endl;
  std::cout << inputs.sizes() << std::endl;
  std::cout << outputs.sizes() << std::endl;
  std::cout << all_inputs.size() << std::endl;
  std::cout << all_outputs.size() << std::endl;
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

  std::cout << inputs_ref.sizes() << std::endl;
  std::cout << outputs_ref.sizes() << std::endl;

  REQUIRE(torch::allclose(inputs, inputs_ref));
  REQUIRE(torch::allclose(outputs, outputs_ref));
}
