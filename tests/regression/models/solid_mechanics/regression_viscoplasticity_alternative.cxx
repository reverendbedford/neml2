#include <catch2/catch.hpp>

#include <fstream>

#include "models/ComposedModel.h"
#include "models/solid_mechanics/ElasticStrain.h"
#include "models/solid_mechanics/LinearElasticity.h"
#include "models/solid_mechanics/IsotropicMandelStress.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"
#include "models/solid_mechanics/J2IsotropicYieldFunction.h"
#include "models/solid_mechanics/AssociativePlasticFlowDirection.h"
#include "models/solid_mechanics/AssociativeIsotropicPlasticHardening.h"
#include "models/solid_mechanics/PerzynaPlasticFlowRate.h"
#include "models/solid_mechanics/PlasticStrainRate.h"
#include "models/ImplicitTimeIntegration.h"
#include "models/ImplicitUpdate.h"
#include "models/TimeIntegration.h"
#include "models/IdentityMap.h"
#include "models/ForceRate.h"
#include "solvers/NewtonNonlinearSolver.h"
#include "StructuralDriver.h"
#include "misc/math.h"

using namespace neml2;

TEST_CASE("Alternative composition of viscoplasticity", "[viscoplasticity alternative]")
{
  NonlinearSolverParameters params = {/*atol =*/1e-10,
                                      /*rtol =*/1e-8,
                                      /*miters =*/100,
                                      /*verbose=*/false};

  Scalar E = 1e5;
  Scalar nu = 0.3;
  SymSymR4 C = SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {E, nu});
  Scalar s0 = 5;
  Scalar K = 1000;
  Scalar eta = 100;
  Scalar n = 2;
  auto Ee = std::make_shared<ElasticStrain>("elastic_strain");
  auto S = std::make_shared<CauchyStressFromElasticStrain>("cauchy_stress", C);
  auto M = std::make_shared<IsotropicMandelStress>("mandel_stress");
  auto gamma = std::make_shared<LinearIsotropicHardening>("isotropic_hardening", K);
  auto f = std::make_shared<J2IsotropicYieldFunction>("yield_function", s0);
  auto gammarate = std::make_shared<PerzynaPlasticFlowRate>("hardening_rate", eta, n);
  auto Np = std::make_shared<AssociativePlasticFlowDirection>("plastic_flow_direction", f);
  auto eprate = std::make_shared<AssociativeIsotropicPlasticHardening>("ep_rate", f);
  auto Eprate = std::make_shared<PlasticStrainRate>("plastic_strain_rate");

  auto rate = std::make_shared<ComposedModel>(
      "rate",
      std::vector<std::shared_ptr<Model>>{Ee, S, M, gamma, f, gammarate, Np, eprate, Eprate});

  auto surface = std::make_shared<ImplicitTimeIntegration>("yield_surface", rate);
  auto solver = std::make_shared<NewtonNonlinearSolver>(params);
  auto return_map = std::make_shared<ImplicitUpdate>("return_map", surface, solver);
  auto strain =
      std::make_shared<IdentityMap<SymR2>>("total_strain",
                                           std::vector<std::string>{"forces", "total_strain"},
                                           std::vector<std::string>{"forces", "total_strain"});
  auto output_Ep =
      std::make_shared<IdentityMap<SymR2>>("output_plastic_strain",
                                           std::vector<std::string>{"state", "plastic_strain"},
                                           std::vector<std::string>{"state", "plastic_strain"});
  auto output_ep = std::make_shared<IdentityMap<Scalar>>(
      "output_equivalent_plastic_strain",
      std::vector<std::string>{"state", "internal_state", "equivalent_plastic_strain"},
      std::vector<std::string>{"state", "internal_state", "equivalent_plastic_strain"});

  auto model = std::make_shared<ComposedModel>(
      "viscoplasticity",
      std::vector<std::shared_ptr<Model>>{return_map, strain, Ee, S, output_Ep, output_ep},
      std::vector<std::shared_ptr<Model>>{return_map, strain},
      std::vector<std::shared_ptr<Model>>{output_Ep, output_ep});

  TorchSize nbatch = 1;
  TorchSize nsteps = 100;
  Real max_strain = 0.10;
  Real min_time = -1;
  Real max_time = 5;

  Scalar end_time = torch::logspace(min_time, max_time, nbatch, 10, TorchDefaults).unsqueeze(-1);
  SymR2 end_strain =
      SymR2::init(max_strain, -0.5 * max_strain, -0.5 * max_strain).batch_expand(nbatch);

  BatchTensor<1> times = math::linspace<1>(torch::zeros_like(end_time), end_time, nsteps);
  BatchTensor<1> strains = math::linspace<1>(torch::zeros_like(end_strain), end_strain, nsteps);

  StructuralDriver driver(*model, times, strains, "total_strain");
  auto [all_inputs, all_outputs] = driver.run();

  std::ofstream ofile;
  std::string fname = "regression/models/solid_mechanics/viscoplasticity_alternative";

  // I use this to write csv for visualization purposes.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // for (TorchSize batch = 0; batch < nbatch; batch++)
  // {
  //   ofile.open(fname + "_forces_batch_" + utils::stringify(batch) + ".csv");
  //   for (size_t i = 0; i < all_inputs.size(); i++)
  //     LabeledVector(all_inputs[i].slice("forces")).write(ofile, ",", batch, i == 0);
  //   ofile.close();

  //   ofile.open(fname + "_state_batch_" + utils::stringify(batch) + ".csv");
  //   for (size_t i = 0; i < all_outputs.size(); i++)
  //     LabeledVector(all_outputs[i].slice("state")).write(ofile, ",", batch, i == 0);
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
  //   torch::save(inputs, fname + "_inputs.pt");
  //   torch::save(outputs, fname + "_outputs.pt");
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
