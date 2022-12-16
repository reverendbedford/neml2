#include <catch2/catch.hpp>

#include <fstream>

#include "models/ComposedModel.h"
#include "models/solid_mechanics/TotalStrain.h"
#include "models/solid_mechanics/LinearElasticity.h"
#include "models/solid_mechanics/IsotropicMandelStress.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"
#include "models/solid_mechanics/J2IsotropicYieldFunction.h"
#include "models/solid_mechanics/AssociativePlasticFlowDirection.h"
#include "models/solid_mechanics/AssociativePlasticHardening.h"
#include "models/solid_mechanics/PerzynaPlasticFlowRate.h"
#include "models/solid_mechanics/PlasticStrainRate.h"
#include "models/ImplicitTimeIntegration.h"
#include "models/ImplicitUpdate.h"
#include "models/IdentityMap.h"
#include "models/TimeIntegration.h"
#include "models/ForceRate.h"
#include "solvers/NewtonNonlinearSolver.h"
#include "StructuralDriver.h"
#include "misc/math.h"

using namespace neml2;

TEST_CASE("Uniaxial stress regression test", "[stress control]")
{
  NonlinearSolverParameters params = {/*atol =*/1e-10,
                                      /*rtol =*/1e-8,
                                      /*miters =*/100,
                                      /*verbose=*/false};

  Scalar E = 1e5;
  Scalar nu = 0.3;
  SymSymR4 S = SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {E, nu}).inverse();
  Scalar s0 = 5;
  Scalar K = 1000;
  Scalar eta = 100;
  Scalar n = 2;

  // The first part:
  // Imput:  [force] cauchy stress
  //         [state] equivalent plastic strain
  // Output: [state] equivalent plastic strain rate
  auto input_stress =
      std::make_shared<IdentityMap<SymR2>>("input_stress",
                                           std::vector<std::string>{"forces", "cauchy_stress"},
                                           std::vector<std::string>{"state", "cauchy_stress"});
  auto mandel_stress = std::make_shared<IsotropicMandelStress>("mandel_stress");
  auto isoharden = std::make_shared<LinearIsotropicHardening>("isotropic_hardening", K);
  auto yieldfunc = std::make_shared<J2IsotropicYieldFunction>("yield_function", s0);
  auto hrate = std::make_shared<PerzynaPlasticFlowRate>("hrate", eta, n);
  auto eprate = std::make_shared<AssociativePlasticHardening>("ep_rate", yieldfunc);
  auto rate = std::make_shared<ComposedModel>(
      "viscoplasticity",
      std::vector<std::shared_ptr<Model>>{
          input_stress, mandel_stress, isoharden, yieldfunc, hrate, eprate});

  // The second part:
  // Imput:  [force] cauchy stress
  //         [force] time
  //         [state] equivalent plastic strain
  //         [old force] cauchy stress
  //         [old force] time
  //         [old state] equivalent plastic strain
  // Output: [state] total strain
  auto yield_surface = std::make_shared<ImplicitTimeIntegration>("yield_surface", rate);
  auto solver = std::make_shared<NewtonNonlinearSolver>(params);
  auto return_map = std::make_shared<ImplicitUpdate>("return_map", yield_surface, solver);
  auto direction = std::make_shared<AssociativePlasticFlowDirection>("flow_direction", yieldfunc);
  auto Eprate = std::make_shared<PlasticStrainRate>("plastic_strain_rate");
  auto stressrate = std::make_shared<ForceRate<SymR2>>("cauchy_stress");
  auto map_stressrate =
      std::make_shared<IdentityMap<SymR2>>("map_stressrate",
                                           std::vector<std::string>{"forces", "cauchy_stress_rate"},
                                           std::vector<std::string>{"state", "cauchy_stress_rate"});
  auto Eerate = std::make_shared<ElasticStrainRateFromCauchyStressRate>("elastic_strain_rate", S);
  auto Erate = std::make_shared<TotalStrainRate>("total_strain_rate");
  auto strain = std::make_shared<TimeIntegration<SymR2>>("total_strain");
  auto output_ep = std::make_shared<IdentityMap<Scalar>>(
      "output_ep",
      std::vector<std::string>{"state", "internal_state", "equivalent_plastic_strain"},
      std::vector<std::string>{"state", "internal_state", "equivalent_plastic_strain"});
  auto input_strain_n =
      std::make_shared<IdentityMap<SymR2>>("input_strain_n",
                                           std::vector<std::string>{"old_state", "total_strain"},
                                           std::vector<std::string>{"old_state", "total_strain"});
  auto input_t_np1 =
      std::make_shared<IdentityMap<Scalar>>("input_t_np1",
                                            std::vector<std::string>{"forces", "time"},
                                            std::vector<std::string>{"forces", "time"});
  auto input_t_n =
      std::make_shared<IdentityMap<Scalar>>("input_t_n",
                                            std::vector<std::string>{"old_forces", "time"},
                                            std::vector<std::string>{"old_forces", "time"});

  auto model = std::make_shared<ComposedModel>(
      "viscoplasticity",
      std::vector<std::shared_ptr<Model>>{input_stress,
                                          return_map,
                                          isoharden,
                                          mandel_stress,
                                          yieldfunc,
                                          direction,
                                          hrate,
                                          Eprate,
                                          stressrate,
                                          map_stressrate,
                                          Eerate,
                                          Erate,
                                          strain,
                                          input_strain_n,
                                          input_t_np1,
                                          input_t_n,
                                          output_ep},
      std::vector<std::shared_ptr<Model>>{
          return_map, input_stress, stressrate, input_strain_n, input_t_np1, input_t_n},
      std::vector<std::shared_ptr<Model>>{output_ep});

  TorchSize nbatch = 20;
  TorchSize nsteps = 100;
  Real max_stress = 120;
  Real min_time = -1;
  Real max_time = 5;

  Scalar end_time = torch::logspace(min_time, max_time, nbatch, 10, TorchDefaults).unsqueeze(-1);
  SymR2 end_stress = SymR2::init(max_stress, 0 * max_stress, 0 * max_stress).batch_expand(nbatch);

  BatchTensor<1> times = math::linspace<1>(torch::zeros_like(end_time), end_time, nsteps);
  BatchTensor<1> stresses = math::linspace<1>(torch::zeros_like(end_stress), end_stress, nsteps);

  StructuralDriver driver(*model, times, stresses, "cauchy_stress");
  auto [all_inputs, all_outputs] = driver.run();

  std::ofstream ofile;
  std::string fname = "regression/models/solid_mechanics/stress_control";

  // I use this to write csv for visualization purposes.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //   for (TorchSize batch = 0; batch < nbatch; batch++)
  //   {
  //     ofile.open(fname + "_forces_batch_" + utils::stringify(batch) + ".csv");
  //     for (size_t i = 0; i < all_inputs.size(); i++)
  //       LabeledVector(all_inputs[i].slice("forces")).write(ofile, ",", batch, i == 0);
  //     ofile.close();

  //     ofile.open(fname + "_state_batch_" + utils::stringify(batch) + ".csv");
  //     for (size_t i = 0; i < all_outputs.size(); i++)
  //       LabeledVector(all_outputs[i].slice("state")).write(ofile, ",", batch, i == 0);
  //     ofile.close();
  //   }
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
