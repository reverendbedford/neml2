#include <catch2/catch.hpp>

#include <fstream>

#include "models/ComposedModel.h"
#include "models/solid_mechanics/AssociativeIsotropicPlasticHardening.h"
#include "models/solid_mechanics/ElasticStrain.h"
#include "models/solid_mechanics/LinearElasticity.h"
#include "models/solid_mechanics/IsotropicMandelStress.h"
#include "models/solid_mechanics/VoceIsotropicHardening.h"
#include "models/solid_mechanics/ChabochePlasticHardening.h"
#include "models/solid_mechanics/J2StressMeasure.h"
#include "models/solid_mechanics/IsotropicAndKinematicHardeningYieldFunction.h"
#include "models/solid_mechanics/AssociativePlasticFlowDirection.h"
#include "models/solid_mechanics/AssociativeKinematicPlasticHardening.h"
#include "models/solid_mechanics/PerzynaPlasticFlowRate.h"
#include "models/solid_mechanics/PlasticStrainRate.h"
#include "models/ImplicitTimeIntegration.h"
#include "models/ImplicitUpdate.h"
#include "models/ForceRate.h"
#include "models/SumModel.h"
#include "models/IdentityMap.h"
#include "solvers/NewtonNonlinearSolver.h"
#include "StructuralDriver.h"
#include "misc/math.h"

using namespace neml2;

TEST_CASE("Chaboche regression", "[Chaboche]")
{
  NonlinearSolverParameters params = {/*atol =*/1e-10,
                                      /*rtol =*/1e-8,
                                      /*miters =*/100,
                                      /*verbose=*/false};

  Scalar E = 1e5;
  Scalar nu = 0.3;
  SymSymR4 C = SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {E, nu});
  Scalar s0 = 5;
  Scalar R = 100;
  Scalar d = 1.2;

  Scalar C1 = 10000.0;
  Scalar g1 = 100.0;
  Scalar A1 = 1.0e-8;
  Scalar a1 = 1.2;

  Scalar C2 = 1000.0;
  Scalar g2 = 9.0;
  Scalar A2 = 1.0e-10;
  Scalar a2 = 3.2;

  Scalar eta = 100;
  Scalar n = 4.0;

  auto Erate = std::make_shared<ForceRate<SymR2>>("total_strain");
  auto Eerate = std::make_shared<ElasticStrainRate>("elastic_strain_rate");
  auto elasticity = std::make_shared<CauchyStressRateFromElasticStrainRate>("elasticity", C);
  auto mandel_stress = std::make_shared<IsotropicMandelStress>("mandel_stress");
  auto isoharden = std::make_shared<VoceIsotropicHardening>("isotropic_hardening", R, d);

  auto sm = std::make_shared<J2StressMeasure>("stress_measure");
  auto yield =
      std::make_shared<IsotropicAndKinematicHardeningYieldFunction>("yield_function", sm, s0);

  std::vector<std::string> bs_name_1{"state", "internal_state", "backstress_1"};
  std::vector<std::string> bs_name_2{"state", "internal_state", "backstress_2"};

  auto bs1_value = std::make_shared<IdentityMap<SymR2>>("bs1_value", bs_name_1, bs_name_1);
  auto bs2_value = std::make_shared<IdentityMap<SymR2>>("bs2_value", bs_name_2, bs_name_2);

  auto bs1 = std::make_shared<ChabochePlasticHardening>("chaboche_1", C1, g1, A1, a1, yield, "_1");
  auto bs2 = std::make_shared<ChabochePlasticHardening>("chaboche_2", C2, g2, A2, a2, yield, "_2");

  std::vector<std::vector<std::string>> bs_names({bs_name_1, bs_name_2});
  std::vector<std::string> kinname({"state", "hardening_interface", "kinematic_hardening"});

  auto kinharden = std::make_shared<SumModel<SymR2>>("kinharden", bs_names, kinname);

  auto direction =
      std::make_shared<AssociativePlasticFlowDirection>("plastic_flow_direction", yield);
  auto eeprate = std::make_shared<AssociativeIsotropicPlasticHardening>("eeprate", yield);
  auto hrate = std::make_shared<PerzynaPlasticFlowRate>("hardening_rate", eta, n);
  auto Eprate = std::make_shared<PlasticStrainRate>("plastic_strain_rate");

  auto rate = std::make_shared<ComposedModel>(
      "rate",
      std::vector<std::shared_ptr<Model>>{Erate,
                                          Eerate,
                                          elasticity,
                                          mandel_stress,
                                          isoharden,
                                          bs1_value,
                                          bs2_value,
                                          bs1,
                                          bs2,
                                          kinharden,
                                          yield,
                                          direction,
                                          eeprate,
                                          hrate,
                                          Eprate},
      std::vector<std::shared_ptr<Model>>{
          Erate, bs1_value, bs2_value, kinharden, mandel_stress, isoharden});

  auto implicit_rate = std::make_shared<ImplicitTimeIntegration>("implicit_time_integration", rate);
  auto solver = std::make_shared<NewtonNonlinearSolver>(params);
  auto model = std::make_shared<ImplicitUpdate>("viscoplasticity", implicit_rate, solver);

  TorchSize nbatch = 20;
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
  std::string fname = "regression/models/solid_mechanics/chaboche";

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
