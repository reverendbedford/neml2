// Copyright 2023, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy 
// of this software and associated documentation files (the "Software"), to deal 
// in the Software without restriction, including without limitation the rights 
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
// copies of the Software, and to permit persons to whom the Software is 
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
// THE SOFTWARE.


#include <catch2/catch.hpp>

#include <fstream>

#include "neml2/models/ComposedModel.h"
#include "neml2/models/solid_mechanics/TotalStrain.h"
#include "neml2/models/solid_mechanics/LinearElasticity.h"
#include "neml2/models/solid_mechanics/IsotropicMandelStress.h"
#include "neml2/models/solid_mechanics/LinearIsotropicHardening.h"
#include "neml2/models/solid_mechanics/J2StressMeasure.h"
#include "neml2/models/solid_mechanics/YieldFunction.h"
#include "neml2/models/solid_mechanics/AssociativePlasticFlowDirection.h"
#include "neml2/models/solid_mechanics/AssociativeIsotropicPlasticHardening.h"
#include "neml2/models/solid_mechanics/PerzynaPlasticFlowRate.h"
#include "neml2/models/solid_mechanics/PlasticStrainRate.h"
#include "neml2/models/ImplicitTimeIntegration.h"
#include "neml2/models/ImplicitUpdate.h"
#include "neml2/models/IdentityMap.h"
#include "neml2/models/TimeIntegration.h"
#include "neml2/models/ForceRate.h"
#include "neml2/solvers/NewtonNonlinearSolver.h"
#include "StructuralDriver.h"
#include "neml2/misc/math.h"

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
  auto sm = std::make_shared<J2StressMeasure>("stress_measure");
  auto yieldfunc = std::make_shared<YieldFunction>("yield_function", sm, s0, true, false);
  auto hrate = std::make_shared<PerzynaPlasticFlowRate>("hrate", eta, n);
  auto eprate = std::make_shared<AssociativeIsotropicPlasticHardening>("ep_rate", yieldfunc);
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
  auto Ep = std::make_shared<TimeIntegration<SymR2>>("plastic_strain");
  auto Ee = std::make_shared<ElasticStrainFromCauchyStress>("elastic_strain", S);
  auto strain = std::make_shared<TotalStrain>("total_strain");

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
                                          Ep,
                                          Ee,
                                          strain},
      std::vector<LabeledAxisAccessor>{Ep->var, isoharden->equivalent_plastic_strain});

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
