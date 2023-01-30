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


#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>

#include "neml2/models/ComposedModel.h"
#include "neml2/models/solid_mechanics/AssociativeIsotropicPlasticHardening.h"
#include "neml2/models/solid_mechanics/ElasticStrain.h"
#include "neml2/models/solid_mechanics/LinearElasticity.h"
#include "neml2/models/solid_mechanics/IsotropicMandelStress.h"
#include "neml2/models/solid_mechanics/VoceIsotropicHardening.h"
#include "neml2/models/solid_mechanics/ChabochePlasticHardening.h"
#include "neml2/models/solid_mechanics/J2StressMeasure.h"
#include "neml2/models/solid_mechanics/IsotropicAndKinematicHardeningYieldFunction.h"
#include "neml2/models/solid_mechanics/AssociativePlasticFlowDirection.h"
#include "neml2/models/solid_mechanics/AssociativeKinematicPlasticHardening.h"
#include "neml2/models/solid_mechanics/PerzynaPlasticFlowRate.h"
#include "neml2/models/solid_mechanics/PlasticStrainRate.h"
#include "neml2/models/ImplicitTimeIntegration.h"
#include "neml2/models/ImplicitUpdate.h"
#include "neml2/models/ForceRate.h"
#include "neml2/models/SumModel.h"
#include "neml2/solvers/NewtonNonlinearSolver.h"
#include "StructuralDriver.h"
#include "neml2/misc/math.h"

using namespace neml2;

TEST_CASE("Benchmark Chaboche", "[BENCHMARK][Chaboche]")
{
  std::vector<TorchSize> nbatches = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  std::vector<TorchSize> nbackstresses = {2, 4, 6, 8, 10};

  // Time stepping
  TorchSize nsteps = 100;
  Real max_strain = 0.10;
  Real min_time = -1;
  Real max_time = 5;

  for (TorchSize nbatch : nbatches)
  {
    Scalar end_time = torch::logspace(min_time, max_time, nbatch, 10, TorchDefaults).unsqueeze(-1);
    SymR2 end_strain =
        SymR2::init(max_strain, -0.5 * max_strain, -0.5 * max_strain).batch_expand(nbatch);

    BatchTensor<1> times = math::linspace<1>(torch::zeros_like(end_time), end_time, nsteps);
    BatchTensor<1> strains = math::linspace<1>(torch::zeros_like(end_strain), end_strain, nsteps);

    for (TorchSize nbackstress : nbackstresses)
    {
      NonlinearSolverParameters params = {/*atol =*/1e-10,
                                          /*rtol =*/1e-8,
                                          /*miters =*/100,
                                          /*verbose=*/false};

      // Model parameters
      Scalar E = 1e5;
      Scalar nu = 0.3;
      SymSymR4 C = SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {E, nu});
      Scalar s0 = 5;
      Scalar R = 100;
      Scalar d = 1.2;
      Scalar eta = 100;
      Scalar n = 4.0;

      // Model building blocks
      auto Erate = std::make_shared<ForceRate<SymR2>>("total_strain");
      auto Eerate = std::make_shared<ElasticStrainRate>("elastic_strain_rate");
      auto elasticity = std::make_shared<CauchyStressRateFromElasticStrainRate>("elasticity", C);
      auto mandel_stress = std::make_shared<IsotropicMandelStress>("mandel_stress");
      auto isoharden = std::make_shared<VoceIsotropicHardening>("isotropic_hardening", R, d);
      auto sm = std::make_shared<J2StressMeasure>("stress_measure");
      auto yield =
          std::make_shared<IsotropicAndKinematicHardeningYieldFunction>("yield_function", sm, s0);
      auto direction =
          std::make_shared<AssociativePlasticFlowDirection>("plastic_flow_direction", yield);
      auto eeprate = std::make_shared<AssociativeIsotropicPlasticHardening>("eeprate", yield);
      auto hrate = std::make_shared<PerzynaPlasticFlowRate>("hardening_rate", eta, n);
      auto Eprate = std::make_shared<PlasticStrainRate>("plastic_strain_rate");
      std::vector<std::shared_ptr<Model>> models = {Erate,
                                                    Eerate,
                                                    elasticity,
                                                    mandel_stress,
                                                    isoharden,
                                                    yield,
                                                    direction,
                                                    eeprate,
                                                    hrate,
                                                    Eprate};

      // Backstresses
      std::vector<std::vector<std::string>> bs_names;
      for (unsigned int i = 0; i < nbackstress; i++)
      {
        Scalar C = 1000 * i;
        Scalar g = 10 * i;
        Scalar A = 1e-10 * i;
        Scalar a = 1 + 0.2 * i;
        auto bs = std::make_shared<ChabochePlasticHardening>(
            "chaboche_bs_" + utils::stringify(i), C, g, A, a, "_" + utils::stringify(i));
        bs_names.push_back(bs->backstress.item_names);
        models.push_back(bs);
      }

      // Kinematic hardening is the sum of all backstresses
      auto kinharden = std::make_shared<SumModel<SymR2>>(
          "kinharden",
          bs_names,
          std::vector<std::string>{"state", "hardening_interface", "kinematic_hardening"});
      models.push_back(kinharden);

      // Composition
      auto rate = std::make_shared<ComposedModel>("rate", models);
      auto implicit_rate =
          std::make_shared<ImplicitTimeIntegration>("implicit_time_integration", rate);
      auto solver = std::make_shared<NewtonNonlinearSolver>(params);
      auto model = std::make_shared<ImplicitUpdate>("viscoplasticity", implicit_rate, solver);

      StructuralDriver driver(*model, times, strains, "total_strain");
      BENCHMARK("{" + utils::stringify(nbatch) + "} " + utils::stringify(nbackstress) +
                " backstresses")
      {
        torch::NoGradGuard no_grad_guard;
        driver.run();
      };
    }
  }
}
