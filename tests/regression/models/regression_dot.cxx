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

#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <iostream>
#include <fstream>
#include <streambuf>

#include "TestUtils.h"
#include "neml2/models/ComposedModel.h"
#include "neml2/models/solid_mechanics/ElasticStrain.h"
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
#include "neml2/models/ForceRate.h"
#include "neml2/solvers/NewtonNonlinearSolver.h"

using namespace neml2;

TEST_CASE("A Model can output the function graph in DOT format", "[DOT]")
{
  Scalar E = 1e5;
  Scalar nu = 0.3;
  SymSymR4 C = SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {E, nu});
  Scalar s0 = 5;
  Scalar K = 1000;
  Scalar eta = 100;
  Scalar n = 2;
  auto Erate = std::make_shared<ForceRate<SymR2>>("total_strain");
  auto Eerate = std::make_shared<ElasticStrainRate>("elastic_strain_rate");
  auto elasticity = std::make_shared<CauchyStressRateFromElasticStrainRate>("elasticity", C);
  auto mandel_stress = std::make_shared<IsotropicMandelStress>("mandel_stress");
  auto isoharden = std::make_shared<LinearIsotropicHardening>("isotropic_hardening", K);
  auto sm = std::make_shared<J2StressMeasure>("stress_measure");
  auto yield = std::make_shared<YieldFunction>("yield_function", sm, s0, true, false);
  auto direction =
      std::make_shared<AssociativePlasticFlowDirection>("plastic_flow_direction", yield);
  auto eprate = std::make_shared<AssociativeIsotropicPlasticHardening>("ep_rate", yield);
  auto hrate = std::make_shared<PerzynaPlasticFlowRate>("hardening_rate", eta, n);
  auto Eprate = std::make_shared<PlasticStrainRate>("plastic_strain_rate");

  auto rate = ComposedModel("rate",
                            {Erate,
                             Eerate,
                             elasticity,
                             mandel_stress,
                             isoharden,
                             yield,
                             direction,
                             eprate,
                             hrate,
                             Eprate});

  // Write the gold file
  // std::ofstream ogold("regression/models/regression_dot.txt");;
  // rate.to_dot(ogold);

  // Read the gold file
  std::ifstream gold("regression/models/regression_dot.txt");
  REQUIRE(gold.is_open());
  std::string correct((std::istreambuf_iterator<char>(gold)), std::istreambuf_iterator<char>());

  // The output shall match the gold
  std::ostringstream oss;
  rate.to_dot(oss);
  std::string mine = oss.str();

  REQUIRE(mine == correct);
}
