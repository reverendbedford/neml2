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
#include <memory>

#include "TestUtils.h"
#include "neml2/models/ComposedModel.h"
#include "neml2/models/solid_mechanics/ElasticStrain.h"
#include "neml2/models/solid_mechanics/LinearElasticity.h"
#include "neml2/models/solid_mechanics/IsotropicMandelStress.h"
#include "neml2/models/solid_mechanics/LinearIsotropicHardening.h"
#include "neml2/models/solid_mechanics/J2StressMeasure.h"
#include "neml2/models/solid_mechanics/YieldFunction.h"
#include "neml2/models/solid_mechanics/AssociativePlasticFlowDirection.h"
#include "neml2/models/solid_mechanics/PerzynaPlasticFlowRate.h"
#include "neml2/models/solid_mechanics/PlasticStrainRate.h"
#include "neml2/models/ImplicitTimeIntegration.h"
#include "neml2/models/ImplicitUpdate.h"
#include "neml2/models/IdentityMap.h"
#include "neml2/solvers/NewtonNonlinearSolver.h"
#include "SampleSubsubaxisModel.h"

using namespace neml2;

TEST_CASE("Linear DAG", "[ComposedModel][Linear DAG]")
{
  TorchSize nbatch = 10;
  Scalar E(100, nbatch);
  Scalar nu(0.3, nbatch);
  SymSymR4 C = SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {E, nu});
  auto estrain = std::make_shared<ElasticStrain>("elastic_strain");
  auto elasticity = std::make_shared<CauchyStressFromElasticStrain>("elasticity", C);
  auto mandel_stress = std::make_shared<IsotropicMandelStress>("mandel_stress");

  // inputs --> "elastic_strain" --> "elasticity" --> "mandel_stress" --> outputs
  // inputs: total strain, plastic strain
  // outputs: mandel stress
  auto model = ComposedModel("foo", {estrain, elasticity, mandel_stress});

  SECTION("model definition")
  {
    REQUIRE(model.input().has_subaxis("forces"));
    REQUIRE(model.input().subaxis("forces").has_variable<SymR2>("total_strain"));
    REQUIRE(model.input().has_subaxis("state"));
    REQUIRE(model.input().subaxis("state").has_variable<SymR2>("plastic_strain"));
    REQUIRE(model.output().has_subaxis("state"));
    REQUIRE(model.output().subaxis("state").has_variable<SymR2>("mandel_stress"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, model.input());
    auto E = SymR2::init(0.1, 0.05, 0).batch_expand(nbatch);
    auto Ep = SymR2::init(0.01, 0.01, 0).batch_expand(nbatch);
    in.slice("forces").set(E, "total_strain");
    in.slice("state").set(Ep, "plastic_strain");

    auto exact = model.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, model.output(), model.input());
    finite_differencing_derivative(
        [model](const LabeledVector & x) { return model.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}

TEST_CASE("Y-junction DAG", "[ComposedModel][Y-junction DAG]")
{
  TorchSize nbatch = 10;
  Scalar s0 = 100.0;
  Scalar K = 1000.0;
  auto isoharden = std::make_shared<LinearIsotropicHardening>("isotropic_hardening", K);
  auto mandel_stress = std::make_shared<IsotropicMandelStress>("mandel_stress");
  auto sm = std::make_shared<J2StressMeasure>("stress_measure");
  auto yield = std::make_shared<YieldFunction>("yield_function", sm, s0, true, false);

  // inputs --> "isotropic_hardening" --
  //                                    `
  //                                     `--> "yield_function" --> outputs
  //                                     '
  //                                    '
  // inputs -> "mandel_stress" ---
  //
  // inputs: cauchy stress, equivalent plastic strain
  // outputs: yield function
  auto model = ComposedModel("foo", {isoharden, mandel_stress, yield});

  SECTION("model definition")
  {
    REQUIRE(model.input().has_subaxis("state"));
    REQUIRE(model.input().subaxis("state").has_variable<SymR2>("cauchy_stress"));
    REQUIRE(model.input().subaxis("state").has_subaxis("internal_state"));
    REQUIRE(model.input()
                .subaxis("state")
                .subaxis("internal_state")
                .has_variable<Scalar>("equivalent_plastic_strain"));

    REQUIRE(model.output().has_subaxis("state"));
    REQUIRE(model.output().subaxis("state").has_variable<Scalar>("yield_function"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, model.input());
    auto S = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(S, "cauchy_stress");
    in.slice("state").slice("internal_state").set(Scalar(0.1, nbatch), "equivalent_plastic_strain");

    auto exact = model.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, model.output(), model.input());
    finite_differencing_derivative(
        [model](const LabeledVector & x) { return model.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}

TEST_CASE("diamond pattern", "[ComposedModel]")
{
  TorchSize nbatch = 10;
  Scalar eta = 150;
  Scalar n = 6;
  Scalar s0 = 10.0;
  auto sm = std::make_shared<J2StressMeasure>("stress_measure");
  auto yield = std::make_shared<YieldFunction>("yield_function", sm, s0, true, false);
  auto direction =
      std::make_shared<AssociativePlasticFlowDirection>("plastic_flow_direction", yield);
  auto eprate = std::make_shared<PerzynaPlasticFlowRate>("plastic_flow_rate", eta, n);
  auto Eprate = std::make_shared<PlasticStrainRate>("plastic_strain_rate");

  auto model = ComposedModel("foo", {yield, direction, eprate, Eprate});

  SECTION("model definition")
  {
    REQUIRE(model.input().has_subaxis("state"));
    REQUIRE(model.input().subaxis("state").has_variable<SymR2>("mandel_stress"));
    REQUIRE(model.input().subaxis("state").has_subaxis("hardening_interface"));
    REQUIRE(model.input()
                .subaxis("state")
                .subaxis("hardening_interface")
                .has_variable<Scalar>("isotropic_hardening"));

    REQUIRE(model.output().has_subaxis("state"));
    REQUIRE(model.output().subaxis("state").has_variable<SymR2>("plastic_strain_rate"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, model.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(M, "mandel_stress");
    in.slice("state").slice("hardening_interface").set(Scalar(200, nbatch), "isotropic_hardening");

    auto exact = model.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, model.output(), model.input());
    finite_differencing_derivative(
        [model](const LabeledVector & x) { return model.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}

TEST_CASE("send to different device", "[ComposedModel]")
{
  Scalar eta = 150;
  Scalar n = 6;
  Scalar s0 = 15.0;
  auto sm = std::make_shared<J2StressMeasure>("stress_measure");
  auto yield = std::make_shared<YieldFunction>("yield_function", sm, s0, true, false);
  auto direction =
      std::make_shared<AssociativePlasticFlowDirection>("plastic_flow_direction", yield);
  auto eprate = std::make_shared<PerzynaPlasticFlowRate>("plastic_flow_rate", eta, n);
  auto Eprate = std::make_shared<PlasticStrainRate>("plastic_strain_rate");

  auto model = ComposedModel("foo", {yield, direction, eprate, Eprate});
  auto params = model.named_parameters();

  SECTION("send to CPU")
  {
    model.to(torch::kCPU);
    for (const auto & param : params)
      REQUIRE(param.value().device().type() == torch::kCPU);
  }

  SECTION("send to CUDA")
  {
    if (torch::cuda::is_available())
    {
      model.to(torch::kCUDA);
      for (const auto & param : params)
        REQUIRE(param.value().device().type() == torch::kCUDA);
    }
  }
}

TEST_CASE("A model with sub-sub-axis", "[ComposedModel][Sub-sub-axis]")
{
  auto sample = std::make_shared<SampleSubsubaxisModel>("saple");
  auto out = std::make_shared<IdentityMap<Scalar>>(
      "o1", std::vector<std::string>{"state", "baz"}, std::vector<std::string>{"state", "wow"});
  auto model = ComposedModel("whatever", {sample, out});

  SECTION("model definition")
  {
    REQUIRE(model.input().has_subaxis("state"));
    REQUIRE(model.input().subaxis("state").has_variable<Scalar>("foo"));
    REQUIRE(model.input().subaxis("state").has_subaxis("substate"));
    REQUIRE(model.input().subaxis("state").subaxis("substate").has_variable<Scalar>("bar"));
    REQUIRE(model.output().has_subaxis("state"));
    REQUIRE(model.output().subaxis("state").has_variable<Scalar>("wow"));
  }

  SECTION("model derivatives")
  {
    TorchSize nbatch = 10;
    LabeledVector in(nbatch, model.input());
    in.slice("state").set(Scalar(5.6, nbatch), "foo");
    in.slice("state").slice("substate").set(Scalar(-111, nbatch), "bar");

    auto exact = model.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, model.output(), model.input());
    finite_differencing_derivative(
        [model](const LabeledVector & x) { return model.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
