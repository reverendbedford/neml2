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
  auto & factory = Factory::get_factory();
  factory.clear();

  // inputs --> "elastic_strain" --> "elasticity" --> "mandel_stress" --> outputs
  // inputs: total strain, plastic strain
  // outputs: mandel stress
  factory.create_object("Models",
                        ElasticStrain::expected_params() +
                            ParameterSet(KS{"name", "estrain"}, KS{"type", "ElasticStrain"}));
  factory.create_object("Models",
                        CauchyStressFromElasticStrain::expected_params() +
                            ParameterSet(KS{"name", "elasticity"},
                                         KS{"type", "CauchyStressFromElasticStrain"},
                                         KR{"E", 100},
                                         KR{"nu", 0.3}));
  factory.create_object(
      "Models",
      IsotropicMandelStress::expected_params() +
          ParameterSet(KS{"name", "mandel"}, KS{"type", "IsotropicMandelStress"}));
  factory.create_object("Models",
                        ComposedModel::expected_params() +
                            ParameterSet(KS{"name", "foo"},
                                         KS{"type", "ComposedModel"},
                                         KVS{"models", {"estrain", "elasticity", "mandel"}}));

  auto & model = Factory::get_object<ComposedModel>("Models", "foo");

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
    TorchSize nbatch = 10;
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
  auto & factory = Factory::get_factory();
  factory.clear();

  // inputs --> "isotropic_hardening" --
  //                                    `
  //                                     `
  //                                      ----> "yield_function" ----> outputs
  //                                     /
  //                                    /
  // inputs --> "mandel_stress" --------
  //
  // inputs: cauchy stress, equivalent plastic strain
  // outputs: yield function
  factory.create_object("Models",
                        LinearIsotropicHardening::expected_params() +
                            ParameterSet(KS{"name", "isoharden"},
                                         KS{"type", "LinearIsotropicHardening"},
                                         KR{"K", 1000}));
  factory.create_object(
      "Models",
      IsotropicMandelStress::expected_params() +
          ParameterSet(KS{"name", "mandel"}, KS{"type", "IsotropicMandelStress"}));
  factory.create_object("Models",
                        J2StressMeasure::expected_params() +
                            ParameterSet(KS{"name", "j2"}, KS{"type", "J2StressMeasure"}));
  factory.create_object("Models",
                        IsotropicHardeningYieldFunction::expected_params() +
                            ParameterSet(KS{"name", "yield"},
                                         KS{"type", "IsotropicHardeningYieldFunction"},
                                         KS{"stress_measure", "j2"},
                                         KR{"yield_stress", 100}));
  factory.create_object("Models",
                        ComposedModel::expected_params() +
                            ParameterSet(KS{"name", "foo"},
                                         KS{"type", "ComposedModel"},
                                         KVS{"models", {"isoharden", "mandel", "yield"}}));

  auto & model = Factory::get_object<ComposedModel>("Models", "foo");

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
    TorchSize nbatch = 10;
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
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        J2StressMeasure::expected_params() +
                            ParameterSet(KS{"name", "j2"}, KS{"type", "J2StressMeasure"}));
  factory.create_object("Models",
                        IsotropicHardeningYieldFunction::expected_params() +
                            ParameterSet(KS{"name", "yield"},
                                         KS{"type", "IsotropicHardeningYieldFunction"},
                                         KS{"stress_measure", "j2"},
                                         KR{"yield_stress", 10}));
  factory.create_object("Models",
                        AssociativePlasticFlowDirection::expected_params() +
                            ParameterSet(KS{"name", "direction"},
                                         KS{"type", "AssociativePlasticFlowDirection"},
                                         KS{"yield_function", "yield"}));
  factory.create_object("Models",
                        PerzynaPlasticFlowRate::expected_params() +
                            ParameterSet(KS{"name", "eprate"},
                                         KS{"type", "PerzynaPlasticFlowRate"},
                                         KR{"eta", 150},
                                         KR{"n", 6}));
  factory.create_object("Models",
                        PlasticStrainRate::expected_params() +
                            ParameterSet(KS{"name", "Eprate"}, KS{"type", "PlasticStrainRate"}));
  factory.create_object(
      "Models",
      ComposedModel::expected_params() +
          ParameterSet(KS{"name", "foo"},
                       KS{"type", "ComposedModel"},
                       KVS{"models", {"yield", "direction", "eprate", "Eprate"}}));

  auto & model = Factory::get_object<ComposedModel>("Models", "foo");

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
    TorchSize nbatch = 10;
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
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        J2StressMeasure::expected_params() +
                            ParameterSet(KS{"name", "j2"}, KS{"type", "J2StressMeasure"}));
  factory.create_object("Models",
                        IsotropicHardeningYieldFunction::expected_params() +
                            ParameterSet(KS{"name", "yield"},
                                         KS{"type", "IsotropicHardeningYieldFunction"},
                                         KS{"stress_measure", "j2"},
                                         KR{"yield_stress", 10}));
  factory.create_object("Models",
                        AssociativePlasticFlowDirection::expected_params() +
                            ParameterSet(KS{"name", "direction"},
                                         KS{"type", "AssociativePlasticFlowDirection"},
                                         KS{"yield_function", "yield"}));
  factory.create_object("Models",
                        PerzynaPlasticFlowRate::expected_params() +
                            ParameterSet(KS{"name", "eprate"},
                                         KS{"type", "PerzynaPlasticFlowRate"},
                                         KR{"eta", 150},
                                         KR{"n", 6}));
  factory.create_object("Models",
                        PlasticStrainRate::expected_params() +
                            ParameterSet(KS{"name", "Eprate"}, KS{"type", "PlasticStrainRate"}));
  factory.create_object(
      "Models",
      ComposedModel::expected_params() +
          ParameterSet(KS{"name", "foo"},
                       KS{"type", "ComposedModel"},
                       KVS{"models", {"yield", "direction", "eprate", "Eprate"}}));

  auto & model = Factory::get_object<ComposedModel>("Models", "foo");
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
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object(
      "Models",
      SampleSubsubaxisModel::expected_params() +
          ParameterSet(KS{"name", "sample"}, KS{"type", "SampleSubsubaxisModel"}));
  factory.create_object("Models",
                        ScalarIdentityMap::expected_params() +
                            ParameterSet(KS{"name", "out"},
                                         KS{"type", "ScalarIdentityMap"},
                                         KVS{"from_var", {"state", "baz"}},
                                         KVS{"to_var", {"state", "wow"}}));
  factory.create_object("Models",
                        ComposedModel::expected_params() +
                            ParameterSet(KS{"name", "whatever"},
                                         KS{"type", "ComposedModel"},
                                         KVS{"models", {"sample", "out"}}));

  auto & model = Factory::get_object<ComposedModel>("Models", "whatever");

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
