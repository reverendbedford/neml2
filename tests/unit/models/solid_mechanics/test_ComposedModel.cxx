#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "models/ComposedModel.h"
#include "models/solid_mechanics/ElasticStrain.h"
#include "models/solid_mechanics/LinearElasticity.h"
#include "models/solid_mechanics/NoKinematicHardening.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"
#include "models/solid_mechanics/J2IsotropicYieldFunction.h"
#include "models/solid_mechanics/AssociativePlasticFlowDirection.h"
#include "models/solid_mechanics/PerzynaPlasticFlowRate.h"
#include "models/solid_mechanics/PlasticStrainRate.h"
#include "models/ImplicitTimeIntegration.h"
#include "models/ImplicitUpdate.h"
#include "models/IdentityMap.h"
#include "solvers/NewtonNonlinearSolver.h"
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
  auto kinharden = std::make_shared<NoKinematicHardening>("kinematic_hardening");

  // inputs --> "elastic_strain" --> "elasticity" --> "kinharden" --> outputs
  // inputs: total strain, plastic strain
  // outputs: mandel stress
  auto model = ComposedModel("foo", {estrain, elasticity, kinharden});

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
  auto isoharden = std::make_shared<LinearIsotropicHardening>("isotropic_hardening", s0, K);
  auto kinharden = std::make_shared<NoKinematicHardening>("kinematic_hardening");
  auto yield = std::make_shared<J2IsotropicYieldFunction>("yield_function");

  // inputs --> "isotropic_hardening" --
  //                                    `
  //                                     `--> "yield_function" --> outputs
  //                                     '
  //                                    '
  // inputs -> "kinematic_hardening" ---
  //
  // inputs: cauchy stress, equivalent plastic strain
  // outputs: yield function
  auto model = ComposedModel("foo", {isoharden, kinharden, yield});

  SECTION("model definition")
  {
    REQUIRE(model.input().has_subaxis("state"));
    REQUIRE(model.input().subaxis("state").has_variable<SymR2>("cauchy_stress"));
    REQUIRE(model.input().subaxis("state").has_variable<Scalar>("equivalent_plastic_strain"));

    REQUIRE(model.output().has_subaxis("state"));
    REQUIRE(model.output().subaxis("state").has_variable<Scalar>("yield_function"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, model.input());
    auto S = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(S, "cauchy_stress");
    in.slice("state").set(Scalar(0.1, nbatch), "equivalent_plastic_strain");

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
  auto yield = std::make_shared<J2IsotropicYieldFunction>("yield_function");
  auto direction =
      std::make_shared<AssociativePlasticFlowDirection>("plastic_flow_direction", yield);
  auto eprate = std::make_shared<PerzynaPlasticFlowRate>("plastic_flow_rate", eta, n);
  auto Eprate = std::make_shared<PlasticStrainRate>("plastic_strain_rate");

  auto model = ComposedModel("foo", {yield, direction, eprate, Eprate});

  SECTION("model definition")
  {
    REQUIRE(model.input().has_subaxis("state"));
    REQUIRE(model.input().subaxis("state").has_variable<SymR2>("mandel_stress"));
    REQUIRE(model.input().subaxis("state").has_variable<Scalar>("isotropic_hardening"));

    REQUIRE(model.output().has_subaxis("state"));
    REQUIRE(model.output().subaxis("state").has_variable<SymR2>("plastic_strain_rate"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, model.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(M, "mandel_stress");
    in.slice("state").set(Scalar(200, nbatch), "isotropic_hardening");

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
  auto yield = std::make_shared<J2IsotropicYieldFunction>("yield_function");
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
  auto out = std::make_shared<IdentityMap<Scalar>>("o1", "state", "baz", "state", "wow");
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
