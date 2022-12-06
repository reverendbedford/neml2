#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <iostream>
#include <fstream>
#include <streambuf>

#include "TestUtils.h"
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

TEST_CASE("A Model can output the function graph in DOT format", "[Model]")
{
  Scalar E = 1e5;
  Scalar nu = 0.3;
  Scalar s0 = 5;
  Scalar K = 1000;
  Scalar eta = 100;
  Scalar n = 2;
  auto Erate = ForceRate<SymR2>("total_strain");
  auto Eerate = ElasticStrainRate("elastic_strain_rate");
  auto elasticity = LinearIsotropicElasticity<true>("elasticity", E, nu);
  auto kinharden = NoKinematicHardening("kinematic_hardening");
  auto isoharden = LinearIsotropicHardening("isotropic_hardening", s0, K);
  auto yield = J2IsotropicYieldFunction("yield_function");
  auto direction = AssociativePlasticFlowDirection("plastic_flow_direction", yield);
  auto eprate = AssociativePlasticHardening("ep_rate", yield);
  auto hrate = PerzynaPlasticFlowRate("hardening_rate", eta, n);
  auto Eprate = PlasticStrainRate("plastic_strain_rate");

  // All these dependency registration thingy can be predefined.
  auto rate = ComposedModel("rate");
  rate.registerModel(Erate);
  rate.registerModel(Eerate);
  rate.registerModel(elasticity);
  rate.registerModel(kinharden);
  rate.registerModel(isoharden);
  rate.registerModel(yield);
  rate.registerModel(direction);
  rate.registerModel(hrate);
  rate.registerModel(eprate);
  rate.registerModel(Eprate);
  rate.registerDependency("total_strain", "elastic_strain_rate");
  rate.registerDependency("kinematic_hardening", "yield_function");
  rate.registerDependency("isotropic_hardening", "yield_function");
  rate.registerDependency("kinematic_hardening", "plastic_flow_direction");
  rate.registerDependency("isotropic_hardening", "plastic_flow_direction");
  rate.registerDependency("kinematic_hardening", "ep_rate");
  rate.registerDependency("isotropic_hardening", "ep_rate");
  rate.registerDependency("hardening_rate", "ep_rate");
  rate.registerDependency("yield_function", "hardening_rate");
  rate.registerDependency("hardening_rate", "plastic_strain_rate");
  rate.registerDependency("plastic_flow_direction", "plastic_strain_rate");
  rate.registerDependency("plastic_strain_rate", "elastic_strain_rate");
  rate.registerDependency("elastic_strain_rate", "elasticity");

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
