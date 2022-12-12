#include <catch2/catch.hpp>

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

#include "VerificationTest.h"

using namespace neml2;

TEST_CASE("Perzyna viscoplasticity verification tests", "[StructuralVerificationTests]")
{
  // NL solver parameters
  NonlinearSolverParameters params = {/*atol =*/1e-10,
                                      /*rtol =*/1e-8,
                                      /*miters =*/100,
                                      /*verbose=*/false};

  // Make the model -- we need serialization...
  Scalar E = 124000.0;
  Scalar nu = 0.32;
  Scalar s0 = 10.0;
  Scalar K = 5500.0;
  Scalar eta = 500.0;
  Scalar n = 5.0;
  auto Erate = std::make_shared<ForceRate<SymR2>>("total_strain");
  auto Eerate = std::make_shared<ElasticStrainRate>("elastic_strain_rate");
  auto elasticity = std::make_shared<LinearIsotropicElasticityRate>("elasticity", E, nu);
  auto kinharden = std::make_shared<NoKinematicHardening>("kinematic_hardening");
  auto isoharden = std::make_shared<LinearIsotropicHardening>("isotropic_hardening", s0, K);
  auto yield = std::make_shared<J2IsotropicYieldFunction>("yield_function");
  auto direction =
      std::make_shared<AssociativePlasticFlowDirection>("plastic_flow_direction", yield);
  auto eprate = std::make_shared<AssociativePlasticHardening>("ep_rate", yield);
  auto hrate = std::make_shared<PerzynaPlasticFlowRate>("hardening_rate", eta, n);
  auto Eprate = std::make_shared<PlasticStrainRate>("plastic_strain_rate");

  // All these dependency registration thingy can be predefined.
  std::vector<std::pair<std::shared_ptr<Model>, std::shared_ptr<Model>>> dependencies = {
      {Erate, Eerate},
      {kinharden, yield},
      {isoharden, yield},
      {kinharden, direction},
      {isoharden, direction},
      {kinharden, eprate},
      {isoharden, eprate},
      {hrate, eprate},
      {yield, hrate},
      {hrate, Eprate},
      {direction, Eprate},
      {Eprate, Eerate},
      {Eerate, elasticity}};
  auto rate = std::make_shared<ComposedModel>("rate", dependencies);

  auto implicit_rate = std::make_shared<ImplicitTimeIntegration>("implicit_time_integration", rate);
  auto solver = std::make_shared<NewtonNonlinearSolver>(params);
  auto model = std::make_shared<ImplicitUpdate>("viscoplasticity", implicit_rate, solver);

  SECTION("Linear isotropic hardening, uniaxial")
  {
    // Load and run the test
    std::string fname = "verification/solid_mechanics/perzyna/isolinear_uniaxial.vtest";
    VerificationTest test(fname);
    REQUIRE(test.compare(*model));
  }

  SECTION("Linear isotropic hardening, multiaxial")
  {
    // Load and run the test
    std::string fname = "verification/solid_mechanics/perzyna/isolinear_multiaxial.vtest";
    VerificationTest test(fname);
    REQUIRE(test.compare(*model));
  }
}
