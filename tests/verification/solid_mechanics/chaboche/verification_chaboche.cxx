#include <catch2/catch.hpp>

#include "models/ComposedModel.h"
#include "models/IdentityMap.h"
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
#include "solvers/NewtonNonlinearSolver.h"
#include "StructuralDriver.h"
#include "misc/math.h"

#include "VerificationTest.h"

using namespace neml2;

TEST_CASE("Chaboche verification", "[StructuralVerificationTests]")
{
  // NL solver parameters
  NonlinearSolverParameters params = {/*atol =*/1e-10,
                                      /*rtol =*/1e-8,
                                      /*miters =*/100,
                                      /*verbose=*/false};

  // Make the model -- we need serialization...
  Scalar E = 1e5;
  Scalar nu = 0.3;
  SymSymR4 C = SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {E, nu});
  Scalar s0 = 10.0;
  Scalar R = 50.0;
  Scalar d = 1.2;

  // Note the factors on g and A
  Scalar C1 = 5000.0;
  Scalar g1 = 10.1 * sqrt(2.0 / 3.0);
  Scalar A1 = 1.0e-6 * sqrt(3.0 / 2.0);
  Scalar a1 = 1.2;

  Scalar C2 = 1000.0;
  Scalar g2 = 5.2 * sqrt(2.0 / 3.0);
  Scalar A2 = 1.0e-10 * sqrt(3.0 / 2.0);
  Scalar a2 = 3.2;

  // NEML has a different definition of eta...
  Scalar eta = 200 * std::pow(sqrt(2.0 / 3.0), 1.0 / 4.0) * sqrt(2.0 / 3.0);
  Scalar n = 4.0;

  auto Erate = std::make_shared<ForceRate<SymR2>>("total_strain");
  auto Eerate = std::make_shared<ElasticStrainRate>("elastic_strain_rate");
  auto elasticity = std::make_shared<CauchyStressRateFromElasticStrainRate>("elasticity", C);
  auto mandel_stress = std::make_shared<IsotropicMandelStress>("mandel_stress");
  auto isoharden = std::make_shared<VoceIsotropicHardening>("isotropic_hardening", R, d);

  auto sm = std::make_shared<J2StressMeasure>("stress_measure");
  auto yield =
      std::make_shared<IsotropicAndKinematicHardeningYieldFunction>("yield_function", sm, s0);

  std::vector<std::string> bs_name_1({"state", "internal_state", "backstress_1"});
  std::vector<std::string> bs_name_2({"state", "internal_state", "backstress_2"});

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

  SECTION("Uniaxial slow deformation")
  {
    // Load and run the test
    std::string fname = "verification/solid_mechanics/chaboche/chaboche.vtest";
    VerificationTest test(fname);
    REQUIRE(test.compare(*model));
  }
}
