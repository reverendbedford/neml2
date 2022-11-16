#include <catch2/catch.hpp>

#include "SmallStrainIsotropicLinearElasticModel.h"
#include "SymR2.h"
#include "SymSymR4.h"

TEST_CASE("Stress update for linear elastic model is correct",
          "[SmallStrainIsotropicLinearElasticModel]")
{

  Scalar E = 100.0;
  Scalar nu = 0.3;

  SmallStrainIsotropicLinearElasticModel model(E, nu);

  int nbatch = 10;

  // Fairly arbitrary strains and previous stress...
  SymR2 strain_np1(torch::tensor({0.25, 0.0, -0.05, 0.15, 0.1, -0.5}, TorchDefaults), nbatch);
  SymR2 strain_n(torch::tensor({0.1, 0.05, 0.5, -0.075, 0.7, -0.1}, TorchDefaults), nbatch);
  SymR2 stress_n(torch::tensor({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, TorchDefaults), nbatch);
  Scalar time_np1(1, nbatch);
  Scalar time_n(0, nbatch);

  // Actual state objects
  State state_n(model.state(), stress_n);

  State forces_np1(model.forces(), nbatch);
  forces_np1.set<SymR2>("strain", strain_np1);
  forces_np1.set<Scalar>("time", time_np1);

  State forces_n(model.forces(), nbatch);
  forces_n.set<SymR2>("strain", strain_n);
  forces_n.set<Scalar>("time", time_n);

  SECTION("Stress update is right")
  {
    State state_np1 = model.state_update(forces_np1, state_n, forces_n);
    // Calculated in numpy...
    SymR2 stress_np1(
        torch::tensor({30.76923077, 11.53846154, 7.69230769, 11.53846154, 7.69230769, -38.46153846},
                      TorchDefaults),
        nbatch);
    REQUIRE(torch::allclose(state_np1.tensor(), stress_np1));
  }

  // Calculated in numpy
  // The last column is the derivative w.r.t. time, which is, surprisingly, zero.
  SymSymR4 correct(torch::tensor({{134.61538462, 57.69230769, 57.69230769, 0., 0., 0.},
                                  {57.69230769, 134.61538462, 57.69230769, 0., 0., 0.},
                                  {57.69230769, 57.69230769, 134.61538462, 0., 0., 0.},
                                  {0., 0., 0., 76.92307692, 0., 0.},
                                  {0., 0., 0., 0., 76.92307692, 0.},
                                  {0., 0., 0., 0., 0., 76.92307692}}),
                   nbatch);

  SECTION("Tangent stiffness is right")
  {
    StateDerivativeOutput tangent = model.linearized_state_update(forces_np1, state_n, forces_n);
    REQUIRE(
        torch::allclose(tangent[0][StateDerivative::derivative_name("stress", "strain")], correct));
  }
}
