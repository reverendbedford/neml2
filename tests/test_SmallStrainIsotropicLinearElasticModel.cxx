#include <catch2/catch.hpp>

#include "SmallStrainIsotropicLinearElasticModel.h"
#include "SymSymR4.h"
#include "SymR2.h"
#include "ElasticityTensors.h"
#include "BatchedSymSymR4.h"

TEST_CASE("Stress update for linear elastic model is correct",
          "[SmallStrainIsotropicLinearElasticModel]")
{

  Scalar E = 100.0;
  Scalar nu = 0.3;

  SmallStrainIsotropicLinearElasticModel model(E, nu);

  int nbatch = 10;

  // Fairly arbitrary strains and previous stress...
  BatchedSymR2 strain_np1(torch::repeat_interleave(
      torch::tensor({0.25, 0.0, -0.05, 0.15, 0.1, -0.5}, TorchDefaults).reshape({1, 6}),
      nbatch,
      0));
  BatchedSymR2 strain_n(torch::repeat_interleave(
      torch::tensor({0.1, 0.05, 0.5, -0.075, 0.7, -0.1}, TorchDefaults).reshape({1, 6}),
      nbatch,
      0));
  BatchedSymR2 stress_n(torch::repeat_interleave(
      torch::tensor({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, TorchDefaults).reshape({1, 6}), nbatch, 0));

  // Actual state objects
  State state_n(model.state(), stress_n);
  State forces_np1(model.forces(), strain_np1);
  State forces_n(model.forces(), strain_n);

  SECTION("Stress update is right")
  {
    State state_np1 = model.state_update(forces_np1, state_n, forces_n);
    // Calculated in numpy...
    BatchedSymR2 stress_np1(torch::repeat_interleave(
        torch::tensor({30.76923077, 11.53846154, 7.69230769, 11.53846154, 7.69230769, -38.46153846},
                      TorchDefaults)
            .reshape({1, 6}),
        nbatch,
        0));
    REQUIRE(torch::allclose(state_np1, stress_np1));
  }

  SECTION("Tangent stiffness is right")
  {
    // Calculated in numpy
    BatchedSymSymR4 correct(torch::repeat_interleave(
        torch::tensor({{134.61538462, 57.69230769, 57.69230769, 0., 0., 0.},
                       {57.69230769, 134.61538462, 57.69230769, 0., 0., 0.},
                       {57.69230769, 57.69230769, 134.61538462, 0., 0., 0.},
                       {0., 0., 0., 76.92307692, 0., 0.},
                       {0., 0., 0., 0., 76.92307692, 0.},
                       {0., 0., 0., 0., 0., 76.92307692}},
                      TorchDefaults)
            .reshape({1, 6, 6}),
        nbatch,
        0));
    StateDerivative tangent = model.linearized_state_update(forces_np1, state_n, forces_n);
    REQUIRE(torch::allclose(tangent, correct));
  }
}
