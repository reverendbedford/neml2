#include <catch2/catch.hpp>
#include <torch/torch.h>

using namespace torch::autograd::forward_ad;

#include "SampleNonlinearSystems.h"

#include "solvers/NewtonNonlinearSolver.h"

// Loop over a number of tests, at least at some point
TEST_CASE("Solve system correctly", "[NewtonNonlinearSolver]")
{
  TorchSize nbatch = 2;
  TorchSize n = 4;

  PowerTestSystem system(nbatch, n);
  torch::Tensor x = system.guess();

  NonlinearSolverParameters params;
  NewtonNonlinearSolver solver(params);

  auto x_res = solver.solve(x, system);

  REQUIRE(torch::allclose(x_res, system.exact_solution()));
}
