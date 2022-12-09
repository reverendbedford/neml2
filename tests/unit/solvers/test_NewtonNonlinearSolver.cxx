#include <catch2/catch.hpp>

#include "SampleNonlinearSystems.h"
#include "solvers/NewtonNonlinearSolver.h"

using namespace neml2;

// Loop over a number of tests, at least at some point
TEST_CASE("Solve system correctly", "[NewtonNonlinearSolver]")
{
  TorchSize nbatch = 2;
  TorchSize n = 4;
  BatchTensor<1> x0(nbatch, n);

  PowerTestSystem system;
  x0 = system.guess(x0);

  NonlinearSolverParameters params;
  NewtonNonlinearSolver solver(params);

  auto x_res = solver.solve(system, x0);

  REQUIRE(torch::allclose(x_res, system.exact_solution(x0)));
}
