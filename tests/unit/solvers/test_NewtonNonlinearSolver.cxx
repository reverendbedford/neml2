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

#include "SampleNonlinearSystems.h"
#include "neml2/solvers/NewtonNonlinearSolver.h"

using namespace neml2;

TEST_CASE("NewtonNonlinearSolver", "[solvers]")
{
  // Create the nonlinear solver
  OptionSet options = NewtonNonlinearSolver::expected_options();
  NewtonNonlinearSolver solver(options);

  // Initial guess
  TorchShape batch_sz = {2};
  TorchSize nbase = 4;
  auto x0 = BatchTensor::full(batch_sz, nbase, 2.0, default_tensor_options());

  SECTION("solve")
  {
    // Create the nonlinear system
    auto options = PowerTestSystem::expected_options();
    PowerTestSystem system(options);
    system.reinit(x0);

    auto [succeeded, iters] = solver.solve(system);

    REQUIRE(succeeded);
    REQUIRE(torch::allclose(system.solution(), system.exact_solution()));
  }

  SECTION("automatic scaling")
  {
    // Create the nonlinear system (with automatic scaling)
    auto options = PowerTestSystem::expected_options();
    options.set<bool>("automatic_scaling") = true;
    PowerTestSystem system(options);
    system.reinit(x0);

    system.init_scaling(solver.verbose);
    auto [succeeded, iters] = solver.solve(system);

    REQUIRE(succeeded);
    REQUIRE(torch::allclose(system.solution(), system.exact_solution()));
  }
}
