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

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "SampleNonlinearSystems.h"
#include "neml2/solvers/Newton.h"
#include "neml2/solvers/NewtonWithLineSearch.h"
#include "neml2/solvers/NewtonWithTrustRegion.h"

using namespace neml2;

using solver_types = std::tuple<Newton, NewtonWithLineSearch, NewtonWithTrustRegion>;

TEMPLATE_LIST_TEST_CASE("NonlinearSolvers", "[solvers]", solver_types)
{
  // System shape
  TorchShape batch_sz = {2};
  TorchSize nbase = 4;

  // Create the nonlinear solver
  OptionSet options = TestType::expected_options();
  options.set<bool>("verbose") = false;
  TestType solver(options);

  SECTION("solve")
  {
    SECTION("power")
    {
      // Initial guess
      auto x = BatchTensor::full(batch_sz, nbase, 2.0, default_tensor_options());

      // Create the nonlinear system
      auto options = PowerTestSystem::expected_options();
      PowerTestSystem system(options);
      system.reinit(x);

      auto [succeeded, iters] = solver.solve(system, x);

      REQUIRE(succeeded);
      REQUIRE(torch::allclose(x, system.exact_solution()));
    }

    SECTION("Rosenbrock")
    {
      // Initial guess
      auto x = BatchTensor::full(batch_sz, nbase, 0.75, default_tensor_options());

      // Create the nonlinear system
      auto options = RosenbrockTestSystem::expected_options();
      PowerTestSystem system(options);
      system.reinit(x);

      auto [succeeded, iters] = solver.solve(system, x);

      REQUIRE(succeeded);
      REQUIRE(torch::allclose(x, system.exact_solution()));
    }
  }

  SECTION("automatic scaling")
  {
    // Initial guess
    auto x = BatchTensor::full(batch_sz, nbase, 2.0, default_tensor_options());

    // Create the nonlinear system (with automatic scaling)
    auto options = PowerTestSystem::expected_options();
    options.set<bool>("automatic_scaling") = true;
    PowerTestSystem system(options);
    system.reinit(x);

    system.init_scaling(solver.verbose);
    auto [succeeded, iters] = solver.solve(system, x);

    REQUIRE(succeeded);
    REQUIRE(torch::allclose(x, system.exact_solution()));
  }
}
