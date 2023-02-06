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

TEST_CASE("Solve system correctly", "[NewtonNonlinearSolver]")
{
  TorchSize nbatch = 2;
  TorchSize n = 4;
  BatchTensor<1> x0(nbatch, n);

  PowerTestSystem system;
  x0 = system.guess(x0);

  ParameterSet params = NewtonNonlinearSolver::expected_params();
  params.set<std::string>("name") = "solver";
  NewtonNonlinearSolver solver(params);

  auto x_res = solver.solve(system, x0);

  REQUIRE(torch::allclose(x_res, system.exact_solution(x0)));
}
