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

#include "neml2/solvers/NewtonWithLineSearch.h"
#include <iomanip>
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(NewtonWithLineSearch);

OptionSet
NewtonWithLineSearch::expected_options()
{
  OptionSet options = Newton::expected_options();
  options.set<unsigned int>("max_linesearch_iterations") = 10;
  options.set<Real>("linesearch_cutback") = 2.0;
  options.set<Real>("linesearch_stopping_criteria") = 1.0e-3;
  return options;
}

NewtonWithLineSearch::NewtonWithLineSearch(const OptionSet & options)
  : Newton(options),
    _linesearch_miter(options.get<unsigned int>("max_linesearch_iterations")),
    _linesearch_sigma(options.get<Real>("linesearch_cutback")),
    _linesearch_c(options.get<Real>("linesearch_stopping_criteria"))
{
}

void
NewtonWithLineSearch::update(NonlinearSystem & system, BatchTensor & x)
{
  auto dx = solve_direction(system);

  linesearch(system, x, dx);
  x.variable_data() += system.scale_direction(_alpha * dx);
  system.set_solution(x);
}

void
NewtonWithLineSearch::linesearch(NonlinearSystem & system,
                                 const BatchTensor & x,
                                 const BatchTensor & dx)
{
  _alpha = Scalar::ones(x.batch_sizes(), x.options());

  const auto & R = system.residual_view();
  auto R0 = R.clone();
  auto nR02 = math::bvv(R0, R0);

  for (size_t i = 1; i < _linesearch_miter; i++)
  {
    system.set_solution(x + system.scale_direction(_alpha * dx));
    system.residual();
    auto nR2 = math::bvv(R, R);
    auto crit = nR02 + 2.0 * _linesearch_c * _alpha * math::bvv(R0, dx);
    if (verbose)
      std::cout << "     LS ITERATION " << std::setw(3) << i << ", alpha = " << std::scientific
                << torch::min(_alpha).item<Real>() << ", |R| = " << std::scientific
                << torch::max(torch::sqrt(nR2)).item<Real>() << ", |Rc| = " << std::scientific
                << torch::min(torch::sqrt(crit)).item<Real>() << std::endl;

    auto stop = torch::logical_or(nR2 <= crit, nR2 <= std::pow(atol, 2));

    if (torch::all(stop).item<bool>())
      break;

    _alpha.batch_index_put({torch::logical_not(stop)},
                           _alpha.batch_index({torch::logical_not(stop)}) / _linesearch_sigma);
  }
}

} // namespace neml2
