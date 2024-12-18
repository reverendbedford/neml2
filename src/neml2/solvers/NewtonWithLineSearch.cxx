// Copyright 2024, UChicago Argonne, LLC
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
  options.doc() = "The Newton-Raphson solver with line search.";

  options.set<std::string>("linesearch_type") = "backtracking";

  options.set<unsigned int>("max_linesearch_iterations") = 10;
  options.set("max_linesearch_iterations").doc() =
      "Maximum allowable linesearch iterations. No error is produced upon reaching the maximum "
      "number of iterations, and the scale factor in the last iteration is used to scale the step.";

  options.set<Real>("linesearch_cutback") = 2.0;
  options.set("linesearch_cutback").doc() = "Linesearch cut-back factor when the current scale "
                                            "factor cannot sufficiently reduce the residual.";

  options.set<Real>("linesearch_stopping_criteria") = 1.0e-3;
  options.set("linesearch_stopping_criteria").doc() =
      "The lineseach tolerance slightly relaxing the definition of residual decrease";

  return options;
}

NewtonWithLineSearch::NewtonWithLineSearch(const OptionSet & options)
  : Newton(options),
    _linesearch_miter(options.get<unsigned int>("max_linesearch_iterations")),
    _linesearch_sigma(options.get<Real>("linesearch_cutback")),
    _linesearch_c(options.get<Real>("linesearch_stopping_criteria")),
    _type(options.get<std::string>("linesearch_type"))
{
}

void
NewtonWithLineSearch::update(NonlinearSystem & system,
                             NonlinearSystem::Sol<true> & x,
                             const NonlinearSystem::Res<true> & r,
                             const NonlinearSystem::Jac<true> & J)
{
  auto dx = solve_direction(r, J);
  auto alpha = linesearch(system, x, dx, r);
  x = NonlinearSystem::Sol<true>(x.variable_data() + alpha * Tensor(dx));
}

Scalar
NewtonWithLineSearch::linesearch(NonlinearSystem & system,
                                 const NonlinearSystem::Sol<true> & x,
                                 const NonlinearSystem::Sol<true> & dx,
                                 const NonlinearSystem::Res<true> & R0) const
{
  auto alpha = Scalar::ones(x.batch_sizes(), x.options());
  const auto nR02 = math::bvv(R0, R0);
  bool check = false;
  bool flag = false;
  auto crit = nR02;

  for (std::size_t i = 1; i < _linesearch_miter; i++)
  {
    NonlinearSystem::Sol<true> xp(Tensor(x) + alpha * Tensor(dx));
    auto R = system.residual(xp);
    auto nR2 = math::bvv(R, R);

    if (_type == "backtracking")
      crit = nR02 + 2.0 * _linesearch_c * alpha * math::bvv(R0, dx);
    else if (_type == "strong_wolfe")
      crit = (1.0 - _linesearch_c * alpha) * nR02;
    else
      neml_assert(false, "Line Search type '", _type, "' has not yet been implemented");

    // std::cout << "nR02: " << nR02.item<Real>() << std::endl;
    // std::cout << "math::bvv(R0, dx): " << math::bvv(R0, dx).item<Real>() << std::endl;
    // std::cout << "R: \n" << R << std::endl;
    // std::cout << "nR2: " << nR2.item<Real>() << std::endl;

    if (std::isnan(torch::max(nR2).item<Real>()))
      neml_assert(false, "One of the residual componenet is NAN");

    if (verbose)
      std::cout << "     LS ITERATION " << std::setw(3) << i << ", min(alpha) = " << std::scientific
                << torch::min(alpha).item<Real>() << ", max(||R||) = " << std::scientific
                << torch::max(math::sqrt(nR2)).item<Real>() << ", min(||Rc||) = " << std::scientific
                << torch::min(math::sqrt(crit)).item<Real>() << std::endl;

    auto stop = Scalar(torch::logical_or(nR2 <= crit, nR2 <= std::pow(atol, 2)));

    if (torch::max(crit).item<Real>() < 0)
      flag = true;

    if (torch::all(stop).item<bool>())
    {
      check = true;
      break;
    }

    alpha = alpha.batch_expand_as(stop).clone();
    alpha.batch_index_put_({torch::logical_not(stop)},
                           alpha.batch_index({torch::logical_not(stop)}) / _linesearch_sigma);
  }

  if (flag)
    neml_assert(check,
                "NOnlinear Solver failed to converge: Line Search produces negative stopping "
                "criteria, try with other "
                "linesearch_type, increase linesearch_cutback "
                "or reduce linesearch_stopping_criteria");

  return alpha;
}

} // namespace neml2
