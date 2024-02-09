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

#include "neml2/solvers/TrustRegionNonlinearSolver.h"
#include <iomanip>
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(TrustRegionNonlinearSolver);

OptionSet
TrustRegionNonlinearSolver::expected_options()
{
  OptionSet options = NonlinearSolver::expected_options();
  options.set<Real>("delta_0") = 1.0;
  options.set<Real>("delta_max") = 10.0;
  options.set<Real>("reduce_criteria") = 0.25;
  options.set<Real>("expand_criteria") = 0.75;
  options.set<Real>("reduce_factor") = 0.25;
  options.set<Real>("expand_factor") = 2.0;
  options.set<Real>("accept_criteria") = 0.1;
  return options;
}

TrustRegionNonlinearSolver::TrustRegionNonlinearSolver(const OptionSet & options)
  : NonlinearSolver(options),
    _delta_0(options.get<Real>("delta_0")),
    _delta_max(options.get<Real>("delta_max")),
    _reduce_criteria(options.get<Real>("reduce_criteria")),
    _expand_criteria(options.get<Real>("expand_criteria")),
    _reduce_factor(options.get<Real>("reduce_factor")),
    _expand_factor(options.get<Real>("expand_factor")),
    _accept_criteria(options.get<Real>("accept_criteria"))
{
}

BatchTensor
TrustRegionNonlinearSolver::solve(const NonlinearSystem & system, const BatchTensor & x0) const
{
  // Setup initial guess, initial residual, and initial trust region
  auto x = x0.clone();
  auto [R, J] = system.residual_and_Jacobian(x);
  auto nR0 = torch::linalg::vector_norm(R, 2, -1, false, c10::nullopt);
  auto nR = nR0.clone();
  auto delta = Scalar::full(x.batch_sizes(), _delta_0, x.options());

  // Continuing iterating until one of:
  // 1. nR < atol (success)
  // 2. nR / nR0 < rtol (success)
  // 3. i > miters (failure)
  for (size_t i = 1; i < miters; i++)
  {
    // Check for convergence
    if (converged(i, nR, nR0, torch::min(delta).item<Real>()))
      return x;

    // Approximately solve the trust region subproblem and return the trial step
    auto p = solve_subproblem(R, J, delta);

    // Figure out the quality of the subproblem solution compared to the quadratic model
    auto xp = x + system.scale_direction(p);
    auto [Rp, Jp] = system.residual_and_Jacobian(xp);
    auto red_a = TrustRegionNonlinearSolver::merit_reduction(p, Jp);
    auto red_b = TrustRegionNonlinearSolver::merit_reduction(p, J);
    auto rho = red_a / red_b;

    // Adjust the trust region
    delta.batch_index_put({rho < _reduce_criteria},
                          _reduce_factor * delta.batch_index({rho < _reduce_criteria}));
    delta.batch_index_put({rho > _expand_criteria},
                          _expand_factor * delta.batch_index({rho > _expand_criteria}));

    // Accept or reject the current step
    auto accept = (rho >= _accept_criteria);
    x = BatchTensor(torch::where(accept.unsqueeze(-1), xp, x), x.batch_dim());
    R = BatchTensor(torch::where(accept.unsqueeze(-1), Rp, R), R.batch_dim());
    J = BatchTensor(torch::where(accept.unsqueeze(-1).unsqueeze(-1), Jp, J), J.batch_dim());
    nR = torch::linalg::vector_norm(R, 2, -1, false, c10::nullopt);
  }

  // Throw if we exceeded miters
  throw NEMLException("Nonlinear solver exceeded miters!");

  return x;
}

bool
TrustRegionNonlinearSolver::converged(size_t itr,
                                      const torch::Tensor & nR,
                                      const torch::Tensor & nR0,
                                      Real min_trust) const
{
  // LCOV_EXCL_START
  if (verbose)
    std::cout << "ITERATION " << std::setw(3) << itr << ", |R| = " << std::scientific
              << torch::max(nR).item<Real>() << ", |R0| = " << std::scientific
              << torch::max(nR0).item<Real>() << ", trust = " << std::scientific << min_trust
              << std::endl;
  // LCOV_EXCL_STOP

  return torch::all(torch::logical_or(nR < atol, nR / nR0 < rtol)).item<bool>();
}

BatchTensor
TrustRegionNonlinearSolver::solve_subproblem(const BatchTensor & R,
                                             const BatchTensor & J,
                                             const Scalar & delta) const
{
  // The full Newton step
  auto p_newton = -BatchTensor(torch::linalg::solve(J, R, true), R.batch_dim());
  // The trust region step
  auto s = scalar_newton([&, R, J, delta](BatchTensor x)
                         { return TrustRegionNonlinearSolver::subproblem(x, R, J, delta); },
                         Scalar::zeros(R.batch_sizes(), R.dtype()));
  s = BatchTensor(torch::maximum(s, torch::zeros_like(s)), s.batch_dim());
  auto p_trust =
      -TrustRegionNonlinearSolver::JJsJ_product(J, s, math::bmv(J.base_transpose(0, 1), R));

  // Now select between the two...
  return BatchTensor(
      torch::where(
          (torch::linalg::vector_norm(p_newton, 2, -1, false, c10::nullopt) <= delta).unsqueeze(-1),
          p_newton,
          p_trust),
      p_newton.batch_dim());
}

BatchTensor
TrustRegionNonlinearSolver::scalar_newton(
    std::function<std::tuple<BatchTensor, BatchTensor>(const BatchTensor &)> RJ,
    const BatchTensor & x0) const
{
  // Setup
  auto x = x0.clone();
  auto [R, J] = RJ(x);
  auto nR = abs(R);
  auto nR0 = nR;

  // Continuing iterating until one of:
  // 1. nR < atol (success)
  // 2. nR / nR0 < rtol (success)
  // 3. i > miters (failure), but return anyway
  // Should these be different from the main system tolerances?
  for (size_t i = 1; i < miters; i++)
  {
    // Do some printing if verbose
    if (verbose)
      std::cout << "SUBPROBLEM ITERATION " << std::setw(3) << i << ", |R| = " << std::scientific
                << torch::max(nR).item<Real>() << ", |R0| = " << std::scientific
                << torch::max(nR0).item<Real>() << std::endl;

    // Check for convergence
    if (torch::all(torch::logical_or(nR < atol, nR / nR0 < rtol)).item<bool>())
      break;

    x -= R / J;
    std::tie(R, J) = RJ(x);
    nR = abs(R);
  }

  return x;
}

BatchTensor
TrustRegionNonlinearSolver::JJsJ_product(const BatchTensor & J,
                                         const Scalar & sigma,
                                         const BatchTensor & v)
{
  return BatchTensor(
      torch::linalg::solve(math::bmm(J.base_transpose(0, 1), J) +
                               sigma * BatchTensor::identity(J.base_sizes()[0], J.dtype()),
                           v,
                           true),
      sigma.batch_dim());
}

std::tuple<BatchTensor, BatchTensor>
TrustRegionNonlinearSolver::subproblem(const Scalar & s,
                                       const BatchTensor & R,
                                       const BatchTensor & J,
                                       const Scalar & delta)
{
  auto p = -TrustRegionNonlinearSolver::JJsJ_product(J, s, math::bmv(J.base_transpose(0, 1), R));
  auto np = BatchTensor(torch::linalg::vector_norm(p, 2, -1, false, c10::nullopt), p.batch_dim());
  auto Ri = 1.0 / np - 1.0 / delta;
  auto Ji = 1.0 / math::pow(np, 3.0) * math::bvv(p, JJsJ_product(J, s, p));
  return {Ri, Ji};
}

Scalar
TrustRegionNonlinearSolver::merit_reduction(const BatchTensor & p, const BatchTensor & J)
{
  return 0.5 * math::bvv(p, math::bmv(J, p));
}

} // namespace neml2
