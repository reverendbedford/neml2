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

#include "neml2/solvers/NonlinearSystem.h"
#include "neml2/misc/math.h"

namespace neml2
{
OptionSet
NonlinearSystem::expected_options()
{
  OptionSet options;

  options.set<bool>("automatic_scaling") = false;
  options.set("automatic_scaling").doc() =
      "Whether to perform automatic scaling. See neml2::NonlinearSystem::init_scaling for "
      "implementation details.";

  options.set<Real>("automatic_scaling_tol") = 0.01;
  options.set("automatic_scaling_tol").doc() =
      "Tolerance used in iteratively updating the scaling matrices.";

  options.set<unsigned int>("automatic_scaling_miter") = 20;
  options.set("automatic_scaling_miter").doc() =
      "Maximum number of automatic scaling iterations. No error is produced upon reaching the "
      "maximum number of scaling iterations, and the scaling matrices obtained at the last "
      "iteration are used to scale the nonlinear system.";

  return options;
}

void
NonlinearSystem::disable_automatic_scaling(OptionSet & options)
{
  options.set("automatic_scaling").suppressed() = true;
  options.set("automatic_scaling_tol").suppressed() = true;
  options.set("automatic_scaling_miter").suppressed() = true;
}

void
NonlinearSystem::enable_automatic_scaling(OptionSet & options)
{
  options.set("automatic_scaling").suppressed() = false;
  options.set("automatic_scaling_tol").suppressed() = false;
  options.set("automatic_scaling_miter").suppressed() = false;
}

NonlinearSystem::NonlinearSystem(const OptionSet & options)
  : _autoscale(options.get<bool>("automatic_scaling")),
    _autoscale_tol(options.get<Real>("automatic_scaling_tol")),
    _autoscale_miter(options.get<unsigned int>("automatic_scaling_miter")),
    _scaling_matrices_initialized(false)
{
}

void
NonlinearSystem::init_scaling(const bool verbose)
{
  if (!_autoscale)
    return;

  if (_scaling_matrices_initialized)
    return;

  using namespace torch::indexing;

  // First compute the Jacobian
  assemble(false, true);

  auto Jp = _Jacobian.clone();
  _row_scaling = BatchTensor::ones_like(_solution);
  _col_scaling = BatchTensor::ones_like(_solution);

  if (verbose)
    std::cout << "Before automatic scaling cond(J) = " << std::scientific
              << torch::max(torch::linalg_cond(Jp)).item<Real>() << std::endl;

  for (unsigned int itr = 0; itr < _autoscale_miter; itr++)
  {
    // check for convergence
    auto rR = torch::max(torch::abs(1.0 - 1.0 / torch::sqrt(std::get<0>(Jp.max(-1))))).item<Real>();
    auto rC = torch::max(torch::abs(1.0 - 1.0 / torch::sqrt(std::get<0>(Jp.max(-2))))).item<Real>();
    if (verbose)
      std::cout << "ITERATION " << itr << ", ROW ILLNESS = " << std::scientific << rR
                << ", COL ILLNESS = " << std::scientific << rC << std::endl;
    if (rR < _autoscale_tol && rC < _autoscale_tol)
      break;

    // scale rows and columns
    for (TorchSize i = 0; i < _ndof; i++)
    {
      auto ar = 1.0 / torch::sqrt(torch::max(Jp.base_index({i})));
      auto ac = 1.0 / torch::sqrt(torch::max(Jp.base_index({Slice(), i})));
      _row_scaling.base_index({i}) *= ar;
      _col_scaling.base_index({i}) *= ac;
      Jp.base_index({i}) *= ar;
      Jp.base_index({Slice(), i}) *= ac;
    }
  }

  _scaling_matrices_initialized = true;

  if (verbose)
    std::cout << " After automatic scaling cond(J) = " << std::scientific
              << torch::max(torch::linalg_cond(Jp)).item<Real>() << std::endl;
}

BatchTensor
NonlinearSystem::scale_residual(const BatchTensor & r) const
{
  neml_assert_dbg(
      _autoscale == _scaling_matrices_initialized,
      _autoscale ? "Automatic scaling is requested but scaling matrices have not been initialized."
                 : "Automatic scaling is not requested but scaling matrices were initialized.");
  return _row_scaling * r;
}

BatchTensor
NonlinearSystem::scale_Jacobian(const BatchTensor & J) const
{
  neml_assert_dbg(
      _autoscale == _scaling_matrices_initialized,
      _autoscale ? "Automatic scaling is requested but scaling matrices have not been initialized."
                 : "Automatic scaling is not requested but scaling matrices were initialized.");
  return math::bmm(math::bmm(math::base_diag_embed(_row_scaling), J),
                   math::base_diag_embed(_col_scaling));
}

BatchTensor
NonlinearSystem::scale_direction(const BatchTensor & p) const
{
  neml_assert_dbg(
      _autoscale == _scaling_matrices_initialized,
      _autoscale ? "Automatic scaling is requested but scaling matrices have not been initialized."
                 : "Automatic scaling is not requested but scaling matrices were initialized.");
  return _autoscale ? _col_scaling * p : p;
}

void
NonlinearSystem::set_solution(const BatchTensor & x)
{
  _solution.variable_data().copy_(x);
}

BatchTensor
NonlinearSystem::residual(const BatchTensor & x)
{
  set_solution(x);
  residual();
  return residual_view();
}

void
NonlinearSystem::residual()
{
  assemble(true, false);

  if (_autoscale)
    _scaled_residual = scale_residual(_residual);
}

BatchTensor
NonlinearSystem::Jacobian(const BatchTensor & x)
{
  set_solution(x);
  Jacobian();
  return Jacobian_view();
}

void
NonlinearSystem::Jacobian()
{
  assemble(false, true);

  if (_autoscale)
    _scaled_Jacobian = scale_Jacobian(_Jacobian);
}

std::tuple<BatchTensor, BatchTensor>
NonlinearSystem::residual_and_Jacobian(const BatchTensor & x)
{
  set_solution(x);
  residual_and_Jacobian();
  return {residual_view(), Jacobian_view()};
}

void
NonlinearSystem::residual_and_Jacobian()
{
  assemble(true, true);

  if (_autoscale)
  {
    _scaled_residual = scale_residual(_residual);
    _scaled_Jacobian = scale_Jacobian(_Jacobian);
  }
}

BatchTensor
NonlinearSystem::residual_norm() const
{
  return math::linalg::vector_norm(residual_view());
}
} // namespace neml2
