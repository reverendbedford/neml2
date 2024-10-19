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

#pragma once

#include "neml2/tensors/Tensor.h"
#include "neml2/base/OptionSet.h"

namespace neml2
{
/**
 * @brief Definition of a nonlinear system of equations.
 *
 */
class NonlinearSystem
{
public:
  /**
   * Convenience struct to hold residual to prevent developers from accidentally confuse the scaled
   * and unscaled residual
   */
  template <bool scaled>
  struct Residual
  {
    Tensor value;
  };

  /**
   * Convenience struct to hold Jacobian to prevent developers from accidentally confuse the scaled
   * and unscaled residual
   */
  template <bool scaled>
  struct Jacobian
  {
    Tensor value;
  };

  /**
   * Convenience struct to hold solution to prevent developers from accidentally confuse the scaled
   * and unscaled solution (or solution increment, search direction, etc.)
   */
  template <bool scaled>
  struct Solution
  {
    Tensor value;
  };

public:
  static OptionSet expected_options();

  static void disable_automatic_scaling(OptionSet & options);

  static void enable_automatic_scaling(OptionSet & options);

  NonlinearSystem(const OptionSet & options);

  /**
   * @brief Compute algebraic Jacobian-based automatic scaling following
   * https://cs.stanford.edu/people/paulliu/files/cs517-project.pdf
   *
   * In a nutshell, given the original linearized system
   *
   * \f$ \mathrm{J} \Delta \mathrm{x} = -\mathrm{r} \f$
   *
   * Instead of solving for \f$ \Delta \mathrm{x} \f$ directly, we solve for a scaled version of it:
   *
   * \f$ \mathrm{J} \mathrm{C} \Delta \mathrm{x}' = -\mathrm{r} \f$
   *
   * where \f$ \mathrm{C} \f$ is a diagonal matrix, and apparently \f$ \Delta \mathrm{x} =
   * \mathrm{C} \Delta \mathrm{x}' \f$. Then, left-multiply both sides by another diagonal matrix
   * \f$ \mathrm{R} \f$, we get
   *
   * \f$ \mathrm{R} \mathrm{J} \mathrm{C} \Delta \mathrm{x}' = -\mathrm{R} \mathrm{r} \f$
   *
   * which is equivalent to
   *
   * \f$ \mathrm{J}' \Delta \mathrm{x}' = -\mathrm{r}' \f$
   *
   * where \f$ \mathrm{J}' = \mathrm{R} \mathrm{J} \mathrm{C} \f$ is the scaled Jacobian, and \f$
   * \mathrm{r}' = \mathrm{R} \mathrm{r} \f$ is the scaled residual. The goal of automatic scaling
   * is to find the scaling matrices so that max-norm of the rows and columns of the scaled Jacobian
   * is as close to 1 as possible.
   *
   * @param x Unscaled initial guess used to compute the initial unscaled residual and Jacobian
   * @param verbose Print automatic scaling convergence information
   */
  virtual void init_scaling(const Solution<false> & x, const bool verbose = false);

  /// Apply scaling to the residual
  Residual<true> scale(const Residual<false> & r) const;
  /// Remove scaling to the residual
  Residual<false> unscale(const Residual<true> & r) const;
  /// Apply scaling to the Jacobian
  Jacobian<true> scale(const Jacobian<false> & J) const;
  /// Remove scaling to the Jacobian
  Jacobian<false> unscale(const Jacobian<true> & J) const;
  /// Apply scaling to the solution
  Solution<true> scale(const Solution<false> & p) const;
  /// Remove scaling to the solution
  Solution<false> unscale(const Solution<true> & p) const;

  /// Set the current guess
  template <bool scaled>
  void set_guess(const Solution<scaled> & x);
  /// Convenient shortcut to set the current guess, assemble and return the residual
  template <bool scaled>
  Residual<scaled> residual(const Solution<scaled> & x);
  /// Convenient shortcut to set the current guess, assemble and return the Jacobian
  template <bool scaled>
  Jacobian<scaled> Jacobian(const Solution<scaled> & x);
  /// Convenient shortcut to set the current guess, assemble and return the residual and Jacobian
  template <bool scaled>
  std::tuple<Residual<scaled>, Jacobian<scaled>> residual_and_Jacobian(const Solution<scaled> & x);

protected:
  /// Set the _unscaled_ current guess
  virtual void set_guess(const Tensor & x) = 0;

  /**
   * @brief Compute the _unscaled_ residual and Jacobian
   *
   * @param r Pointer to the residual vector -- nullptr if not requested
   * @param J Pointer to the Jacobian matrix -- nullptr if not requested
   */
  virtual void assemble(Tensor * r, Tensor * J) = 0;

  /// If true, do automatic scaling
  const bool _autoscale;

  /// Tolerance for convergence check of the iterative automatic scaling algorithm
  const Real _autoscale_tol;

  /// Maximum number of iterations allowed for the iterative automatic scaling algorithm
  const unsigned int _autoscale_miter;

  /// Flag to indicate whether scaling matrices have been computed
  bool _scaling_matrices_initialized;

  /// Row scaling "matrix" -- since it's a batched diagonal matrix, we are only storing its diagonals
  Tensor _row_scaling;

  /// Column scaling "matrix" -- since it's a batched diagonal matrix, we are only storing its diagonals
  Tensor _col_scaling;
};

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

template <bool scaled>
void
NonlinearSystem::set_guess(const Solution<scaled> & x)
{
  if constexpr (scaled)
    set_guess(unscale(x).value);
  else
    set_guess(x.value);
}

template <bool scaled>
Residual<scaled>
NonlinearSystem::residual(const Solution<scaled> & x)
{
  Tensor r;
  set_guess(x);
  assemble(&r, nullptr);
  Residual<false> r_unscaled = {r, false};
  return scaled ? scale(r_unscaled) : r_unscaled;
}

template <bool scaled>
Jacobian<scaled>
NonlinearSystem::Jacobian(const Solution<scaled> & x)
{
  Tensor J;
  set_guess(x);
  assemble(nullptr, &J);
  Jacobian<false> J_unscaled = {J, false};
  return scaled ? scale({J, false}) : J_unscaled;
}

template <bool scaled>
std::tuple<Residual<scaled>, Jacobian<scaled>>
NonlinearSystem::residual_and_Jacobian(const Solution<scaled> & x)
{
  Tensor r, J;
  set_guess(x);
  assemble(&r, &J);
  Residual<false> r_unscaled = {r, false};
  Jacobian<false> J_unscaled = {J, false};
  return scaled ? {scale(r_unscaled), scale({J, false})} : {r_unscaled, J_unscaled};
}
} // namespace neml2
