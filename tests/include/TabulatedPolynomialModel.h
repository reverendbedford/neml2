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

#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
/**
 * @brief This class spits out the creep strain rate along with the rate of two other internal
 * variables, given the von Mises stress, temperature, and the current internal state as input.
 *
 * This is notionally the example for the so-called LAROMANCE type of reduced order models. For
 * demonstration purposes, the rate equations are just 2nd order polynomials of the form
 *
 * \[
 * \dot{y}_k = \tilde{\delta}_{ij} \left( A^0_{ij} + A^1_{ij} x + A^2_{ij} x_k^2 \right)
 * \]
 *
 * where $x_k$ is the model input, and $\dot{y}_k$ is the model output. $A^0_{ij}$, $A^1_{ij}$, and
 * $A^2_{ij}$ are coefficient matrices. $\tilde{\delta}_{ij}$ is the regularized Kronecker delta
 * function for selecting the input domain.
 */
class TabulatedPolynomialModel : public Model
{
public:
  TabulatedPolynomialModel(const OptionSet & options);

  static OptionSet expected_options();

protected:
  void set_value(bool, bool, bool) override;

  /**
   * @brief Sigmoid-like smoothing of the Kronecker delta function
   *
   * @param x The function argument
   * @param lb Lower bounds of the table intervals
   * @param ub Upper bounds of the table intervals
   * @return torch::Tensor The smooth Kronecker delta
   */
  torch::Tensor
  smooth_index(const torch::Tensor & x, const torch::Tensor & lb, const torch::Tensor & ub) const;

  /**
   * @brief Setting up the coefficient matrices:
   *
   * Each coefficient matrix is of shape (...; 2, 3, 3, 4) -- It is vital to understand why they
   * have this particular shape!
   *
   * The last two base dimensions are 3-by-4, because we have 4 input variables: von Mises stress,
   * temperature, and two internal variables, and 3 output variables: equivalent creep strain rate
   * and rates of the two internal variables.
   *
   * The first two base dimensions are 2-by-3. They are essentially the size of the table used to
   * choose the function domain. In this example, I use von Mises stress and temperature as the two
   * axes of the table. The von Mises stress is tiled into 2 intervals, and the temperature is tiled
   * into 3 intervals. Hence the shape 2-by-3.
   *
   * In case it is not apparent, the coefficients are completely made up.
   *
   * @return BatchTensor
   */
  // @{
  BatchTensor A0() const;
  BatchTensor A1() const;
  BatchTensor A2() const;
  // @}

  /**
   * @brief Setting up the table intervals
   *
   * The von Mises stress axis has 2 intervals: [0, 50), [50, 100)
   * The temperature axis has 3 intervals: [0, 300), [300, 600), [600, 1000)
   *
   * @return BatchTensor Lower/Upper bounds for each interval
   */
  // @{
  BatchTensor s_lb() const;
  BatchTensor s_ub() const;
  BatchTensor T_lb() const;
  BatchTensor T_ub() const;
  // @}

  /// The von Mises stress
  const Variable<Scalar> & _s;

  /// Temperature
  const Variable<Scalar> & _T;

  /// Internal variables, could be wall dislocation density etc.
  const Variable<Scalar> & _s1;
  const Variable<Scalar> & _s2;

  /// Creep strain rate
  Variable<Scalar> & _ep_dot;

  /// Rate of the 1st internal state
  Variable<Scalar> & _s1_dot;

  /// Rate of the 2nd internal state
  Variable<Scalar> & _s2_dot;

  /// Polynomial coefficients
  // @{
  const BatchTensor & _A0;
  const BatchTensor & _A1;
  const BatchTensor & _A2;
  // @}

  /// Intervals of the 1st table axis: von Mises stress
  // @{
  const BatchTensor & _s_lb;
  const BatchTensor & _s_ub;
  // @}

  /// Intervals of the 2nd table axis: temperature
  // @{
  const BatchTensor & _T_lb;
  const BatchTensor & _T_ub;
  // @}

  /// Parameter controlling the sharpness of the smooth indexing
  const Real _k;
};
}
