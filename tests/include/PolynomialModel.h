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
 * demonstration purposes, the rate equations are just decoupled, 2nd order polynomials of the form
 *
 * \[
 * f = a_0 + a_1 x + a_2 x^2
 * \]
 */
class PolynomialModel : public Model
{
public:
  PolynomialModel(const OptionSet & options);

  static OptionSet expected_options();

protected:
  void set_value(bool, bool, bool) override;

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

  /// Polynomial coefficients for the creep strain rate equation
  const std::vector<Real> _s_coef;

  /// Polynomial coefficients for the 1st internal state variable
  const std::vector<Real> _s1_coef;

  /// Polynomial coefficients for the 2nd internal state variable
  const std::vector<Real> _s2_coef;
};
}
