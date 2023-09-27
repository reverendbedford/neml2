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

#include "neml2/tensors/BatchTensor.h"

namespace neml2
{
/**
 * @brief Definition of a nonlinear system of equations.
 *
 */
class NonlinearSystem
{
public:
  /// Convenient shortcut to construct and return the system residual
  virtual BatchTensor residual(const BatchTensor & in) const final;

  /// Convenient shortcut to construct and return the system Jacobian
  virtual BatchTensor Jacobian(const BatchTensor & in) const final;

  /// Convenient shortcut to construct and return the system residual and Jacobian
  virtual std::tuple<BatchTensor, BatchTensor>
  residual_and_Jacobian(const BatchTensor & in) const final;

protected:
  /**
   * @brief Compute the residual and Jacobian at the current guess \f$x\f$
   *
   * @param x The current guess of the solution
   * @param residual The current residual. The residual calculation is *requested* if it is *not* a
   * nullptr.
   * @param Jacobian The current Jacobian. The Jacobian calculation is *requested* if it is *not* a
   * nullptr.
   */
  virtual void assemble(const BatchTensor & x,
                        BatchTensor * residual,
                        BatchTensor * Jacobian = nullptr) const = 0;
};
} // namespace neml2
