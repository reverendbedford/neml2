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

#include "neml2/drivers/TransientDriver.h"

namespace neml2
{
/**
 * @brief The transient driver specialized for solid mechanics problems.
 *
 */
class SolidMechanicsDriver : public TransientDriver
{
public:
  static ParameterSet expected_params();

  /**
   * @brief Construct a new SolidMechanicsDriver object
   *
   * @param params The parameters extracted from the input file
   */
  SolidMechanicsDriver(const ParameterSet & params);

protected:
  virtual void update_forces() override;
  void check_integrity() const override;

  /**
   * @brief The control method to drive the constitutive update.
   *
   * STRAIN: Use strain control to drive the update.
   * STRESS: Use stress control to drive the update.
   */
  std::string _control;

  /**
   * The value of the driving force, depending on `_control` this is either the prescribed strain or
   * the prescribed stress.
   */
  torch::Tensor _driving_force;

  /**
   * The name of the driving force, depending on `_control` this is either the prescribed strain or
   * the prescribed stress.
   */
  LabeledAxisAccessor _driving_force_name;
};
}
