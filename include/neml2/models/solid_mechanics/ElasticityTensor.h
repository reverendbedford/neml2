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

#include "neml2/models/NonlinearParameter.h"

namespace neml2
{
/**
 * @brief Superclass for defining elasticity tensors in terms of other parameters
 */
class ElasticityTensor : public NonlinearParameter<SSR4>
{
public:
  /// Various options for parameter inputs, will need to add more as we get more tensors
  enum class ParamType
  {
    YOUNGS,
    POISSONS,
    SHEAR,
    INVALID
  };

  static OptionSet expected_options();

  ElasticityTensor(const OptionSet & options);

protected:
  /// Whether to declare coefficients as parameters
  const bool _coef_as_param;

  /// Input coefficients
  const Tensor & _coef;

  /// Input coefficient types
  const std::vector<ParamType> _coef_types;

private:
  Tensor make_coef(const OptionSet & options) const;
};
} // namespace neml2
