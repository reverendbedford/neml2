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

#include <tuple>

#include "neml2/models/solid_mechanics/ElasticityTensor.h"

namespace neml2
{
/**
 * @brief Define an cubic symmetry elasticity tensor in various ways
 */
class CubicElasticityTensor : public ElasticityTensor
{
public:
  enum class ParamType
  {
    YOUNGS,
    POISSONS,
    SHEAR,
    INVALID
  };

  static OptionSet expected_options();

  CubicElasticityTensor(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Convert input to Lame parameter C1 with derivatives
  std::tuple<Scalar, Scalar, Scalar, Scalar> convert_to_C1();

  /// Convert input to Lame parameter C2 with derivatives
  std::tuple<Scalar, Scalar, Scalar, Scalar> convert_to_C2();

  /// Convert input to Lame parameter C3 with derivatives
  std::tuple<Scalar, Scalar, Scalar, Scalar> convert_to_C3();

  /// First input parameter
  const Scalar & _p1;

  /// First parameter type
  const ParamType _p1_type;

  /// Second input parameter
  const Scalar & _p2;

  /// Second parameter type
  const ParamType _p2_type;

  /// Third input parameter
  const Scalar & _p3;

  /// Third parameter type
  const ParamType _p3_type;
};
} // namespace neml2
