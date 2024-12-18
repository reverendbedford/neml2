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

#include "neml2/models/solid_mechanics/elasticity/ElasticityConverter.h"

namespace neml2
{
std::string
name(ElasticConstant p)
{
  switch (p)
  {
    case ElasticConstant::BULK_MODULUS:
      return "K";
    case ElasticConstant::YOUNGS_MODULUS:
      return "E";
    case ElasticConstant::LAME_LAMBDA:
      return "lambda";
    case ElasticConstant::SHEAR_MODULUS:
      return "G";
    case ElasticConstant::POISSONS_RATIO:
      return "nu";
    case ElasticConstant::P_WAVE_MODULUS:
      return "M";
    case ElasticConstant::CUBIC_C1:
      return "C1";
    case ElasticConstant::CUBIC_C2:
      return "C2";
    case ElasticConstant::CUBIC_C3:
      return "C3";
    default:
      return "INVALID";
  }
}
} // namespace neml2
