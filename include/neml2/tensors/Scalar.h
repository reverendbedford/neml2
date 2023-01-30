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

#include "neml2/tensors/FixedDimTensor.h"

namespace neml2
{
class Scalar : public FixedDimTensor<1, 1>
{
public:
  /// Forward all the constructors
  using FixedDimTensor<1, 1>::FixedDimTensor;

  /// A convenient conversion to allow people to do
  /// ~~~~~~~~~~~~~~~~~~~~cpp
  /// Scalar a = 1.0;
  /// Scalar b(5.6);
  /// Scalar c(2.3, 1000);
  /// ~~~~~~~~~~~~~~~~~~~~
  Scalar(double init, TorchSize batch_size = 1);

  /// Fill with zeros
  static Scalar zeros(TorchSize batch_size);

  /// Negation
  Scalar operator-() const;

  /// Exponentiation
  Scalar pow(Scalar n) const;

  /// The derivative of a Scalar with respect to itself
  [[nodiscard]] static Scalar identity_map() { return 1; }
};

Scalar operator+(const Scalar & a, const Scalar & b);
BatchTensor<1> operator+(const BatchTensor<1> & a, const Scalar & b);
BatchTensor<1> operator+(const Scalar & a, const BatchTensor<1> & b);

Scalar operator-(const Scalar & a, const Scalar & b);
BatchTensor<1> operator-(const BatchTensor<1> & a, const Scalar & b);
BatchTensor<1> operator-(const Scalar & a, const BatchTensor<1> & b);

Scalar operator*(const Scalar & a, const Scalar & b);
BatchTensor<1> operator*(const BatchTensor<1> & a, const Scalar & b);
BatchTensor<1> operator*(const Scalar & a, const BatchTensor<1> & b);

Scalar operator/(const Scalar & a, const Scalar & b);
BatchTensor<1> operator/(const BatchTensor<1> & a, const Scalar & b);
BatchTensor<1> operator/(const Scalar & a, const BatchTensor<1> & b);

Scalar macaulay(const Scalar & a, const Scalar & a0);
Scalar dmacaulay(const Scalar & a, const Scalar & a0);

Scalar exp(const Scalar & a);

} // namespace neml2
