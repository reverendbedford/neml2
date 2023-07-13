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

#include "neml2/tensors/R2.h"

namespace neml2
{

R2
R2::init(const SymR2 & sym)
{
  return torch::cat({
                    sym(0,0), sym(0,1), sym(0,2),
                    sym(1,0), sym(1,1), sym(1,2),
                    sym(2,0), sym(2,1), sym(2,2)}, -1).reshape({-1,3,3});
}

R2
R2::identity()
{
  return torch::eye(3, TorchDefaults).unsqueeze(0);
}

R2
R2::zero()
{
  return torch::zeros({3, 3}, TorchDefaults).unsqueeze(0);
}

Scalar
R2::operator()(TorchSize i, TorchSize j) const
{
  return base_index({i,j}).unsqueeze(-1);
}

R2
R2::transpose() const
{
  return torch::transpose(*this, -2, -1);
}

SymR2
R2::to_symmetric() const
{
  return SymR2::init(*this);
}

R2
operator*(const R2 & A, const R2 & B)
{
  return einsum({A, B}, {"ik", "kj"});
}

Vector
operator*(const R2 & A, const Vector & b)
{
  return einsum({A, b}, {"ik", "k"});
}

R2
operator*(const R2 & A, const Scalar & b)
{
  return torch::operator*(A, b.unsqueeze(-1));
}

R2
operator*(const Scalar & a, const R2 & B)
{
  return B * a;
}

} // namespace neml2
