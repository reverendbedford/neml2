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

#include "neml2/tensors/Vec.h"
#include "neml2/misc/utils.h"

namespace neml2
{

Vec
Vec::init(const Scalar & v1, const Scalar & v2, const Scalar & v3)
{
  return torch::cat({v1, v2, v3}, -1);
}

Scalar
Vec::operator()(TorchSize i) const
{
  return base_index({i});
}

Scalar
Vec::dot(const Vec & v) const
{
  return torch::linalg_vecdot(*this, v).unsqueeze(-1);
}

Vec
Vec::cross(const Vec & v) const
{
  return torch::linalg_cross(*this, v);
}

R2
Vec::outer(const Vec & v) const
{
  return einsum({*this, v}, {"i", "j"});
}

Vec
operator*(const Vec & a, const Scalar & b)
{
  return torch::operator*(a, b);
}

Vec
operator*(const Vec & a, const Real & b)
{
  return torch::operator*(a, b);
}

Vec
operator*(const Scalar & a, const Vec & b)
{
  return b * a;
}

Vec
operator*(const Real & a, const Vec & b)
{
  return b * a;
}

} // namespace neml2
