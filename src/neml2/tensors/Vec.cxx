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
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/Rot.h"

namespace neml2
{
Vec::Vec(const Rot & r)
  : Vec(BatchTensor(r))
{
}

Vec
Vec::rotate(const Rot & r) const
{
  auto rr = r.norm_sq();
  return ((1.0 - rr) * (*this) + 2.0 * r.dot(*this) * Vec(r) - 2.0 * this->cross(r)) / (1.0 + rr);
}

R2
Vec::drotate(const Rot & r) const
{
  auto rr = r.norm_sq();
  auto rv = rotate(r);

  return 2.0 *
         (-rv.outer(r) - outer(r) + r.outer(*this) + R2::fill(r.dot(*this)) - R2::skew(*this)) /
         (1.0 + rr);
}
} // namespace neml2
