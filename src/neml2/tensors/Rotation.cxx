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

#include "neml2/tensors/Rotation.h"
#include "neml2/tensors/R3.h"

#include "neml2/misc/utils.h"

namespace neml2
{

Rotation
Rotation::init(const Scalar & r0, const Scalar & r1, const Scalar & r2)
{
  return torch::cat({r0, r1, r2}, -1);
}

Rotation
Rotation::identity()
{
  return Rotation(torch::tensor({0.0, 0.0, 0.0}, TorchDefaults));
}

Scalar
Rotation::operator()(TorchSize i) const
{
  return base_index({i}).unsqueeze(-1);
}

Rotation
Rotation::inverse() const
{
  return -torch::Tensor(*this);
}

Rotation
operator*(const Rotation & r1, const Rotation & r2)
{
  return (torch::Tensor(r1) + torch::Tensor(r2) +
          einsum({R3::init(R3::levi_civita), r1, r2}, {"ijk", "j", "k"})) /
         (1.0 - einsum({r1, r2}, {"i", "i"}).unsqueeze(-1));
}

} // namemspace neml2
