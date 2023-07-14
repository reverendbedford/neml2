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

#include "neml2/tensors/R4.h"

namespace neml2
{

R4
R4::init(const SymSymR4 & T)
{
  return torch::cat({T(0, 0, 0, 0), T(0, 0, 0, 1), T(0, 0, 0, 2), T(0, 0, 1, 0), T(0, 0, 1, 1),
                     T(0, 0, 1, 2), T(0, 0, 2, 0), T(0, 0, 2, 1), T(0, 0, 2, 2), T(0, 1, 0, 0),
                     T(0, 1, 0, 1), T(0, 1, 0, 2), T(0, 1, 1, 0), T(0, 1, 1, 1), T(0, 1, 1, 2),
                     T(0, 1, 2, 0), T(0, 1, 2, 1), T(0, 1, 2, 2), T(0, 2, 0, 0), T(0, 2, 0, 1),
                     T(0, 2, 0, 2), T(0, 2, 1, 0), T(0, 2, 1, 1), T(0, 2, 1, 2), T(0, 2, 2, 0),
                     T(0, 2, 2, 1), T(0, 2, 2, 2), T(1, 0, 0, 0), T(1, 0, 0, 1), T(1, 0, 0, 2),
                     T(1, 0, 1, 0), T(1, 0, 1, 1), T(1, 0, 1, 2), T(1, 0, 2, 0), T(1, 0, 2, 1),
                     T(1, 0, 2, 2), T(1, 1, 0, 0), T(1, 1, 0, 1), T(1, 1, 0, 2), T(1, 1, 1, 0),
                     T(1, 1, 1, 1), T(1, 1, 1, 2), T(1, 1, 2, 0), T(1, 1, 2, 1), T(1, 1, 2, 2),
                     T(1, 2, 0, 0), T(1, 2, 0, 1), T(1, 2, 0, 2), T(1, 2, 1, 0), T(1, 2, 1, 1),
                     T(1, 2, 1, 2), T(1, 2, 2, 0), T(1, 2, 2, 1), T(1, 2, 2, 2), T(2, 0, 0, 0),
                     T(2, 0, 0, 1), T(2, 0, 0, 2), T(2, 0, 1, 0), T(2, 0, 1, 1), T(2, 0, 1, 2),
                     T(2, 0, 2, 0), T(2, 0, 2, 1), T(2, 0, 2, 2), T(2, 1, 0, 0), T(2, 1, 0, 1),
                     T(2, 1, 0, 2), T(2, 1, 1, 0), T(2, 1, 1, 1), T(2, 1, 1, 2), T(2, 1, 2, 0),
                     T(2, 1, 2, 1), T(2, 1, 2, 2), T(2, 2, 0, 0), T(2, 2, 0, 1), T(2, 2, 0, 2),
                     T(2, 2, 1, 0), T(2, 2, 1, 1), T(2, 2, 1, 2), T(2, 2, 2, 0), T(2, 2, 2, 1),
                     T(2, 2, 2, 2)},
                    -1)
      .reshape({-1, 3, 3, 3, 3});
}

Scalar
R4::operator()(TorchSize i, TorchSize j, TorchSize k, TorchSize l) const
{
  return base_index({i, j, k, l}).unsqueeze(-1);
}

SymSymR4
R4::to_symmetric() const
{
  return SymSymR4::init(*this);
}

} // namespace neml2
